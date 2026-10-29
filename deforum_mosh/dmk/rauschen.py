"""dmk.rauschen: Rausch-Transport für den Sampler (nach ∫-Noise, Chang et al., ICLR 2024).

Das Startrauschen des Samplers wird nicht pro Frame gewürfelt (jeder Durchgang erfindet die Textur neu →
"Springen") und nicht festgehalten (Textur klebt am Bildschirm, brennt ein), sondern als Rauschfeld in
PIXELauflösung entlang der Bewegung mitgeführt. Für das Latent (1/8) wird das Feld blockweise gemittelt und auf
Einheitsvarianz gebracht; aufgedeckte oder unglaubwürdige Bereiche werden frisch gewürfelt.

Einspeisung in ComfyUI nur mit Kern-Knoten: Rauschen als .latent-Datei hochladen (LoadLatent), mit σ₀ skalieren
(LatentMultiply), auf das Startlatent addieren (LatentAdd) und mit DisableNoise durch SamplerCustomAdvanced.
"""
from __future__ import annotations

import io
import math

import cv2
import numpy as np

LATENT_CH = 4
BLOCK = 8  # Pixel pro Latent-Zelle bei SD 1.5 / SDXL
SCALE_SD15 = 0.18215  # latent_format.scale_factor; SDXL wäre 0.13025


class NoiseTransport:
    def __init__(self, h: int, w: int, seed: int = 0, ch: int = LATENT_CH):
        self.h, self.w, self.ch = h, w, ch
        self.rng = np.random.default_rng(seed)
        self.field = self.rng.standard_normal((ch, h, w), dtype=np.float32)
        self.frisch_anteil = 0.0

    def advect(self, mx: np.ndarray, my: np.ndarray, frisch: np.ndarray | None = None, rho: float = 0.0) -> None:
        """mx, my: für jeden Pixel des neuen Frames die Position im vorigen Frame (wie bei cv2.remap).
        frisch: 0..1, Anteil, zu dem der Pixel neu gewürfelt wird (Verdeckung, unglaubwürdiger Fluss).
        rho: AR-Mischung ε = √(1−ρ²)·W(ε) + ρ·ξ, varianzerhaltend. ρ=0 reiner Transport (brennt wie ein fester
        Seed ein: Fixpunkt-Iteration), ρ=1 reines Würfeln (springt). Dazwischen der Regler für den Wandel."""
        xi = np.rint(mx).astype(np.int32)
        yi = np.rint(my).astype(np.int32)
        inside = (xi >= 0) & (xi < self.w) & (yi >= 0) & (yi < self.h)
        xi, yi = np.clip(xi, 0, self.w - 1), np.clip(yi, 0, self.h - 1)
        neu = self.field[:, yi, xi]  # nächster Nachbar: hält die Unabhängigkeit der Samples
        fresh = self.rng.standard_normal(neu.shape, dtype=np.float32)
        mask = (~inside).astype(np.float32)
        if frisch is not None:
            mask = np.maximum(mask, np.clip(frisch, 0, 1).astype(np.float32))
        # harte Entscheidung pro Pixel statt Mischung, sonst sinkt die Varianz
        wuerfel = self.rng.random((self.h, self.w), dtype=np.float32) < mask
        self.field = np.where(wuerfel[None], fresh, neu)
        self.frisch_anteil = float(wuerfel.mean())
        rho = float(np.clip(rho, 0.0, 1.0))
        if rho > 0:
            self.field = (math.sqrt(1.0 - rho * rho) * self.field
                          + rho * self.rng.standard_normal(self.field.shape, dtype=np.float32))

    def latent(self) -> np.ndarray:
        """Rauschen für das Latent: Blockmittel ×√N (Varianzerhalt bei unabhängigen Samples), dann lokale
        Nachnormierung, weil Duplikate aus dem Nächste-Nachbar-Warp die Varianz verschieben. Form (1, ch, h/8, w/8)."""
        ch, h, w = self.field.shape
        hb, wb = h // BLOCK, w // BLOCK
        f = self.field[:, :hb * BLOCK, :wb * BLOCK].reshape(ch, hb, BLOCK, wb, BLOCK)
        eps = f.mean(axis=(2, 4)) * BLOCK
        out = np.empty_like(eps)
        for c in range(ch):
            m = cv2.blur(eps[c], (5, 5))
            v = cv2.blur(eps[c] * eps[c], (5, 5)) - m * m
            out[c] = (eps[c] - m) / np.sqrt(np.maximum(v, 1e-4)) * 1.0
        out -= out.mean(axis=(1, 2), keepdims=True)
        out /= out.std(axis=(1, 2), keepdims=True) + 1e-6
        return out[None].astype(np.float32)


def latent_datei(eps: np.ndarray) -> bytes:
    """Serialisiert ein Latent so, wie ComfyUIs LoadLatent es erwartet (safetensors, Schlüssel latent_tensor)."""
    from safetensors.numpy import save
    return save({"latent_tensor": np.ascontiguousarray(eps, dtype=np.float32),
                 "latent_format_version_0": np.zeros((0,), dtype=np.float32)})


# ---------------------------------------------------------------- σ₀ wie ComfyUI

def _sd15_sigma_bounds() -> tuple[float, float]:
    """sigma_min/max des SD-1.5-Rauschplans (linear betas 0.00085..0.012, 1000 Schritte), wie model_sampling."""
    betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float64) ** 2
    ac = np.cumprod(1.0 - betas)
    sig = np.sqrt((1.0 - ac) / ac)
    return float(sig[0]), float(sig[-1])


def karras_sigmas(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0) -> np.ndarray:
    ramp = np.linspace(0, 1, n)
    lo, hi = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    return np.append((hi + ramp * (lo - hi)) ** rho, 0.0)


def sigma0(steps: int, denoise: float, scheduler: str = "karras") -> float:
    """Erste Sigma des Plans, den BasicScheduler(steps, denoise) liefert: Plan über int(steps/denoise)
    Schritte, davon die letzten steps+1 Werte. Nur karras/normal werden nachgebildet."""
    smin, smax = _sd15_sigma_bounds()
    total = steps if denoise >= 1.0 else int(steps / denoise)
    if scheduler == "karras":
        sig = karras_sigmas(total, smin, smax)
    else:  # "normal": gleichmäßig in Timesteps, Näherung über lineare Sigma-Interpolation im Timestep-Raum
        betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
        ac = np.cumprod(1.0 - betas)
        all_sig = np.sqrt((1.0 - ac) / ac)
        ts = np.linspace(999, 0, total + 1)[:-1]
        sig = np.append(np.interp(ts, np.arange(1000), all_sig), 0.0)
    return float(sig[-(steps + 1)])
