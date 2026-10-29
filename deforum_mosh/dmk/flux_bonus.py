"""Flux-Bonus ("Built with FLUX"): FLUX.1 Kontext [dev] zeichnet die Beat-Keyframes, die dmk-Taktgrammatik bestimmt wann
und welche Welt, der optische Fluss des Videos trägt jedes Keyframe bis zum nächsten Schlag mit.

  Ton → Beats auf dem Anschlag (dmk.audio) → auf jedem Beat: Video-Frame + Welt-Prompt → Flux Kontext (ComfyUI-Bordmittel:
  UNETLoader, DualCLIPLoader, ReferenceLatent, FluxGuidance; über das Comfy SDK) → zwischen den Beats wird das letzte
  Keyframe entlang der Videobewegung (DIS-Fluss, CPU) weitergetragen → Film-Finish → MP4 mit Ton.

Weltwechsel auf jeder Eins (jede Welt mit eigenem festen Seed), auf den übrigen Beats zeichnet Flux dieselbe Welt neu aus
dem aktuellen Video-Frame – ein Schnitt auf dem Schlag, die Komposition bleibt die des Videos.

Lizenz: FLUX.1 Kontext [dev] steht unter der FLUX.1 [dev] Non-Commercial License (nur nicht-kommerzielle Nutzung).

    python -m dmk.flux_bonus <video> <ausgabeordner> --sek 6 --start 87.5 --welten "prompt1|prompt2|..."
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import audio, core, method_a, proc

UNET = "flux1-dev-kontext_fp8_scaled.safetensors"
T5, CLIP_L, VAE = "t5xxl_fp8_e4m3fn_scaled.safetensors", "clip_l.safetensors", "ae.safetensors"
# Prompttest 26.10. (dmk.flux_prompt_test): "Change ... while keeping the same view/framing/outline" hält die Komposition
# am besten (Kantenkorrelation 0,31 gegen 0,08–0,18 für "Restyle as ..." bzw. "Cover ..."), Guidance 1,5 besser als 2,5
ANWEISUNG = ("Change everything in this image to {welt}, while keeping the same camera view, the same framing and the "
             "exact position and outline of every shape. Natural light, photograph.")
GUIDANCE = 1.5


def workflow(bild: str, prompt: str, seed: int, steps: int = 20, guidance: float = 2.5) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": CLIP_L, "clip_name2": T5, "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "LoadImage", "inputs": {"image": bild}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "7": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["5", 0]}},
        "8": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["7", 0], "guidance": guidance}},
        "9": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "10": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["8", 0], "negative": ["9", 0],
                                                    "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "dmk_flux"}},
    }


def fluss(dis, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rückwärtsfluss: für jedes Pixel in b, wo liegt es in a (Video-Frames, Graustufen)."""
    return dis.calc(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)


def render(video: Path, out: Path, sek: float, start: float, welten: list[str], track: Path, breite: int = 768,
           hoehe: int = 432, steps: int = 20, film: float = 1.0, log=print, dry: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    fdir = out / "frames"
    fdir.mkdir(exist_ok=True)
    qdir = out / "quelle"
    if not qdir.exists() or len(core.list_images(qdir)) < int(sek * 30):
        qdir.mkdir(exist_ok=True)
        proc.run([proc.tool("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-t", f"{sek:.3f}", "-vf",
                  f"fps=30,scale={breite}:{hoehe}:force_original_aspect_ratio=increase,crop={breite}:{hoehe}",
                  str(qdir / "%05d.png")], "Video zerlegen")
    quellen = core.list_images(qdir)
    n = min(len(quellen), int(round(sek * 30)))
    y = audio.dekodieren(track, audio.SR)
    mm, info = audio.merkmale(y, audio.SR, n, 1 / 30, start, log=lambda s: None)
    beats = [int(b) for b in info["beat_frames"]]
    takte = set(int(t) for t in info["takt_frames"])
    keys = sorted(set([0] + beats))
    # Welt pro Keyframe: Wechsel auf jeder Eins
    welt_idx, w, letzter = {}, 0, 0
    for k in keys:
        if k in takte and k - letzter >= 30:   # frühestens 1 s nach dem letzten Wechsel (sonst steht Welt 1 nur Frames)
            w, letzter = w + 1, k
        welt_idx[k] = w % len(welten)
    log(f"[flux] {n} frames, {info['bpm']} bpm, {len(keys)} keyframes (Flux calls), world changes on {sorted(takte)}")
    comfy = None if dry else method_a.make_client("127.0.0.1:8188", "sdk", "http://127.0.0.1:8189")
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ys, xs = np.mgrid[0:hoehe, 0:breite].astype(np.float32)
    t0 = time.time()
    key_bild, key_quelle, aufrufe = None, None, []
    for i in range(n):
        src = core.imread(quellen[i])
        if i in welt_idx:
            k_png = out / "keys" / f"{i:05d}.png"
            if k_png.exists():
                img = core.imread(k_png)
            elif dry:
                img = src.copy()
            else:
                wi = welt_idx[i]
                prompt = ANWEISUNG.format(welt=welten[wi])
                name = comfy.upload(src, "dmk_flux_in.png")
                tk = time.time()
                img = comfy.image(comfy.wait(comfy.queue(workflow(name, prompt, 1000 + wi, steps, GUIDANCE)), timeout=1800))
                if img.shape[:2] != (hoehe, breite):
                    img = cv2.resize(img, (breite, hoehe), interpolation=cv2.INTER_LANCZOS4)
                aufrufe.append(round(time.time() - tk, 1))
                k_png.parent.mkdir(exist_ok=True)
                core.imwrite(k_png, img)
                log(f"[flux] keyframe {i} world {wi}: {aufrufe[-1]} s")
            key_bild, key_quelle = img, src
            frame = img
        else:
            # Keyframe entlang der Videobewegung mitnehmen: Pixel von Frame i liegt im Keyframe-Quellframe bei +flow
            f = fluss(dis, key_quelle, src)
            frame = cv2.remap(key_bild, xs + f[..., 0], ys + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        core.imwrite(fdir / f"{i:05d}.png", core.film_finish(frame, i, film, False, 7))
    if comfy is not None:
        comfy.uploads_aufraeumen()
    core.frames_to_mp4(fdir, out / "video.mp4", 30)
    audio.ton_anlegen(out / "video.mp4", track, out / "video_ton.mp4", start, n / 30, n / 30)
    meta = {"frames": n, "keyframes": len(keys), "flux_aufrufe_s": aufrufe, "minuten": round((time.time() - t0) / 60, 1),
            "bpm": info["bpm"], "welten": welten, "start": start, "video": str(video), "steps": steps,
            "modell": UNET, "lizenz": "FLUX.1 [dev] Non-Commercial License"}
    (out / "flux_run.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"[flux] done: {meta['minuten']} min, {len(aufrufe)} Flux calls, mean {np.mean(aufrufe) if aufrufe else 0:.0f} s")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--sek", type=float, default=6)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--track", default=str(Path(__file__).resolve().parents[2] / "examples" / "signal_decay_20s.m4a"))
    ap.add_argument("--welten", required=True, help="prompts separated by |")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    render(Path(a.video), Path(a.out), a.sek, a.start, [w.strip() for w in a.welten.split("|") if w.strip()], Path(a.track),
           steps=a.steps, dry=a.dry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
