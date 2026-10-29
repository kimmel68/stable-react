"""dmk.anker: Kennzahlen gegen die REFERENZbewegung (nicht gegen die eigene) – nach der Kritik des unabhängigen
Reviews 2026-10-09: erhalt_k/flow_jitter messen entlang des Flusses des Ergebnisses und übersehen Inhalt, der in
sich glatt fließt, aber gegenüber der Szene rutscht; mitfluss vergleicht mit dem Modell, mit dem auch getragen wird.

ankertreue_{band}_{τ}  NCC des Bandpasses (fein <2 px, mittel 2–8 px, grob 8–32 px, bei 768 px Breite) zwischen
                       Frame t und t+τ·fps, nachdem t+τ entlang der Quellbewegung (DIS-Fluss der Quelle, nur wo
                       glaubwürdig) auf t zurückgetragen wurde. Deforum-Vorlage grob: 0.87 (0,5 s), 0.78 (1 s).
pop_rate               Flecken pro Sekunde, die nach Kompensation mit der Quellbewegung im Mittelband neu
                       auftauchen oder verschwinden (> 3σ, ≥ 0,1 % Bildfläche), ohne Verdeckungsbereiche.
pump                   (max − min der Laplace-Varianz je Cadence-Phase) / Mittel; Schärfe-Pumpen im Cadence-Takt.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import core

BANDS = {"fein": (0.0, 2.0), "mittel": (2.0, 8.0), "grob": (8.0, 32.0)}


def _dis():
    d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    d.setUseSpatialPropagation(True)
    return d


def _band(gray: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Bandpass: Strukturen zwischen lo und hi Pixeln (Differenz zweier Gauß-Tiefpässe; lo=0 = ungeglättet)."""
    g = gray.astype(np.float32)
    fine = g if lo <= 0 else cv2.GaussianBlur(g, (0, 0), lo)
    return fine - cv2.GaussianBlur(g, (0, 0), hi)


def _ncc(a, b, mask):
    a, b = a[mask], b[mask]
    if a.size < 100:
        return None
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def messen(out_frames, src_frames, fps: float = 24.0, taus=(0.5, 1.0), side: int = 768, cadence: int = 1,
           ref_bewegung=None, mask=None, mask_rand: int = 12) -> dict:
    """out_frames und src_frames müssen deckungsgleich sein (Ausgabeframe i ↔ Quellframe i).
    ref_bewegung(i, (w, h)) -> (bwd, gueltig) liefert optional die EXAKTE Referenzbewegung für das Paar i→i+1
    (z. B. Blender-Vector-Pass über passes.PassSequenz.rueckabbildung): bwd = Versatz der Pixel von Frame i+1 zu
    ihrer Position in Frame i; gueltig = nicht verdeckt/aufgedeckt. Ohne: DIS-Fluss auf der Quelle mit Konfidenz."""
    n = min(len(out_frames), len(src_frames))
    outs = [core._small(core.imread(p, gray=True), side) for p in out_frames[:n]]
    h, w = outs[0].shape
    # Quelle exakt auf die Ausgabegröße bringen (prep() schneidet auf Vielfache von 8, die Quelle nicht)
    srcs = [cv2.resize(core.imread(p, gray=True), (w, h), interpolation=cv2.INTER_AREA) for p in src_frames[:n]]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dis = _dis()
    # Feste Maske (Projektionsmapping): die Maskenkanten bewegen sich NICHT mit der Szene; ein Rand um sie herum
    # wird von der Messung ausgenommen, sonst zählen die festen Fensterlöcher als "springender Inhalt".
    gueltig = None
    if mask is not None:
        m = core.imread(mask, gray=True) if isinstance(mask, (str, Path)) else np.asarray(mask)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) > 127
        k = 2 * mask_rand + 1
        gueltig = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    # Referenzbewegung (t+1 -> t) und Glaubwürdigkeit pro Paar
    flows, confs = [], []
    for i in range(n - 1):
        ref = ref_bewegung(i, (w, h)) if ref_bewegung is not None else None
        if ref is not None:
            bwd, conf = ref
            conf = np.asarray(conf, bool) & (srcs[i + 1] > 4)
            if gueltig is not None:
                conf &= gueltig
            flows.append(np.ascontiguousarray(bwd, dtype=np.float32))
            confs.append(conf)
            continue
        bwd = dis.calc(srcs[i + 1], srcs[i], None)  # Position in i für Pixel aus i+1
        fwd = dis.calc(srcs[i], srcs[i + 1], None)
        f_at = cv2.remap(fwd, xs + bwd[..., 0], ys + bwd[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = np.linalg.norm(bwd + f_at, axis=2)
        g = srcs[i + 1].astype(np.float32)
        tex = cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 3)
        conf = (err < 1.0) & (tex > np.percentile(tex, 40))
        if gueltig is not None:
            conf &= gueltig
        flows.append(bwd)
        confs.append(conf)

    res = {}
    for tau in taus:
        k = max(1, int(round(tau * fps)))
        if k >= n - 1:
            continue
        acc = {b: [] for b in BANDS}
        for t in range(0, n - k, max(1, k // 2)):
            mx, my, ok = xs.copy(), ys.copy(), np.ones((h, w), bool)
            for j in range(k, 0, -1):  # von t+k zurück nach t: Position in t+j-1 der Pixel aus t+k
                f = flows[t + j - 1]
                fx = cv2.remap(f[..., 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                fy = cv2.remap(f[..., 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                c = cv2.remap(confs[t + j - 1].astype(np.float32), mx, my, cv2.INTER_NEAREST) > 0.5
                ok &= c
                mx, my = mx + fx, my + fy
            # mx,my: wo liegt der Inhalt von Frame t+k im Frame t?  -> t+k auf t zurücktragen heißt: t an (mx,my) lesen
            inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
            mask = ok & inside
            if mask.mean() < 0.05:
                continue
            back = cv2.remap(outs[t].astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            for b, (lo, hi) in BANDS.items():
                v = _ncc(_band(outs[t + k], lo, hi), _band(back, lo, hi), mask)
                if v is not None:
                    acc[b].append(v)
        for b in BANDS:
            res[f"anker_{b}_{tau:g}s"] = float(np.mean(acc[b])) if acc[b] else None

    pops, areas = [], []
    for i in range(n - 1):
        f = flows[i]
        back = cv2.remap(outs[i].astype(np.float32), xs + f[..., 0], ys + f[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        d = _band(outs[i + 1], 2.0, 8.0) - _band(back, 2.0, 8.0)
        d = np.abs(cv2.GaussianBlur(d, (0, 0), 2))
        valid = confs[i]
        thr = 3.0 * (d[valid].std() + 1e-3)
        bw = ((d > thr) & valid).astype(np.uint8)
        cnt, _, stats, _ = cv2.connectedComponentsWithStats(bw)
        big = stats[1:, cv2.CC_STAT_AREA] >= 0.001 * h * w
        pops.append(int(big.sum()))
        areas.extend((stats[1:, cv2.CC_STAT_AREA][big] / (h * w)).tolist())
    res["pop_rate"] = float(np.mean(pops) * fps) if pops else None
    res["pop_flaeche_p95"] = float(np.percentile(areas, 95)) if areas else 0.0

    lap = np.array([cv2.Laplacian(o, cv2.CV_32F).var() for o in outs])
    if cadence > 1 and len(lap) >= 2 * cadence:
        ph = [lap[p::cadence].mean() for p in range(cadence)]
        res["pump"] = float((max(ph) - min(ph)) / (np.mean(ph) + 1e-6))
    else:
        res["pump"] = float(np.abs(np.diff(lap)).mean() / (lap.mean() + 1e-6))
    return res
