"""dmk.depth: Tiefenkarten schätzen (Depth-Anything-V2-Small), wenn die Sequenz keine mitbringt.

Ausgabe: 16-bit-PNG pro Frame, Konvention hell = nah (wie read_depth erwartet). Die Karten werden
projektlokal unter <cache_dir>/<sequenzname>/ abgelegt, nie in den Quellordner geschrieben.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import core

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_PIPE = None  # einmal geladen, im Prozess wiederverwendet


def _pipe():
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    # liegt das Modell schon im Cache, ohne Netz laden: am 27.10. hing ein Lauf 3,5 h in der Update-Prüfung des Hubs
    # (offene HTTPS-Verbindung zum CDN, kein Fortschritt, 0 % CPU)
    try:
        from huggingface_hub import snapshot_download
        lokal = snapshot_download(MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001  (noch nicht geladen: erster Start lädt herunter)
        lokal = MODEL
    return pipeline("depth-estimation", model=lokal, device=device)


def estimate(seq: core.Sequence, cache_dir, max_side: int = 768, log=print) -> list[Path]:
    """Schätzt für jeden Frame eine Tiefenkarte; vorhandene Karten im Cache werden wiederverwendet."""
    out = Path(cache_dir) / seq.name
    out.mkdir(parents=True, exist_ok=True)
    global _PIPE
    pipe = None
    paths = [out / (p.stem + "_depth.png") for p in seq.frames]
    # dict statt Liste: ein Einzelbild steht n-mal in seq.frames, geschätzt wird es nur einmal
    todo = list({d: (f, d) for f, d in zip(seq.frames, paths) if not d.exists()}.values())
    if not todo:
        return paths
    log(f"[depth] estimating {len(todo)} depth maps with {MODEL} -> {out}")
    from PIL import Image
    if _PIPE is None:
        _PIPE = _pipe()
    pipe = _PIPE
    t0 = time.time()
    for k, (f, d) in enumerate(todo):
        img = core.imread(f, max_side)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = pipe(pil)
        depth = res["predicted_depth"]
        depth = depth.squeeze().float().cpu().numpy() if hasattr(depth, "cpu") else np.asarray(res["depth"], np.float32)
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        lo, hi = float(depth.min()), float(depth.max())
        d16 = ((depth - lo) / (hi - lo + 1e-6) * 65535).astype(np.uint16)  # Depth Anything: groß = nah
        core.imwrite(d, d16)
        if (k + 1) % 25 == 0 or k + 1 == len(todo):
            log(f"[depth] {k + 1}/{len(todo)}  {(time.time() - t0) / (k + 1):.2f} s/frame")
    return paths


def schaetze(img: np.ndarray) -> np.ndarray:
    """Tiefe eines Bildes im Speicher (BGR) als float32 0..1, hell = nah, in Bildgröße. Für Anker der endlosen Fahrt:
    das eigene generierte Bild wird zur neuen Quelle und braucht eine eigene Tiefe."""
    global _PIPE
    from PIL import Image
    if _PIPE is None:
        _PIPE = _pipe()
    res = _PIPE(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    d = res["predicted_depth"]
    d = d.squeeze().float().cpu().numpy() if hasattr(d, "cpu") else np.asarray(res["depth"], np.float32)
    d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(d.min()), float(d.max())
    return ((d - lo) / (hi - lo + 1e-6)).astype(np.float32)


def raum_tiefe(out_frames, src_depth, cache_dir, sub: int = 1, step: int = 4, side: int = 320, log=print,
               mask=None) -> float:
    """Räumlichkeit: Pearson-Korrelation der geschätzten Tiefe des Ergebnisses mit der Tiefe der Quelle,
    über jeden step-ten Frame. 1.0 = gleiche Raumstruktur, ~0 = Raum verloren. Skaleninvariant, und
    anders als ein Kantenvergleich unempfindlich gegen neue Textur.
    mask: Bild/Array (weiß = gültig) – nur dort vergleichen. Bei maskiertem Material (z. B. Blende außen
    schwarz, in der Blender-Tiefe aber als nahe Ebene) verfälscht der Außenbereich sonst die Korrelation."""
    out_frames = list(out_frames)
    idx = list(range(0, len(out_frames), step))
    probe = core.Sequence(root=Path(out_frames[0]).parent, frames=[out_frames[i] for i in idx])
    est = estimate(probe, cache_dir, log=log)
    valid = None
    if mask is not None:
        m = core.imread(mask, gray=True) if isinstance(mask, (str, Path)) else np.asarray(mask)
        valid = cv2.resize(m.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA) > 127
        if valid.mean() < 0.05:
            valid = None
    cors = []
    for i, d_out in zip(idx, est):
        d_src = src_depth[min(round(i / sub), len(src_depth) - 1)]
        a = core.read_depth(d_out, (side, side))
        b = core.read_depth(d_src, (side, side))
        if valid is not None:
            a, b = a[valid], b[valid]
        a, b = a - a.mean(), b - b.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-6
        cors.append(float((a * b).sum() / den))
    return float(np.mean(cors))


def freigeben() -> None:
    """Tiefenmodell aus RAM/VRAM entfernen (wird nur zu Beginn eines Laufs gebraucht)."""
    global _PIPE
    if _PIPE is None:
        return
    _PIPE = None
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def ensure(seq: core.Sequence, cache_dir, log=print) -> core.Sequence:
    """Gibt die Sequenz mit Tiefenkarten zurück (vorhanden oder geschätzt)."""
    if seq.depth:
        return seq
    seq.depth = estimate(seq, cache_dir, log=log)
    return seq
