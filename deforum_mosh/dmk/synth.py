"""dmk.synth: synthetische Kamerafahrt mit Tiefenkarten als Referenz- und Beispielsequenz."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import core


def _world(w: int, h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    for octave in range(2, 8):
        small = rng.random((max(2, h >> octave), max(2, w >> octave)), dtype=np.float32)
        acc += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * (2.0 ** octave)
    acc = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.applyColorMap(acc, cv2.COLORMAP_TWILIGHT_SHIFTED)
    cx, cy = w // 2, h // 2
    for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):  # Fluchtlinien zur Mitte
        end = (int(cx + np.cos(a) * w), int(cy + np.sin(a) * w))
        cv2.line(img, (cx, cy), end, (235, 235, 235), 3, cv2.LINE_AA)
    for _ in range(260):
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        col = tuple(int(c) for c in rng.integers(30, 255, 3))
        r = int(rng.integers(6, 60))
        if rng.random() < 0.5:
            cv2.circle(img, p, r, col, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, p, (p[0] + r * 2, p[1] + r), col, -1)
    grain = rng.integers(-10, 10, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)


def make(out_dir, frames: int = 40, size=(768, 432), zoom: float = 1.02, pan: float = 1.5, seed: int = 7) -> Path:
    """Zoomfahrt in eine feste Welt: `zoom` pro Frame, dazu ein leichter Schwenk."""
    out_dir = Path(out_dir)
    w, h = size
    total = zoom ** (frames - 1)
    ww, wh = int(w * total * 1.15) + 2 * int(pan * frames), int(h * total * 1.15)
    world = _world(ww, wh, seed)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xs - (w - 1) / 2, ys - (h - 1) / 2)
    depth = (r / r.max() * 65535).astype(np.uint16)  # Tunnel: Mitte fern (dunkel), Rand nah (hell)
    for i in range(frames):
        s = total / zoom ** i  # Weltpixel pro Bildpixel
        cw, ch = w * s, h * s
        x0 = (ww - cw) / 2 + pan * (i - frames / 2)
        y0 = (wh - ch) / 2
        m = np.float32([[s, 0, x0], [0, s, y0]])
        img = cv2.warpAffine(world, m, (w, h), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)
        core.imwrite(out_dir / f"{i:05d}.png", img)
        core.imwrite(out_dir / "depth" / f"{i:05d}.png", depth)
    return out_dir
