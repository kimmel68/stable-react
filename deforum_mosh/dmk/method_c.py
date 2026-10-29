"""dmk.method_c: Flow-Advektion.

Das vorherige Ausgabebild wird mit dem optischen Fluss der Quelle weitergetragen; Flow und
Zoom landen als ein gemeinsames Feld in einem einzigen Resampling-Schritt.

    adv   = resample(out_{n-1}, zoom + flow)
    out_n = trail * adv + (1 - trail) * input_n       (außerhalb der Occlusion)
    Occlusion:  hold -> adv (Smear, Datamosh ohne Codec)   fill -> input_n
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from . import core


class FarnebackFlow:
    name = "farneback"

    def pair(self, a, b):
        ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        return core.farneback(ga, gb), core.farneback(gb, ga)


class RaftFlow:
    name = "raft"

    def __init__(self, size: str = "large"):
        import torch
        from torchvision.models import optical_flow as of
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA nicht verfügbar")
        self.torch = torch
        if size == "small":
            model, weights = of.raft_small, of.Raft_Small_Weights.DEFAULT
        else:
            model, weights = of.raft_large, of.Raft_Large_Weights.DEFAULT
        self.model = model(weights=weights, progress=False).eval().cuda()

    def _tensor(self, img):
        t = self.torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).cuda()
        return t.permute(2, 0, 1).float().div(127.5).sub(1.0)

    def pair(self, a, b):
        torch = self.torch
        h, w = a.shape[:2]
        ta, tb = self._tensor(a), self._tensor(b)
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        x1, x2 = torch.stack([ta, tb]), torch.stack([tb, ta])
        if ph or pw:
            pad = lambda x: torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")
            x1, x2 = pad(x1), pad(x2)
        with torch.no_grad():
            flow = self.model(x1, x2, num_flow_updates=12)[-1][:, :, :h, :w]
        flow = flow.permute(0, 2, 3, 1).float().cpu().numpy()
        return flow[0], flow[1]


def make_backend(name: str = "auto", log=print):
    if name in ("auto", "raft", "raft_small"):
        try:
            be = RaftFlow("small" if name == "raft_small" else "large")
            return be
        except Exception as e:  # fehlendes torch, kein CUDA, Gewichte nicht ladbar, OOM
            if name != "auto":
                raise
            log(f"[C] RAFT nicht verfügbar ({type(e).__name__}: {e}), nehme Farnebäck.")
    return FarnebackFlow()


def _flow_pair(backend, a, b, flow_side):
    """Flow vorwärts (a->b) und rückwärts (b->a), bei Bedarf in reduzierter Auflösung gerechnet."""
    h, w = a.shape[:2]
    s = flow_side / max(h, w) if flow_side and max(h, w) > flow_side else 1.0
    if s < 1.0:
        size = (round(w * s), round(h * s))
        a, b = (cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in (a, b))
    fwd, bwd = backend.pair(a, b)
    if s < 1.0:
        fwd, bwd = (cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) / s for f in (fwd, bwd))
    return fwd.astype(np.float32), bwd.astype(np.float32)


def flow_konfidenz(src_bgr, fwd, bwd, xs, ys) -> np.ndarray:
    """Wo ist der geschätzte Fluss glaubwürdig? Textur vorhanden UND vorwärts/rückwärts stimmig (0..1)."""
    g = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    grad = cv2.GaussianBlur(grad, (0, 0), 4)
    tex = np.clip(grad / (np.percentile(grad, 70) + 1e-3), 0, 1)
    f_at = cv2.remap(fwd, xs + bwd[..., 0], ys + bwd[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.linalg.norm(bwd + f_at, axis=2)
    return (tex * np.exp(-(err / 1.5) ** 2)).astype(np.float32)


def kamera_fluss(flow, depth, conf, iters: int = 3) -> np.ndarray:
    """Flussfeld einer Kamerafahrt durch eine ruhende Szene als glattes Modell in Bildkoordinaten und Tiefe:
    Basis [1, x, y, D, xD, yD, x², xy, y²] je Komponente (Schwenk, Zoom, Drehung, Perspektive und
    tiefenabhängige Parallaxe). Robust (Huber) an die glaubwürdigen Flusswerte angepasst und dann für ALLE
    Bildpunkte ausgewertet, also auch für glatte Flächen, auf denen die Bildschätzung nichts findet."""
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = (xs - (w - 1) / 2) / w, (ys - (h - 1) / 2) / w
    basis = np.stack([np.ones_like(x), x, y, depth, x * depth, y * depth, x * x, x * y, y * y], -1).reshape(-1, 9)
    idx = np.arange(0, h * w, 5)
    bs, w0 = basis[idx], conf.reshape(-1)[idx].astype(np.float32) + 1e-4
    out = np.empty((h * w, 2), np.float32)
    for c in range(2):
        t = flow[..., c].reshape(-1)[idx]
        wgt = w0.copy()
        coef = np.zeros(9, np.float32)
        for _ in range(iters):
            sw = np.sqrt(wgt)
            coef, *_ = np.linalg.lstsq(bs * sw[:, None], t * sw, rcond=None)
            r = np.abs(bs @ coef - t)
            s = 1.48 * float(np.median(r[w0 > 0.2])) + 1e-2 if (w0 > 0.2).any() else 1.0
            wgt = w0 * np.minimum(1.0, 2.0 * s / (r + 1e-6))
        out[:, c] = basis @ coef
    return out.reshape(h, w, 2)


def occlusion(fwd, bwd, xs, ys, thresh: float) -> np.ndarray:
    """Vorwärts-Rückwärts-Check im Raster von Frame n: 1.0 = verdeckt/neu aufgedeckt."""
    f_at = cv2.remap(fwd, xs + bwd[..., 0], ys + bwd[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.linalg.norm(bwd + f_at, axis=2)
    mag = np.linalg.norm(bwd, axis=2) + np.linalg.norm(f_at, axis=2)
    mask = (err > thresh + 0.05 * mag).astype(np.float32)
    return cv2.GaussianBlur(mask, (0, 0), 1.5)


def run(seq_root, out_dir, count=0, start=0, zoom="0:(1.02)", max_side=768, fps=24,
        sub=1, trail=0.85, occl="hold", occl_thresh=1.5, backend="auto", flow_side=768,
        hf=0.0, hf_sigma=2.0, border="reflect", pattern=None, log=print, **_):
    t0 = time.time()
    seq = core.load_sequence(seq_root, count, start, pattern)
    n = len(seq)
    zs = core.schedule(zoom, n)
    fdir = core.clear_frames(out_dir)
    be = make_backend(backend, log)
    log(f"[C] Flow-Backend: {be.name}, sub={sub}, occl={occl}")

    prev_in = core.prep(core.imread(seq.frames[0]), max_side, 2)
    h, w = prev_in.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    mode = core.BORDERS[border]
    sub = max(1, int(sub))
    trail_sub = float(trail) ** (1.0 / sub)

    out = prev_in.astype(np.float32)
    core.imwrite(fdir / "00000.png", prev_in)
    k_out = 1
    for i in range(1, n):
        inp = core.prep(core.imread(seq.frames[i]), max_side, 2)
        if inp.shape[:2] != (h, w):
            inp = cv2.resize(inp, (w, h), interpolation=cv2.INTER_AREA)
        try:
            fwd, bwd = _flow_pair(be, prev_in, inp, flow_side)
        except RuntimeError as e:  # z. B. CUDA out of memory mitten im Lauf
            if be.name == "farneback":
                raise
            log(f"[C] {be.name} fällt aus ({e}), weiter mit Farnebäck.")
            be = FarnebackFlow()
            fwd, bwd = _flow_pair(be, prev_in, inp, flow_side)
        occ = occlusion(fwd, bwd, xs, ys, occl_thresh)[..., None] if occl != "none" else None

        z_sub = float(zs[i]) ** (1.0 / sub)
        # Zoom als radiales Feld; der Flow wird an der entzoomten Position abgelesen
        zx, zy = cx + (xs - cx) / z_sub, cy + (ys - cy) / z_sub
        b_sub = cv2.remap(bwd, zx, zy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) / sub
        mx, my = zx + b_sub[..., 0], zy + b_sub[..., 1]
        for k in range(1, sub + 1):
            t = k / sub
            adv = cv2.remap(out, mx, my, cv2.INTER_LANCZOS4, borderMode=mode)
            if k == sub:
                tgt = inp.astype(np.float32)
            else:  # bewegungsinterpolierter Zwischenframe der Quelle
                a = cv2.remap(prev_in, xs + t * bwd[..., 0], ys + t * bwd[..., 1], cv2.INTER_LINEAR, borderMode=mode)
                b = cv2.remap(inp, xs - (1 - t) * bwd[..., 0], ys - (1 - t) * bwd[..., 1], cv2.INTER_LINEAR,
                              borderMode=mode)
                tgt = (1 - t) * a.astype(np.float32) + t * b.astype(np.float32)
            base = trail_sub * adv + (1.0 - trail_sub) * tgt
            if occ is None:
                out = base
            elif occl == "hold":
                out = occ * adv + (1.0 - occ) * base
            else:  # fill
                out = occ * tgt + (1.0 - occ) * base
            out = np.clip(out, 0, 255)
            if hf > 0:  # Schärfe der Quelle zurückholen, Farbe/Form bleiben advektiert
                out = core.hf_inject(out.astype(np.uint8), np.clip(tgt, 0, 255).astype(np.uint8),
                                     hf, hf_sigma).astype(np.float32)
            core.imwrite(fdir / f"{k_out:05d}.png", out.astype(np.uint8))
            k_out += 1
        prev_in = inp
        if i % 10 == 0 or i == n - 1:
            log(f"[C] {i + 1}/{n}  {(time.time() - t0) / (i + 1):.2f} s/Quellframe")

    return core.finish_run(out_dir, fps * sub, {
        "method": "c", "zoom": str(zoom), "sub": sub, "trail": trail, "occl": occl,
        "occl_thresh": occl_thresh, "backend": be.name, "hf": hf, "size": f"{w}x{h}"}, t0)
