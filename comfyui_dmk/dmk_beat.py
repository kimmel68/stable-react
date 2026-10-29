"""dmk beat grammar for ComfyUI: audio → beats on a 4/4 bar → per-frame parameter curves and bar-synced prompt schedules.

Self-contained (numpy only, no ffmpeg/cv2/librosa), so the nodes work in any ComfyUI install. The analysis is the same as
in the dmk app (deforum_mosh/dmk/audio.py):
  * spectral flux onset envelope, tempo from autocorrelation, beats by dynamic programming (Ellis 2007)
  * every beat refined to the actual transient (short-window flux, ~3 ms) – the coarse tracker grid put the picture
    20–40 ms ahead of the sound; an event belongs to the first frame that starts no more than 1/4 frame before it
  * optional fixed BPM grid fitted to the beats (for tightly produced music: no per-beat detection error)
  * count 1 = the beat phase with the most bass; counts 2–4, offbeats, bar/beat phase ramps
Grammar = JSON {parameter: binding | [binding, ...]}, binding = {"quelle": feature, "min", "max", "attack", "release",
"schwelle", "kurve", "invert", "glatt", "lokal"}; several bindings on one parameter add up (the first sets the rest value).
"""
from __future__ import annotations

import json

import numpy as np

SR = 22050
N_FFT = 2048
HOP = 512
BANDS = {"bass": (20, 150), "lowmid": (150, 400), "mid": (400, 2000), "high": (2000, 8000)}
# feature names: English for the nodes; the German keys of dmk recipes are accepted as aliases
ALIAS = {"huelle": "envelope", "takt": "count1", "takt1": "count1", "takt2": "count2", "takt3": "count3", "takt4": "count4",
         "takt_phase": "bar_phase", "beat_phase": "beat_phase", "hoehen": "high", "mitten": "mid", "tief": "lowmid",
         "hell": "brightness", "wechsel": "world_change"}
COUNT_COLORS = {1: (255, 60, 40), 2: (255, 220, 0), 3: (60, 220, 60), 4: (30, 140, 255)}   # RGB, same as dmk check videos


# ------------------------------------------------------------------ signal helpers

def _gauss1d(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


def _norm(x: np.ndarray, lo_p: float = 2, hi_p: float = 98) -> np.ndarray:
    lo, hi = np.percentile(x, lo_p), np.percentile(x, hi_p)
    return np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(x)


def _stft_power(y: np.ndarray, nfft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    yp = np.pad(y, (nfft // 2, nfft // 2), mode="reflect")
    n = 1 + (len(yp) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(yp[idx] * np.hanning(nfft)[None, :], axis=1)) ** 2


def _beats(onset: np.ndarray, fps_env: float, bpm_min: float, bpm_max: float, tight: float = 400.0):
    o = onset - onset.mean()
    ac = np.correlate(o, o, "full")[len(o) - 1:]
    lags = np.arange(int(fps_env * 60 / bpm_max), min(int(fps_env * 60 / bpm_min), len(ac) - 1) + 1)
    bpm = 60.0 * fps_env / lags
    period = int(lags[np.argmax(ac[lags] * np.exp(-0.5 * (np.log2(bpm / 120.0)) ** 2))])
    n = len(onset)
    score, back = onset.copy(), np.full(n, -1, dtype=int)
    cands = np.arange(int(round(period * 0.5)), int(round(period * 2.0)) + 1)
    pen = -tight * (np.log(cands / period)) ** 2
    for i in range(cands[0], n):
        js = i - cands
        ok = js >= 0
        vals = score[js[ok]] + pen[ok]
        k = int(np.argmax(vals))
        if vals[k] > 0:
            score[i], back[i] = onset[i] + vals[k], js[ok][k]
    tail = max(1, n - period)
    i = int(tail + np.argmax(score[tail:]))
    out = [i]
    while back[i] >= 0:
        i = back[i]
        out.append(i)
    return np.array(out[::-1], dtype=int), 60.0 * fps_env / period


def _refine(y: np.ndarray, t: np.ndarray, window: float = 0.04) -> np.ndarray:
    P = _stft_power(y, 512, 64)
    L = np.log1p(P * 1e3)
    f = np.concatenate([[0.0], np.maximum(L[1:] - L[:-1], 0).sum(1)])
    tt = np.arange(len(f)) * 64 / SR
    out = np.asarray(t, float).copy()
    for k, g in enumerate(out):
        s = (tt >= g - window) & (tt <= g + window)
        if s.any():
            out[k] = tt[s][int(np.argmax(f[s]))]
    return out


def grid_fit(t: np.ndarray) -> dict:
    t = np.sort(np.asarray(t, float))
    if len(t) < 4:
        return {}
    T, t0, ok = float(np.median(np.diff(t))), float(t[0]), np.ones(len(t), bool)
    for _ in range(4):
        k = np.round((t - t0) / T)
        (t0, T), *_ = np.linalg.lstsq(np.stack([np.ones(ok.sum()), k[ok]], 1), t[ok], rcond=None)
        res = t - (t0 + k * T)
        ok = np.abs(res) < max(0.04, 3 * np.std(res[ok]))
    return {"bpm": float(60.0 / T), "period": float(T), "t0": float(t0), "scatter_ms": float(np.std(res[ok]) * 1000)}


def _resample(w: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return w.astype(np.float64)
    n = int(len(w) * SR / sr)
    return np.interp(np.arange(n) * sr / SR, np.arange(len(w)), w).astype(np.float64)


# ------------------------------------------------------------------ analysis

def analyse(audio: dict, fps: float, frames: int, offset: float, beat_grid: bool, bpm_min: float, bpm_max: float) -> dict:
    wf = audio["waveform"]
    w = wf.detach().cpu().float().numpy() if hasattr(wf, "detach") else np.asarray(wf, np.float32)
    y = _resample(w[0].mean(0) if w.ndim == 3 else w.mean(0), int(audio["sample_rate"]))
    dur = len(y) / SR
    spf = 1.0 / fps
    n = int(frames) if frames > 0 else max(1, int((dur - offset) * fps))
    P = _stft_power(y)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    fps_env = SR / HOP
    te = np.arange(P.shape[0]) * HOP / SR
    env = {"rms": np.sqrt(P.mean(1))}
    for name, (f0, f1) in BANDS.items():
        env[name] = np.log1p(P[:, (freqs >= f0) & (freqs < f1)].sum(1) * 1e3)
    env["brightness"] = (P * freqs[None, :]).sum(1) / (P.sum(1) + 1e-10)
    L = np.log1p(P * 1e3)
    on = np.concatenate([[0.0], np.maximum(L[1:] - L[:-1], 0).sum(1)])
    w2 = int(fps_env * 0.5)
    on = np.maximum(on - np.convolve(on, np.ones(w2) / w2, "same") * 0.5, 0)
    env["onset"] = on
    beats_e, bpm = _beats(on / (on.max() + 1e-10), fps_env, bpm_min, bpm_max)
    tb = _refine(y, te[beats_e])
    grid = None
    if beat_grid:
        win = (tb >= offset - 8) & (tb <= offset + n * spf + 8)
        grid = grid_fit(tb[win] if win.sum() >= 8 else tb)
        if grid:
            k0 = int(np.ceil((max(0.0, tb[0] - 4 * grid["period"]) - grid["t0"]) / grid["period"]))
            k1 = int(np.floor((te[-1] - grid["t0"]) / grid["period"]))
            tb = grid["t0"] + np.arange(k0, k1 + 1) * grid["period"]
    be = np.clip(np.round(tb * SR / HOP).astype(int), 0, len(te) - 1)
    phase = int(np.argmax([env["bass"][be[p::4]].mean() if len(be[p::4]) else 0 for p in range(4)]))

    t_frame = offset + np.arange(n + 1) * spf
    idx = np.searchsorted(t_frame, te, side="right") - 1
    inside = (idx >= 0) & (idx < n)
    cnt = np.bincount(idx[inside], minlength=n).astype(np.float64)
    feat: dict[str, np.ndarray] = {}
    for name, e in env.items():
        e = _norm(e, 2, 99.5 if name == "onset" else 98)
        if name == "onset":
            m = np.zeros(n)
            np.maximum.at(m, idx[inside], e[inside])
        else:
            m = np.bincount(idx[inside], weights=e[inside], minlength=n) / np.maximum(cnt, 1)
        feat[name] = np.clip(m, 0, 1)
    feat["envelope"] = np.clip(_gauss1d(feat["rms"], max(1.0, round(0.5 / spf)) / 2.0), 0, 1)

    def impulses(times):
        s = np.zeros(n)
        f = np.ceil((np.asarray(times) - offset) / spf - 0.25).astype(int)
        f = np.unique(f[(f >= 0) & (f < n)])
        s[f] = 1.0
        return s, f
    feat["beat"], beat_frames = impulses(tb)
    feat["count1"], bar_frames = impulses(tb[phase::4])
    for c in range(2, 5):
        feat[f"count{c}"], _ = impulses(tb[(phase + c - 1) % 4::4])
    feat["offbeat"], _ = impulses((tb[:-1] + tb[1:]) / 2 if len(tb) > 1 else [])
    t_mid = offset + (np.arange(n) + 0.5) * spf

    def ramp(tm):
        if len(tm) < 2:
            return np.zeros(n)
        j = np.clip(np.searchsorted(tm, t_mid, side="right") - 1, 0, len(tm) - 2)
        return np.clip((t_mid - tm[j]) / (tm[j + 1] - tm[j]), 0, 1)
    feat["bar_phase"], feat["beat_phase"] = ramp(tb[phase::4]), ramp(tb)
    return {"features": feat, "frames": n, "fps": fps, "offset": offset, "bpm": float(bpm), "grid": grid,
            "beat_frames": beat_frames.tolist(), "bar_frames": bar_frames.tolist()}


# ------------------------------------------------------------------ grammar

def _envelope(x: np.ndarray, attack: float, release: float) -> np.ndarray:
    if attack <= 0 and release <= 0:
        return x
    ka = 1.0 - np.exp(-1.0 / attack) if attack > 0 else 1.0
    kr = 1.0 - np.exp(-1.0 / release) if release > 0 else 1.0
    out, v = np.empty_like(x), x[0]
    for i, s in enumerate(x):
        v += (s - v) * (ka if s > v else kr)
        out[i] = v
    return out


def binding(sig, min=0.0, max=1.0, kurve=1.0, schwelle=0.0, invert=False, attack=0.0, release=0.0, glatt=0.0,
            lokal=False, **_) -> np.ndarray:
    s = np.asarray(sig, dtype=np.float64)
    if lokal:
        s = _norm(s)
    if schwelle > 0:
        s = np.clip((s - schwelle) / (1.0 - schwelle), 0, 1)
    s = _envelope(s, attack, release)
    if glatt > 0:
        s = _gauss1d(s, glatt)
    s = np.clip(s, 0, 1) ** kurve
    if invert:
        s = 1.0 - s
    return min + (max - min) * s


def grammar(beats: dict, spec) -> dict:
    spec = json.loads(spec) if isinstance(spec, str) else spec
    feat = beats["features"]

    def one(b):
        q = ALIAS.get(b.get("quelle", b.get("source", "rms")), b.get("quelle", b.get("source", "rms")))
        if q not in feat:
            raise ValueError(f"unknown feature {q!r}; available: {', '.join(sorted(feat))}")
        return binding(feat[q], **{k: v for k, v in b.items() if k not in ("quelle", "source")})
    curves = {}
    for p, b in spec.items():
        if isinstance(b, list):
            ks = [one(x) for x in b]
            rest = float(b[0]["min"])
            lo = min(min(float(x["min"]), float(x["max"])) for x in b)
            hi = max(max(float(x["min"]), float(x["max"])) for x in b)
            curves[p] = np.clip(ks[0] + sum(k - rest for k in ks[1:]), lo, hi)
        else:
            curves[p] = one(b)
    return curves


def bar_prompts(beats: dict, prompts: list[str], every: int) -> tuple[list[int], list[int]]:
    """Frames where the world changes (every `every`-th count 1) and the world index for each frame."""
    bars = np.array(beats["bar_frames"], int)
    jumps = bars[max(1, every) - 1::max(1, every)]
    jumps = jumps[jumps > 0]          # count 1 on frame 0: keep the first world (same rule as dmk.audio)
    world = np.zeros(beats["frames"], int)
    for f in jumps:
        world[f:] += 1
    return jumps.tolist(), (world % max(1, len(prompts))).tolist()


DEFAULT_GRAMMAR = json.dumps({
    "zoom": [{"quelle": "beat", "min": 1.002, "max": 1.008, "attack": 3, "release": 5},
             {"quelle": "count1", "min": 1.002, "max": 1.016, "attack": 3, "release": 5}],
    "denoise": [{"quelle": "onset", "min": 0.40, "max": 0.52, "schwelle": 0.3, "release": 4},
                {"quelle": "count4", "min": 0.40, "max": 0.62, "release": 5}],
    "blur": {"quelle": "count2", "min": 0.0, "max": 3.75, "release": 1},
    "source_pull": {"quelle": "count3", "min": 0.05, "max": 0.6, "release": 3},
    "memory": {"quelle": "bar_phase", "min": 0.375, "max": 0.85, "kurve": 0.7},
    "cfg": {"quelle": "envelope", "min": 8.0, "max": 11.0, "glatt": 3, "lokal": True},
}, indent=1)


# ------------------------------------------------------------------ plot

def plot(beats: dict, curves: dict | None, width: int = 1400, row: int = 70) -> np.ndarray:
    names = list(curves or {}) or ["onset", "bass", "envelope", "bar_phase"]
    src = curves if curves else beats["features"]
    n = beats["frames"]
    H = row * len(names) + 30
    img = np.full((H, width, 3), 22, np.uint8)
    xs = (np.arange(n) / max(1, n - 1) * (width - 1)).astype(int)
    feat = beats["features"]
    for c in range(1, 5):            # beat markers coloured by count (1 red, 2 yellow, 3 green, 4 blue)
        for f in np.flatnonzero(feat[f"count{c}"] > 0.5):
            img[:, max(0, xs[f] - 1):xs[f] + 2] = (np.array(COUNT_COLORS[c]) * (1.0 if c == 1 else 0.7)).astype(np.uint8)
    xcol = np.arange(width) / (width - 1) * max(1, n - 1)
    for r, name in enumerate(names):
        v = np.asarray(src[name], float)
        v = (v - v.min()) / (np.ptp(v) or 1.0)
        top, bot = 15 + r * row + 6, 15 + (r + 1) * row - 6
        ys = (bot - np.interp(xcol, np.arange(n), v) * (bot - top)).astype(int)   # continuous line, one value per column
        for x in range(width):
            y0, y1 = sorted((ys[x], ys[x - 1] if x else ys[x]))
            img[y0:y1 + 2, x] = (235, 235, 235)
        img[bot + 5, :] = 60
    return img
