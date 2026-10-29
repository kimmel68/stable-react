"""dmk.sync: prüft, ob Bild und Ton eines fertigen Videos synchron sind.

Drei Ebenen:
  1. Container: Startzeit und Länge von Video- und Tonspur (ffprobe).
  2. Tonversatz: Kreuzkorrelation der Tonspur im MP4 mit dem Original-Track am erwarteten Ausschnitt → Versatz in ms
     (AAC-Encoder und atempo können verschieben; < 1 Frame = synchron).
  3. Bild ↔ Takt: Bildänderung pro Frame gegen das Beat-Signal des Laufs, bestes Lag in Frames (0 = Ereignis
     im selben Frame wie der Beat).

    python -m dmk.sync <lauf_ordner> [<lauf_ordner> …]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import audio, proc

SR = 8000


def _probe(path) -> dict:
    r = subprocess.run([proc.tool("ffprobe"), "-v", "error", "-show_entries",
                        "stream=codec_type,start_time,duration,r_frame_rate,nb_frames:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True)
    return json.loads(r.stdout)


def tonversatz(mp4, track, start_sek: float, tempo: float = 1.0, max_ms: float = 300) -> tuple[float, float]:
    """Versatz (ms) der Tonspur im MP4 gegenüber dem Track; positiv = Ton im Video kommt später. Dazu die
    Korrelationsgüte (0..1). tempo ≠ 1: der Track wird vorher genauso per atempo gestreckt wie beim Anlegen."""
    a = audio.dekodieren(mp4, SR)
    dauer = len(a) / SR
    if abs(tempo - 1.0) < 1e-6:
        b = audio.dekodieren(track, SR, start=start_sek, dauer=dauer + 1)
    else:
        r = proc.run([proc.tool("ffmpeg"), "-v", "error", "-ss", f"{start_sek:.3f}", "-t", f"{dauer * tempo + 1:.3f}",
                      "-i", str(track), "-filter:a", f"atempo={tempo:.6f}", "-ac", "1", "-ar", SR, "-f", "f32le", "-"],
                     "Track strecken")
        b = np.frombuffer(r.stdout, np.float32).copy()
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    m = int(max_ms / 1000 * SR)
    fa, fb = np.fft.rfft(a, 2 * n), np.fft.rfft(b, 2 * n)
    xc = np.fft.irfft(fa * np.conj(fb))
    xc = np.concatenate([xc[-m:], xc[:m + 1]])            # Lags -m..m
    k = int(np.argmax(xc))
    lag = k - m
    guete = float(xc[k] / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-12))
    return lag / SR * 1000.0, guete


def bild_lag(run: Path, max_lag: int = 4) -> dict:
    """Bildänderung (1 − SSIM zum Vorframe) gegen das Beat-Signal, Lag −max..+max Frames.
    Positiv = Bild reagiert nach dem Ton, negativ = Bild vor dem Ton. Dazu: Bildänderung genau auf den Beat-Frames
    im Verhältnis zur Ruhe (wie stark der Beat sichtbar ist)."""
    import cv2
    from . import core
    fs = core.list_images(run / "frames")
    beat = np.load(run / "audio_kurven.npz")["m_beat"]
    prev, ch = None, []
    for p in fs:
        g = cv2.resize(core.imread(p, gray=True), (320, 180), interpolation=cv2.INTER_AREA)
        ch.append(0.0 if prev is None else 1.0 - core.ssim(prev, g))
        prev = g
    ch = np.array(ch)
    m = min(len(ch), len(beat))
    ch, beat = ch[:m], beat[:m]
    best = (-9.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        a = ch[max(0, lag):m + min(0, lag)]
        b = beat[max(0, -lag):m - max(0, lag)]
        if a.std() > 0 and b.std() > 0:
            r = float(np.corrcoef(a, b)[0, 1])
            if r > best[0]:
                best = (r, lag)
    bf = np.flatnonzero(beat > 0.5)
    ruhe = ch[beat < 0.2].mean() if (beat < 0.2).any() else np.nan
    return {"bild_lag_frames": best[1], "bild_r": round(best[0], 3), "beats": int(len(bf)),
            "beat_zu_ruhe": round(float(ch[bf].mean() / ruhe), 2) if len(bf) and ruhe > 0 else None}


def pruefe(run: Path) -> dict:
    run = Path(run)
    vid = run / "video_ton.mp4"
    out = {"lauf": run.name}
    if not vid.exists():
        out["fehler"] = "kein video_ton.mp4"
        return out
    info = json.loads((run / "audio.json").read_text(encoding="utf-8")) if (run / "audio.json").exists() else {}
    meta = json.loads((run / "run.json").read_text(encoding="utf-8")) if (run / "run.json").exists() else {}
    pr = _probe(vid)
    v = next(s for s in pr["streams"] if s["codec_type"] == "video")
    a = next(s for s in pr["streams"] if s["codec_type"] == "audio")
    fps = eval(v["r_frame_rate"])  # noqa: S307  "30/1"
    out.update(fps=fps, video_s=round(float(v["duration"]), 3), ton_s=round(float(a["duration"]), 3),
               video_start_ms=round(float(v.get("start_time", 0)) * 1000, 1),
               ton_start_ms=round(float(a.get("start_time", 0)) * 1000, 1))
    spec = info.get("spec", {})
    track = spec.get("datei")
    if track and Path(track).exists():
        frames = int(meta.get("frames", 0)) or int(v.get("nb_frames", 0))
        basis = float(spec.get("basis_fps", 30))
        start = float(info.get("offset", 0.0))
        quell_s = frames * int(meta.get("stride", 1) or 1) / basis
        tempo = quell_s / (frames / fps) if frames else 1.0
        ms, g = tonversatz(vid, track, start, tempo)
        out.update(ton_versatz_ms=round(ms, 1), korrelation=round(g, 3), tempo=round(tempo, 4))
    if (run / "audio_kurven.npz").exists():
        out.update(bild_lag(run))
    out["frame_ms"] = round(1000 / fps, 1)
    return out


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    rows = [pruefe(Path(p)) for p in (argv if argv is not None else sys.argv[1:])]
    keys = ["lauf", "fps", "video_s", "ton_s", "video_start_ms", "ton_start_ms", "tempo", "ton_versatz_ms",
            "korrelation", "beats", "bild_lag_frames", "bild_r", "beat_zu_ruhe", "frame_ms"]
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(str(r.get(k, "–")) for k in keys) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
