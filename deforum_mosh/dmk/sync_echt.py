"""dmk.sync_echt: Ende-zu-Ende-Synchronprüfung am fertigen MP4, ohne das Beat-Signal des Laufs.

dmk.sync misst die Bildänderung gegen das Beat-Signal, das der Lauf selbst erkannt hat. Liegt dieses Signal falsch (z. B.
der Beat-Tracker setzt Beats systematisch zu früh), sieht der Test das nicht. Hier wird beides direkt aus dem MP4 gelesen:
  Ton:  Anschlag jedes Beats fein aufgelöst (spektraler Fluss, 3-ms-Raster) aus der Tonspur des MP4
  Bild: Frame mit der größten Bildänderung in ±150 ms um den Anschlag; er ist ab k/fps zu sehen
Versatz = Bildzeit − Anschlag; negativ = das Bild kommt VOR dem Ton.

    python -m dmk.sync_echt <lauf_ordner|video.mp4> [...]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from . import audio, proc

SR = 22050


def anschlaege(y: np.ndarray, grob_s: np.ndarray, fenster: float = 0.12) -> np.ndarray:
    """Feiner Anschlagszeitpunkt je Grob-Beat: steilster Anstieg der log-Energie (Fenster 256, Hop 32)."""
    hop, win = 32, 256
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    e = np.log((y[idx] ** 2).sum(1) + 1e-9)
    e = cv2.GaussianBlur(e.reshape(-1, 1), (0, 0), 1.5).ravel()
    d = np.diff(e, prepend=e[0])
    t = (np.arange(n) * hop + win / 2) / SR
    out = []
    for g in grob_s:
        sel = (t >= g - fenster) & (t <= g + fenster)
        out.append(float(t[sel][np.argmax(d[sel])]) if sel.any() else np.nan)
    return np.array(out)


def bildaenderung(mp4) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev, ch = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (256, 144), interpolation=cv2.INTER_AREA).astype(np.float32)
        ch.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    return np.array(ch), float(fps)


def pruefe(pfad) -> dict:
    pfad = Path(pfad)
    mp4 = pfad / "video_ton.mp4" if pfad.is_dir() else pfad
    y = audio.dekodieren(mp4, SR)
    # Beats aus dem GANZEN Original-Track (auf einem 8-s-Ausschnitt rastet der Tracker auf ein falsches Tempo ein),
    # fein auf den Anschlag; dann um den gemessenen Versatz der Tonspur im MP4 korrigiert. Ohne audio.json: aus dem MP4.
    aj = pfad / "audio.json" if pfad.is_dir() else None
    spec = json.loads(aj.read_text(encoding="utf-8")) if aj and aj.exists() else {}
    track = spec.get("spec", {}).get("datei")
    if track and not Path(track).exists():
        track = str(Path(__file__).resolve().parents[2] / track)
    if track and Path(track).exists():
        # audio.json "offset" ist der Analyse-Offset (Tonausschnitt minus Bildversatz): Versatz wieder addieren
        start = float(spec.get("offset", 0.0)) + float(spec.get("spec", {}).get("bild_versatz_ms", 0.0)) / 1000
        yt = audio.dekodieren(track, SR)
        _, info = audio.merkmale(yt, SR, int(len(yt) / SR * 30), 1 / 30, 0.0, log=lambda s: None)
        from . import sync
        ms, _ = sync.tonversatz(mp4, track, start)
        fein = np.array(info["beat_zeiten"]) - start + ms / 1000.0
        fein = fein[(fein > 0.1) & (fein < len(y) / SR - 0.1)]
    else:
        _, info = audio.merkmale(y, SR, int(len(y) / SR * 30), 1 / 30, 0.0, log=lambda s: None)
        fein = np.array(info["beat_zeiten"])
    grob = fein
    ch, fps = bildaenderung(mp4)
    vers, tracker = [], []
    for a, g in zip(fein, grob):
        if np.isnan(a):
            continue
        k0, k1 = int(np.floor((a - 0.15) * fps)), int(np.ceil((a + 0.15) * fps))
        k0, k1 = max(1, k0), min(len(ch) - 1, k1)
        if k1 <= k0:
            continue
        k = k0 + int(np.argmax(ch[k0:k1 + 1]))
        vers.append(k / fps - a)
        tracker.append(np.ceil(a * fps - 0.25) / fps - a)   # wo der Schnitt laut Zuordnung liegen sollte
    v = np.array(vers) * 1000
    return {"lauf": pfad.name if pfad.is_dir() else pfad.stem, "beats": len(v),
            "versatz_median_ms": round(float(np.median(v)), 1) if len(v) else None,
            "versatz_mittel_ms": round(float(np.mean(v)), 1) if len(v) else None,
            "anteil_bild_zu_frueh": round(float((v < -10).mean()), 2) if len(v) else None,
            "soll_ms": round(float(np.median(tracker) * 1000), 1) if tracker else None,
            "frame_ms": round(1000 / fps, 1)}


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    keys = ["lauf", "beats", "versatz_median_ms", "versatz_mittel_ms", "anteil_bild_zu_frueh", "soll_ms", "frame_ms"]
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for p in (argv if argv is not None else sys.argv[1:]):
        r = pruefe(Path(p))
        print("| " + " | ".join(str(r.get(k, "–")) for k in keys) + " |", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
