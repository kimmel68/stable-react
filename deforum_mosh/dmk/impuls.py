"""dmk.impuls: Wann setzt die generative Verwandlung nach dem Beat ein?

Pro Frame wird die Änderung gemessen, die NICHT durch Bewegung erklärt ist (Bild i gegen Bild i-1 nach DIS-Fluss-Ausgleich,
das "Brodeln" pro Frame) – das ist die Inhaltsverwandlung, unabhängig von Zoom und Kamerafahrt. Dazu getrennt die globale
Bewegung (Zoom-Puls). Beide werden um jeden Beat herum gemittelt (Fenster −4..+10 Frames); ausgegeben werden das Profil
und die Lage des Maximums relativ zum Beat (0 = im Beat-Frame, positiv = später).

    python -m dmk.impuls <lauf> [<lauf> …]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from . import core

VOR, NACH = 4, 10


def reihen(run: Path) -> tuple[np.ndarray, np.ndarray]:
    fs = core.list_images(run / "frames")
    g = [cv2.resize(core.imread(p, gray=True), (320, 180), interpolation=cv2.INTER_AREA) for p in fs]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    ys, xs = np.mgrid[0:180, 0:320].astype(np.float32)
    inhalt, bewegung = [0.0], [0.0]
    for a, b in zip(g[:-1], g[1:]):
        f = dis.calc(a, b, None)
        w = cv2.remap(b, xs + f[..., 0], ys + f[..., 1], cv2.INTER_LINEAR)
        inhalt.append(float(np.abs(w.astype(np.float32) - a).mean()))
        bewegung.append(float(np.linalg.norm(f, axis=2).mean()))
    return np.array(inhalt), np.array(bewegung)


def profil(sig: np.ndarray, beats: np.ndarray) -> np.ndarray:
    st = []
    for b in beats:
        if b - VOR >= 1 and b + NACH < len(sig):
            st.append(sig[b - VOR:b + NACH + 1])
    if not st:
        return np.zeros(VOR + NACH + 1)
    p = np.mean(st, 0)
    return p - np.median(sig)          # über dem Grundrauschen


def analyse(run: Path, quelle: str = "beat") -> dict:
    run = Path(run)
    beats = np.flatnonzero(np.load(run / "audio_kurven.npz")[f"m_{quelle}"] > 0.5)
    inh, bew = reihen(run)
    pi, pb = profil(inh, beats), profil(bew, beats)
    return {"lauf": run.name, "beats": int(len(beats)),
            "inhalt_spitze": int(np.argmax(pi)) - VOR, "inhalt_profil": np.round(pi, 2).tolist(),
            "bewegung_spitze": int(np.argmax(pb)) - VOR, "bewegung_profil": np.round(pb, 2).tolist(),
            # Schwerpunkt der Verwandlung nach dem Beat (0..NACH): wie lang zieht sie sich?
            "inhalt_schwerpunkt": round(float((np.clip(pi[VOR:], 0, None) * np.arange(NACH + 1)).sum() /
                                              max(1e-6, np.clip(pi[VOR:], 0, None).sum())), 2)}


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    print(f"Frames relativ zum Beat: {list(range(-VOR, NACH + 1))}")
    for a in (argv if argv is not None else sys.argv[1:]):
        r = analyse(Path(a))
        print(f"\n{r['lauf']} ({r['beats']} Beats): Verwandlung max bei {r['inhalt_spitze']:+d} Frames, Schwerpunkt "
              f"{r['inhalt_schwerpunkt']} · Zoom-Bewegung max bei {r['bewegung_spitze']:+d}")
        print("  Verwandlung:", r["inhalt_profil"])
        print("  Bewegung:   ", r["bewegung_profil"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
