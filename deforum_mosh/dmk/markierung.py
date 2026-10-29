"""dmk.markierung (internes Entwicklerwerkzeug, nicht Teil der App): Prüffassung eines Laufs mit farbig markierten Raster-Frames (Nutzer 21.10.: "färbe die betroffenen
Raster-Frames farbig ein zur Erkennung").

Auf jedem Beat-Frame liegt ein farbiger Rahmen mit der Zählzeit (1 rot, 2 gelb, 3 grün, 4 blau), ein Weltwechsel
färbt zusätzlich das ganze Bild magenta ein. Unten läuft ein Balken über den Takt (Taktphase): reißt er nicht und
springt er genau mit dem Schlag zurück, sitzt das Raster. Ausgabe: video_raster.mp4 mit Ton, neben video_ton.mp4.

    python -m dmk.markierung <lauf_ordner> [...]
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import audio, core

FARBEN = {1: (40, 40, 255), 2: (0, 220, 255), 3: (60, 220, 60), 4: (255, 140, 30)}   # BGR
MAGENTA = (255, 0, 255)


def markieren(run, rand: int = 10) -> Path | None:
    run = Path(run)
    npz = run / "audio_kurven.npz"
    if not npz.exists():
        return None
    k = np.load(npz)
    fs = core.list_images(run / "frames")
    n = len(fs)

    def reihe(name):
        return k[name][:n] if name in k.files else np.zeros(n)
    beat, wechsel, phase = reihe("m_beat") > 0.5, reihe("m_wechsel") > 0.5, reihe("m_takt_phase")
    zaehl = np.zeros(n, int)
    for z in range(1, 5):
        zaehl[reihe(f"m_takt{z}") > 0.5] = z
    tmp = Path(tempfile.mkdtemp(prefix="dmk_raster_"))
    try:
        for i, f in enumerate(fs):
            img = core.imread(f)
            h, w = img.shape[:2]
            if wechsel[i]:
                img = cv2.addWeighted(img, 0.65, np.full_like(img, MAGENTA), 0.35, 0)
            if beat[i]:
                c = FARBEN.get(int(zaehl[i]), (255, 255, 255))
                cv2.rectangle(img, (0, 0), (w - 1, h - 1), c, rand)
                cv2.putText(img, str(int(zaehl[i]) or "B"), (rand + 6, rand + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3)
            x = int(round(float(phase[i]) * (w - 1)))
            cv2.rectangle(img, (0, h - 5), (x, h - 1), (255, 255, 255), -1)
            core.imwrite(tmp / f"{i:05d}.png", img)
        core.frames_to_mp4(tmp, tmp / "v.mp4", 30)
        info = json.loads((run / "audio.json").read_text(encoding="utf-8"))
        spec = info.get("spec", {})
        track = spec.get("datei")
        if track and not Path(track).exists():
            track = str(Path(__file__).resolve().parents[2] / track)
        ziel = run / "video_raster.mp4"
        if track and Path(track).exists():
            start = float(info.get("offset", 0.0)) + float(spec.get("bild_versatz_ms", 0.0)) / 1000
            basis = float(spec.get("basis_fps", 30))
            audio.ton_anlegen(tmp / "v.mp4", track, ziel, start, n / basis, n / 30.0)
        else:
            shutil.copy(tmp / "v.mp4", ziel)
        return ziel
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(markieren(p))
