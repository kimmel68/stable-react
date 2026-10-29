"""Tiefenvergleich Depth Anything V2 (pro Frame) gegen DVD (Video-Tiefe): Zittern = mittlere Abweichung der Tiefe von Frame
zu Frame, nachdem die vorige Tiefenkarte entlang der Videobewegung verschoben wurde (beide Karten pro Frame normiert).
Niedriger = zeitlich stabiler. Dazu ein Video Quelle | Depth Anything V2 | DVD.

    python -m dmk.tiefe_vergleich <quelle_frames> <da2_ordner> <dvd_ordner> <ausgabe>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import core


def lade(p: Path, size) -> np.ndarray:
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
    d = cv2.resize(d, size, interpolation=cv2.INTER_AREA)
    return (d - d.mean()) / (d.std() + 1e-6)


def main() -> int:
    q, da, dv, out = (Path(x) for x in sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)
    fs = core.list_images(q)
    w, h = 384, 216
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rand = np.zeros((h, w), bool)
    rand[8:-8, 8:-8] = True
    werte = {"da2": [], "dvd": []}
    vdir = out / "_frames"
    vdir.mkdir(exist_ok=True)
    prev = None
    for i, f in enumerate(fs):
        g = cv2.cvtColor(cv2.resize(core.imread(f), (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        karten = {"da2": lade(da / f"{f.stem}_depth.png", (w, h)), "dvd": lade(dv / f"{f.stem}_depth.png", (w, h))}
        if prev is not None:
            flow = dis.calc(g, prev[0], None)
            for k in karten:
                alt = cv2.remap(prev[1][k], xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_LINEAR)
                werte[k].append(float(np.abs(karten[k] - alt)[rand].mean()))
        prev = (g, karten)
        bild = cv2.resize(core.imread(f), (w, h))
        farbe = [cv2.applyColorMap(np.clip((karten[k] * 0.25 + 0.5) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                 for k in ("da2", "dvd")]
        tafel = np.hstack([bild] + farbe)
        for x, t in zip((0, w, 2 * w), ("Quelle", "Depth Anything V2 (pro Frame)", "DVD (Video-Tiefe)")):
            cv2.putText(tafel, t, (x + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        core.imwrite(vdir / f"{i:05d}.png", tafel)
    core.frames_to_mp4(vdir, out / "tiefe_vergleich.mp4", 30)
    erg = {k: {"zittern_mittel": round(float(np.mean(v)), 4), "zittern_p95": round(float(np.percentile(v, 95)), 4)}
           for k, v in werte.items()}
    erg["dvd_besser_um_prozent"] = round(100 * (1 - erg["dvd"]["zittern_mittel"] / erg["da2"]["zittern_mittel"]), 1)
    (out / "tiefe_vergleich.json").write_text(json.dumps(erg, indent=1), encoding="utf-8")
    print(json.dumps(erg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
