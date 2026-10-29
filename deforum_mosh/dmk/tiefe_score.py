"""Wie viel Raum liest die Tiefenschätzung aus einem Bild? (Nutzer 28.10.: "Depth liest ja kaum Informationen aus –
finde Beispiele, wo die Depth Map besser funktioniert".) Depth Anything V2 auf jedem Bild, drei Größen:
  spanne    – Abstand nah/fern (P90 − P10 der normierten Tiefe; Draufsicht ≈ flacher Verlauf)
  kanten    – Tiefensprünge (mittlerer Gradient der Tiefe, ×100): Ebenen, Vordergrund vor Hintergrund
  detail    – Feinstruktur der Tiefe (Std der Tiefe minus ihrer Unschärfe, ×100)
score = spanne + kanten/10 + detail/5. Schreibt rangliste.json und ein Raster Bild | Tiefe der besten.

    python -m dmk.tiefe_score <bild|ordner> [...] --out <ordner> [--top 16]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import depth


def werte(img: np.ndarray) -> tuple[dict, np.ndarray]:
    img = cv2.resize(img, (512, int(512 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    d = depth.schaetze(img)
    spanne = float(np.percentile(d, 90) - np.percentile(d, 10))
    g = cv2.GaussianBlur(d, (0, 0), 1.5)
    kanten = float(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean() * 100)
    detail = float((d - cv2.GaussianBlur(d, (0, 0), 8)).std() * 100)
    w = {"spanne": round(spanne, 3), "kanten": round(kanten, 2), "detail": round(detail, 2)}
    w["score"] = round(spanne + kanten / 10 + detail / 5, 3)
    return w, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("quellen", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bilder = []
    for q in map(Path, a.quellen):
        bilder += sorted(p for p in q.iterdir() if p.suffix.lower() in (".jpg", ".png")) if q.is_dir() else [q]
    erg, tafeln = [], {}
    for p in bilder:
        img = cv2.imread(str(p))
        if img is None:
            continue
        w, d = werte(img)
        erg.append(dict(w, bild=p.stem))
        f = cv2.applyColorMap((np.clip(d, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        a_ = cv2.resize(img, (320, 180))
        b_ = cv2.resize(f, (320, 180))
        cv2.putText(a_, f"{p.stem}  {w['score']:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tafeln[p.stem] = np.hstack([a_, b_])
    erg.sort(key=lambda x: -x["score"])
    (out / "rangliste.json").write_text(json.dumps(erg, indent=1), encoding="utf-8")
    beste = [tafeln[e["bild"]] for e in erg[: a.top]]
    while len(beste) % 2:
        beste.append(np.zeros_like(beste[0]))
    cv2.imwrite(str(out / "beste.jpg"), np.vstack([np.hstack(beste[i:i + 2]) for i in range(0, len(beste), 2)]))
    for e in erg:
        print(e["score"], e["spanne"], e["kanten"], e["detail"], e["bild"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
