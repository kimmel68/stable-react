"""Raumtreue der Flux-Keyframes (Nutzer 27.10.: "die räumliche Vorlage soll immer dieselbe bleiben, es sieht aus wie
Szenenwechsel ohne Zusammenhang"). Kontext mit Denoise < 1 startet vom Latent des Originalframes statt aus reinem
Rauschen; gemessen wird die Kantenkorrelation zum Originalframe (h1-Küste: tropisch 0,52, Arktis 0,03–0,28).

    python -m dmk.flux_raumtest <job> <frame,frame,...> <welt_index,...> [--denoise 0.9,0.8,0.7]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import core, method_a
from .flux_bonus import ANWEISUNG, GUIDANCE, workflow


def kanten(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(img, (384, 216)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 2)
    e = cv2.GaussianBlur(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 3)
    return (e - e.mean()) / (e.std() + 1e-6)


def treue(a: np.ndarray, b: np.ndarray) -> float:
    return round(float((kanten(a) * kanten(b)).mean()), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("frames")
    ap.add_argument("welten")
    ap.add_argument("--denoise", default="0.9,0.8,0.7")
    ap.add_argument("--anweisung", default="", help="eigene Anweisung mit {welt}; leer = Standard")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    job = Path(a.job)
    welten = json.loads((job / "flux" / "flux_stufe.json").read_text(encoding="utf-8"))["welten"]
    quellen = core.list_images(job / "quelle_video")
    out = job / "raumtest"
    out.mkdir(exist_ok=True)
    comfy = method_a.make_client("127.0.0.1:8188", "sdk", "http://127.0.0.1:8189")
    erg, zeilen = [], []
    for fr, wi in zip((int(x) for x in a.frames.split(",")), (int(x) for x in a.welten.split(","))):
        src = cv2.resize(core.imread(quellen[fr]), (768, 432), interpolation=cv2.INTER_AREA)
        reihe = [src]
        alt = job / "flux" / "keys" / f"{fr:05d}_w{wi}.png"
        if alt.exists():
            reihe.append(core.imread(alt))
            erg.append({"frame": fr, "welt": wi, "denoise": 1.0, "treue": treue(reihe[-1], src)})
        for d in (float(x) for x in a.denoise.split(",")):
            name = comfy.upload(src, "dmk_flux_in.png")       # das SDK verbraucht den Upload pro Auftrag
            wf = workflow(name, (a.anweisung or ANWEISUNG).format(welt=welten[wi]), 1000 + wi, 20, GUIDANCE)
            wf["10"]["inputs"]["denoise"] = d
            img = comfy.image(comfy.wait(comfy.queue(wf), timeout=1800))
            img = cv2.resize(img, (768, 432), interpolation=cv2.INTER_LANCZOS4)
            core.imwrite(out / f"{fr:05d}_w{wi}_d{d:.2f}{a.tag}.png", img)
            reihe.append(img)
            erg.append({"frame": fr, "welt": wi, "denoise": d, "tag": a.tag, "treue": treue(img, src),
                        "aenderung": round(float(np.abs(img.astype(np.float32) - src).mean()), 1)})
            print(erg[-1], flush=True)
        zeilen.append(np.hstack([cv2.resize(x, (384, 216)) for x in reihe]))
    comfy.uploads_aufraeumen()
    core.imwrite(out / f"vergleich{a.tag}.jpg", np.vstack(zeilen))
    (out / f"raumtest{a.tag}.json").write_text(json.dumps(erg, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
