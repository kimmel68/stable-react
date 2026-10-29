"""Flux-Kontext-Prompttest: welche Anweisung hält die Komposition (Draufsicht) am besten? Maß: Korrelation der Kantenbilder
von Quelle und Ergebnis (64×36, weichgezeichnet) – 1 = gleiche Formen an gleicher Stelle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import core, flux_bonus as F, method_a

VARIANTEN = {
    "a_restyle": "Restyle this exact image as an arctic coast: the rocks become ice and snow, the water becomes dark cold "
                 "sea. Do not change the camera, perspective or framing: every shape, outline and position stays exactly "
                 "where it is.",
    "b_change": "Change the rocks to snow-covered ice and the water to dark arctic sea, while keeping the same top-down drone "
                "view, the same framing and the exact position and outline of every rock.",
    "c_kurz": "Cover the rocks with snow and ice. Keep everything else in the image exactly the same.",
}


def kanten(img):
    g = cv2.cvtColor(cv2.resize(img, (256, 144), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 2), cv2.CV_32F))
    e = cv2.resize(cv2.GaussianBlur(e, (0, 0), 3), (64, 36), interpolation=cv2.INTER_AREA).ravel()
    return (e - e.mean()) / (e.std() + 1e-6)


def main() -> int:
    src_p, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    src = core.imread(src_p)
    comfy = method_a.make_client("127.0.0.1:8188", "sdk", "http://127.0.0.1:8189")
    ks = kanten(src)
    erg = {}
    for name, prompt in VARIANTEN.items():
        for g in (2.5, 1.5):
            ziel = out / f"{name}_g{g}.png"
            if not ziel.exists():
                nm = comfy.upload(src, "dmk_flux_in.png")
                img = comfy.image(comfy.wait(comfy.queue(F.workflow(nm, prompt, 1000, 20, g)), timeout=1800))
                core.imwrite(ziel, img)
            img = core.imread(ziel)
            erg[f"{name}_g{g}"] = round(float(np.mean(ks * kanten(cv2.resize(img, (src.shape[1], src.shape[0]))))), 3)
            print(name, g, erg[f"{name}_g{g}"], flush=True)
    (out / "ergebnis.json").write_text(json.dumps(erg, indent=1), encoding="utf-8")
    comfy.uploads_aufraeumen()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
