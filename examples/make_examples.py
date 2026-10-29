"""Neutrale Beispielbilder ohne fremde Rechte: ein prozedural gezeichneter Flur in Zentralperspektive mit exakter
Tiefenkarte, daraus erzeugt SD 1.5 (DreamShaper 8, offene Gewichte) mit Depth-ControlNet drei Architekturbilder.

    python examples/make_examples.py          (ComfyUI muss laufen, Comfy SDK über comfy-api-proxy oder HTTP)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deforum_mosh"))
from dmk import core, method_a  # noqa: E402

W, H = 768, 432
OUT = Path(__file__).resolve().parent

PROMPTS = {
    "hall": "vast empty brutalist concrete hall, rows of massive pillars, coffered ceiling, raking light, "
            "strong one-point perspective, symmetrical, crisp architectural detail, black and white photograph",
    "cloister": "long gothic cloister corridor, repeating pointed arches, ribbed vaults, stone floor, soft daylight, "
                "strong one-point perspective, symmetrical, crisp architectural detail, black and white photograph",
    "hangar": "huge industrial steel hangar interior, lattice trusses, skylights, polished floor, "
              "strong one-point perspective, symmetrical, crisp architectural detail, black and white photograph",
}
NEGATIV = "people, person, text, watermark, logo, color, blurry, lowres, distorted, tilted"


def flur(w=W, h=H, breite=1.0, hoehe=0.62, tiefe=9.0, f=0.9):
    """Kastenflur: Strahl durch Pixel trifft Wand (|x| = breite), Boden/Decke (|y| = hoehe) oder Rückwand (z = tiefe).
    Gibt Grauwertbild (Schattierung + Fugen) und Disparität (1/z, nah = hell) zurück."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = (xs - w / 2) / (f * w / 2), (ys - h / 2) / (f * w / 2)
    tx = np.where(np.abs(dx) > 1e-6, breite / np.maximum(np.abs(dx), 1e-6), np.inf)
    ty = np.where(np.abs(dy) > 1e-6, hoehe / np.maximum(np.abs(dy), 1e-6), np.inf)
    z = np.minimum(np.minimum(tx, ty), tiefe)
    wand = (tx <= ty) & (z < tiefe)
    boden = (ty < tx) & (dy > 0) & (z < tiefe)
    decke = (ty < tx) & (dy < 0) & (z < tiefe)
    img = np.full((h, w), 120, np.float32)
    img[wand] = 150 - 6 * z[wand]
    img[boden] = 95 - 4 * z[boden]
    img[decke] = 175 - 7 * z[decke]
    img[z >= tiefe] = 60
    fugen = (np.abs(np.sin(z * np.pi * 1.2)) < 0.06) & (z < tiefe)   # Querfugen alle ~0,8 Einheiten
    img[fugen] *= 0.6
    disp = 1.0 / z
    disp = (disp - disp.min()) / (disp.max() - disp.min())
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), disp


def main():
    img, disp = flur()
    core.imwrite(OUT / "flur_init.png", img)
    core.imwrite(OUT / "flur_depth.png", (disp * 65535).astype(np.uint16))
    c = method_a.make_client(backend="sdk") if "--http" not in sys.argv else method_a.Comfy()
    wf = json.loads(Path(method_a.DEPTH_WORKFLOW).read_text(encoding="utf-8"))
    ckpt = next(x for x in c.checkpoints() if "dreamshaper_8" in x.lower())
    cn = next(x for x in c.controlnets() if "depth" in x.lower() and "sd 1.5" in x.lower())
    ctrl = cv2.cvtColor((disp * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for k, (name, prompt) in enumerate(PROMPTS.items()):
        a, b = c.upload(img), c.upload(ctrl, "dmk_control.png")
        w = method_a.patch_workflow(wf, a, prompt, NEGATIV, ckpt, 4242 + k, 30, 7.0, 0.92, None, None, b, cn, 0.9, 1.0)
        out = c.image(c.wait(c.queue(w)))
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        core.imwrite(OUT / f"{name}.png", out)
        print(name, out.shape)


if __name__ == "__main__":
    main()
