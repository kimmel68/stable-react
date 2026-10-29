"""Flux in der dmk-Rückkopplungsschleife (Nutzer 26.10.: "das Fließen und die Bewegungen stecken noch nicht drin, der
Aufquell-Look geht verloren, die Motive wechseln einfach auf den Takt").

Statt jedes Keyframe frisch aus dem Video-Frame zu zeichnen, arbeitet Flux wie SD 1.5 in dmk:
  * Ausgangsbild jedes Samples = das vorige GENERIERTE Bild, entlang der Videobewegung verschoben (Rückkopplung)
  * Flux zeichnet davon nur einen Teil neu (Denoise zwischen den Beats ~0,42, auf dem Beat mehr, beim Weltwechsel am meisten)
    → Formen quellen auf und wachsen weiter statt neu erfunden zu werden
  * Kontext-Referenz ist das Welt-Keyframe: beim Weltwechsel zeichnet Flux einmal voll aus dem Video-Frame (hält die
    Komposition), danach wird dieses Keyframe entlang der Videobewegung mitgenommen und trägt Welt UND Komposition.
    (Test 26.10.: mit dem rohen Video-Frame als Referenz zog Flux das Bild in ~50 Frames zur Original-Küste zurück.)
  * Sample jeden 3. Frame und auf jedem Beat; dazwischen werden altes und neues Sample, beide entlang der Bewegung
    mitgenommen, überblendet (auf dem Beat harter Schnitt); bewegungsgeführte Glättung zwischen den Beats

    python -m dmk.flux_fluss <video> <ausgabeordner> --sek 6 --start 87.5 --welten "prompt1|prompt2|..."
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import audio, core, method_a, proc
from .flux_bonus import ANWEISUNG, CLIP_L, GUIDANCE, T5, UNET, VAE

CADENCE = 3
DN_RUHE, DN_BEAT, DN_WECHSEL = 0.42, 0.55, 0.8
GLAETTUNG = 0.3       # Anteil des mitbewegten Vorbilds im neuen Sample zwischen den Beats
QUELLZUG = 0.0        # Anteil des Video-Frames im Ausgangsbild (0.05 liess die Lava-Glut zur Quelle verblassen, Test 26.10.)
SCHRITTE = 20         # für Denoise 1; gerechnet werden SCHRITTE·Denoise


def workflow(init: str, ref: str, prompt: str, seed: int, denoise: float, guidance: float = GUIDANCE) -> dict:
    steps = max(6, round(SCHRITTE * denoise))
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": CLIP_L, "clip_name2": T5, "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "LoadImage", "inputs": {"image": init}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}},
        "13": {"class_type": "LoadImage", "inputs": {"image": ref}},
        "14": {"class_type": "VAEEncode", "inputs": {"pixels": ["13", 0], "vae": ["3", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "7": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["14", 0]}},
        "8": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["7", 0], "guidance": guidance}},
        "9": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "10": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["8", 0], "negative": ["9", 0],
                                                    "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                                                    "sampler_name": "euler", "scheduler": "simple", "denoise": denoise}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "dmk_flux_loop"}},
    }


def render(video: Path, out: Path, sek: float, start: float, welten: list[str], track: Path, breite: int = 768,
           hoehe: int = 432, film: float = 1.0, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    fdir, sdir, qdir = out / "frames", out / "samples", out / "quelle"
    fdir.mkdir(exist_ok=True)
    sdir.mkdir(exist_ok=True)
    if not qdir.exists() or len(core.list_images(qdir)) < int(sek * 30):
        qdir.mkdir(exist_ok=True)
        proc.run([proc.tool("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-t", f"{sek:.3f}", "-vf",
                  f"fps=30,scale={breite}:{hoehe}:force_original_aspect_ratio=increase,crop={breite}:{hoehe}",
                  str(qdir / "%05d.png")], "Video zerlegen")
    quellen = core.list_images(qdir)
    n = min(len(quellen), int(round(sek * 30)))
    y = audio.dekodieren(track, audio.SR)
    _, info = audio.merkmale(y, audio.SR, n, 1 / 30, start, log=lambda s: None)
    beats, takte = set(info["beat_frames"]), set(info["takt_frames"])
    welt, w, letzter = np.zeros(n, int), 0, 0
    for i in range(n):
        if i in takte and i - letzter >= 30 and i > 0:
            w, letzter = w + 1, i
        welt[i] = w % len(welten)
    wechsel = {i for i in range(1, n) if welt[i] != welt[i - 1]}
    log(f"[fluxloop] {n} frames, {info['bpm']} bpm, beats {sorted(beats)}, world changes {sorted(wechsel)}")
    comfy = method_a.make_client("127.0.0.1:8188", "sdk", "http://127.0.0.1:8189")
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ys, xs = np.mgrid[0:hoehe, 0:breite].astype(np.float32)

    def flux(init, ref, i, dn):
        p = sdir / f"{i:05d}.png"
        if p.exists():
            return core.imread(p)
        a, b = comfy.upload(init, "dmk_fl_init.png"), comfy.upload(ref, "dmk_fl_ref.png")
        img = comfy.image(comfy.wait(comfy.queue(workflow(a, b, ANWEISUNG.format(welt=welten[welt[i]]), 2000 + i, dn)),
                                     timeout=1800))
        if img.shape[:2] != (hoehe, breite):
            img = cv2.resize(img, (breite, hoehe), interpolation=cv2.INTER_LANCZOS4)
        core.imwrite(p, img)
        return img

    t0, dauer = time.time(), []
    src0 = core.imread(quellen[0])
    cur = flux(src0, src0, 0, 1.0)              # erstes Bild: volle Verwandlung = erstes Welt-Keyframe
    ref = cur.copy()
    prev, letzt_sample, harte = cur.copy(), 0, True
    core.imwrite(fdir / "00000.png", core.film_finish(cur, 0, film, False, 7))
    q_alt = src0
    for i in range(1, n):
        src = core.imread(quellen[i])
        f = dis.calc(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.cvtColor(q_alt, cv2.COLOR_BGR2GRAY), None)
        mx, my = xs + f[..., 0], ys + f[..., 1]

        def mit(img):
            return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cur, prev, ref = mit(cur), mit(prev), mit(ref)   # Samples und Welt-Keyframe entlang der Videobewegung mitnehmen
        q_alt = src
        if i in beats or i - letzt_sample >= CADENCE:
            tk = time.time()
            if i in wechsel:                  # neue Welt: volles Keyframe aus dem Video-Frame wird die neue Referenz
                dn = 1.0
                s = flux(src, src, i, 1.0)
                ref = s.copy()
            else:
                dn = DN_BEAT if i in beats else DN_RUHE
                init = cv2.addWeighted(cur, 1.0 - QUELLZUG, src, QUELLZUG, 0)
                s = flux(init, ref, i, dn)
            dauer.append(time.time() - tk)
            if i not in beats:                # Glättung zwischen den Beats, auf dem Beat ungebremst
                s = cv2.addWeighted(cur, GLAETTUNG, s, 1.0 - GLAETTUNG, 0)
            prev, cur, letzt_sample, harte = cur, s, i, i in beats
            if i % 15 == 0 or i in wechsel:
                log(f"[fluxloop] {i}/{n} denoise {dn}, {dauer[-1]:.1f} s/sample")
        t = 1.0 if harte else min(1.0, (i - letzt_sample + 1) / CADENCE)
        frame = cur if t >= 1.0 else cv2.addWeighted(prev, 1.0 - t, cur, t, 0)
        core.imwrite(fdir / f"{i:05d}.png", core.film_finish(frame, i, film, False, 7))
    comfy.uploads_aufraeumen()
    core.frames_to_mp4(fdir, out / "video.mp4", 30)
    audio.ton_anlegen(out / "video.mp4", track, out / "video_ton.mp4", start, n / 30, n / 30)
    meta = {"frames": n, "samples": len(dauer) + 1, "s_pro_sample": round(float(np.mean(dauer)), 1) if dauer else None,
            "minuten": round((time.time() - t0) / 60, 1), "welten": welten, "denoise": [DN_RUHE, DN_BEAT, DN_WECHSEL],
            "cadence": CADENCE, "glaettung": GLAETTUNG, "modell": UNET, "lizenz": "FLUX.1 [dev] Non-Commercial License"}
    (out / "flux_loop.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"[fluxloop] done: {meta['minuten']} min, {meta['samples']} samples, {meta['s_pro_sample']} s/sample")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--sek", type=float, default=6)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--track", default=str(Path(__file__).resolve().parents[2] / "examples" / "signal_decay_20s.m4a"))
    ap.add_argument("--welten", required=True)
    a = ap.parse_args()
    render(Path(a.video), Path(a.out), a.sek, a.start, [w.strip() for w in a.welten.split("|") if w.strip()], Path(a.track))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
