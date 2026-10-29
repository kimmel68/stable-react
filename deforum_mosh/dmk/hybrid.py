"""Flux Hybrid (Nutzer 26./27.10.: "Flow, aber präzise Flux-Inhalte"; eigene App neben SD 1.5, gleiche Einstellungen).

Stufe 1: FLUX.1 Kontext [dev] zeichnet Keyframes aus dem Quellmaterial – Welt und Zeitpunkt kommen aus DERSELBEN
Audio-Auflösung (audio.aufloesen), die danach auch die SD-Schleife steuert; so liegt jeder Weltwechsel in beiden Stufen
auf demselben Frame (26.10.: eigene Wechselregel der Flux-Stufe lag sonst neben der Eins der SD-Stufe → blasse Lava).
Zwischen den Keyframes trägt der optische Fluss des Videos das letzte Keyframe weiter (bei einem Bild: steht es).
Stufe 2: die dmk-Schleife (SD 1.5) läuft mit weniger Neuzeichnen und mehr Quellzug über dem Flux-Video – Inhalt von
Flux, Aufquellen und Fließen aus der SD-Rückkopplung (Test 26.10.: Ruhe 1,62, Schlag 100 %, Struktur 0,86).

Lizenz der Flux-Gewichte: FLUX.1 [dev] Non-Commercial License.
"""
from __future__ import annotations

import copy
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import audio, core, method_a
from .flux_bonus import ANWEISUNG, GUIDANCE, UNET, fluss, workflow  # noqa: F401  (UNET: Modellprüfung der App)

KEYS = {"Every beat": "beat", "Every bar": "takt", "World change only": "wechsel"}
FLUX_LANG = 768                       # lange Seite der Flux-Bilder (768×432 bei 16:9: ~35 s pro Bild auf 8 GB)
FLUX_S = 35.0                         # Sekunden pro Flux-Bild auf der RTX 3070 Ti (gemessen 26.10.)
# Raumtreue (Nutzer 27.10.: "die räumliche Vorlage soll immer dieselbe bleiben"): bei Denoise 1,0 komponiert Kontext
# drastische Welten neu (Arktis: Kantenkorrelation zum Originalframe 0,03); Denoise 0,95 startet vom Latent des
# Originalframes – Küstenlinie und Felsen bleiben an ihrem Platz (0,47–0,82), die Welt ist trotzdem klar erkennbar.
# 0,9 und darunter ändert nur noch Flecken (dmk.flux_raumtest, out/final/h1_kueste_flux/raumtest)
ANWEISUNG_RAUM = ("Transform this photograph into {welt}. Keep the exact same layout, the same shapes, edges and "
                  "openings in exactly the same places, the same perspective, camera angle and framing; change only "
                  "materials, colours, light and surface. Photograph.")
DENOISE = 0.95


def freigeben(server: str = "127.0.0.1:8188") -> None:
    """Flux aus VRAM und RAM nehmen, bevor die SD-Stufe lädt (12-GB-Modell auf 8-GB-Karte)."""
    try:
        req = urllib.request.Request(f"http://{server}/free", data=json.dumps({"unload_models": True, "free_memory": True})
                                     .encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:  # noqa: BLE001
        pass


def _roh(prompt: str, schwanz: str) -> str:
    """Welt-Text ohne den Foto-Zusatz der SD-Stufe (Flux bekommt die Bearbeitungsanweisung)."""
    return prompt[: -len(schwanz)] if schwanz and prompt.endswith(schwanz) else prompt


def plan(p: dict, n: int, modus: str = "Every beat", job: Path | None = None) -> dict:
    """Keyframes und Welt pro Frame aus der Audio-Spezifikation der SD-Stufe."""
    a = p.get("audio")
    welten = [p["prompt"]]
    welt = [0] * n
    beats, takte, wechsel = [], [], []
    if a:
        kurven, _, mm, info = audio.aufloesen(a, n, 1, 0, job, log=lambda s: None)
        beats = [int(b) for b in info["beat_frames"]]
        takte = [int(t) for t in info["takt_frames"]]
        wechsel = [int(w) for w in info.get("welt_wechsel", [])]
        if "_prompt" in kurven:
            welten = list(a["welt_wechsel"]["welten"])
            welt = [welten.index(x) for x in kurven["_prompt"]]
    art = KEYS.get(modus, "beat")
    keys = {0} | set(wechsel)
    if art == "beat":
        keys |= set(beats)
    elif art == "takt":
        keys |= set(takte)
    return {"keys": sorted(k for k in keys if k < n), "welt": welt, "welten": welten, "wechsel": wechsel}


def schaetzung_min(p: dict, n: int, modus: str, quelle_ist_bild: bool) -> float:
    pl = plan(p, n, modus)
    aufrufe = len(set(pl["welt"])) if quelle_ist_bild else len(pl["keys"])
    return round(aufrufe * FLUX_S / 60, 1)


def flux_stufe(quelle, out: Path, n: int, pl: dict, schwanz: str = "", steps: int = 20, log=print, abbruch=None,
               dry: bool = False) -> Path:
    """Flux-Keyframes zeichnen und entlang der Videobewegung mitnehmen → Frame-Ordner (n Bilder) für die SD-Stufe."""
    out.mkdir(parents=True, exist_ok=True)
    fdir, kdir = out / "frames", out / "keys"
    fdir.mkdir(exist_ok=True)
    kdir.mkdir(exist_ok=True)
    q = Path(quelle)
    bild = q.is_file()
    quellen = [q] * n if bild else core.list_images(q)[:n]
    n = len(quellen)
    h0, w0 = core.imread(quellen[0]).shape[:2]
    s = FLUX_LANG / max(h0, w0)
    breite, hoehe = int(round(w0 * s / 16) * 16), int(round(h0 * s / 16) * 16)
    roh = [_roh(w, schwanz) for w in pl["welten"]]
    keys = set(pl["keys"])
    log(f"[flux] {n} frames, {len(keys)} keyframes, worlds {len(roh)}, {breite}×{hoehe}")
    comfy = None if dry else method_a.make_client("127.0.0.1:8188", "sdk", "http://127.0.0.1:8189")
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ys, xs = np.mgrid[0:hoehe, 0:breite].astype(np.float32)
    pro_welt: dict[int, np.ndarray] = {}
    key_bild = key_quelle = None
    t0 = time.time()
    for i in range(n):
        if abbruch is not None and abbruch():
            raise KeyboardInterrupt
        ziel = fdir / f"{i:05d}.png"
        src = cv2.resize(core.imread(quellen[i]), (breite, hoehe), interpolation=cv2.INTER_AREA)
        wi = pl["welt"][i]
        if i in keys or key_bild is None:
            k_png = kdir / f"{i:05d}_w{wi}_raum2.png"
            if bild and wi in pro_welt:                       # Standbild: pro Welt genügt ein Flux-Bild
                img = pro_welt[wi]
            elif k_png.exists():
                img = core.imread(k_png)
            elif dry:
                img = src.copy()
            else:
                tk = time.time()
                name = comfy.upload(src, "dmk_flux_in.png")
                wf = workflow(name, ANWEISUNG_RAUM.format(welt=roh[wi]), 1000 + wi, steps, GUIDANCE)
                wf["10"]["inputs"]["denoise"] = DENOISE
                img = comfy.image(comfy.wait(comfy.queue(wf), timeout=1800))
                if img.shape[:2] != (hoehe, breite):
                    img = cv2.resize(img, (breite, hoehe), interpolation=cv2.INTER_LANCZOS4)
                log(f"[flux] keyframe {i} world {wi + 1}: {time.time() - tk:.0f} s")
            core.imwrite(k_png, img)
            pro_welt[wi] = img
            key_bild, key_quelle = img, src
            frame = img
        elif bild:
            frame = key_bild
        else:
            f = fluss(dis, key_quelle, src)
            frame = cv2.remap(key_bild, xs + f[..., 0], ys + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        core.imwrite(ziel, frame)
    if comfy is not None:
        comfy.uploads_aufraeumen()
        freigeben()
    (out / "flux_stufe.json").write_text(json.dumps({"frames": n, "keyframes": sorted(keys), "welten": roh,
                                                     "minuten": round((time.time() - t0) / 60, 1),
                                                     "modell": "flux1-dev-kontext_fp8_scaled",
                                                     "lizenz": "FLUX.1 [dev] Non-Commercial License"},
                                                    indent=1, ensure_ascii=False), encoding="utf-8")
    return fdir


def flux_bild(img: np.ndarray, text: str, seed: int, steps: int = 20) -> np.ndarray:
    """Ein Bild raumtreu in eine Welt umzeichnen (Kontext, Denoise 0,95, Material-Anweisung)."""
    comfy = method_a.make_client("127.0.0.1:8188", "sdk", "http://127.0.0.1:8189")
    h, w = img.shape[:2]
    s = FLUX_LANG / max(h, w)
    src = cv2.resize(img, (int(round(w * s / 16) * 16), int(round(h * s / 16) * 16)), interpolation=cv2.INTER_AREA)
    wf = workflow(comfy.upload(src, "dmk_flux_in.png"), ANWEISUNG_RAUM.format(welt=text), seed, steps, GUIDANCE)
    wf["10"]["inputs"]["denoise"] = DENOISE
    out = comfy.image(comfy.wait(comfy.queue(wf), timeout=1800))
    comfy.uploads_aufraeumen()
    return out


def bild_hybrid(quelle, pl: dict, schwanz: str, out: Path, log=print):
    """Standbild im Flux Hybrid: Startbild = Welt 1 von Flux; auf jedem Weltwechsel zeichnet Flux den aktuellen Blick der
    Fahrt um (method_a: welt_anker). Gibt (Startbild-Pfad, Rückruf für method_a) zurück."""
    out.mkdir(parents=True, exist_ok=True)
    roh = [_roh(w, schwanz) for w in pl["welten"]]
    start = out / "welt_0.png"
    if not start.exists():
        w0 = pl["welt"][0]
        core.imwrite(start, flux_bild(core.imread(Path(quelle)), roh[w0], 1000 + w0))
    zaehler = {"n": 0}

    def welt_anker(bild, prompt):
        wi = pl["welten"].index(prompt)
        t0 = time.time()
        neu = flux_bild(bild, roh[wi], 1000 + wi)
        zaehler["n"] += 1
        core.imwrite(out / f"wechsel_{zaehler['n']:02d}_w{wi}.png", neu)
        log(f"[flux] world {wi + 1} drawn over the current view: {time.time() - t0:.0f} s")
        return neu
    return start, welt_anker


def sd_stufe(p: dict, fuehrung=None) -> dict:
    """Parameter der SD-Schleife über dem Flux-Video: weniger Neuzeichnen, mehr Zug zum Flux-Bild, im Weltwechsel das
    neue Flux-Bild fast ganz übernehmen (Werte aus dem Test 26.10., out/flux_sd/hybrid4_kueste_6s)."""
    p = copy.deepcopy(p)
    if not (fuehrung and Path(str(fuehrung)).is_file()):   # Bild: virtuelle Kamera mit endloser Fahrt bleibt
        p["motion"] = "kamera"
    if fuehrung:   # Tiefe und Bewegung vom Original, nicht von den Flux-Bildern (method_a: fuehrung)
        p["fuehrung"] = str(fuehrung)
    p["color_match"] = "input"
    a = p.get("audio")
    if not a:
        p["denoise"] = max(0.2, float(p.get("denoise", 0.4)) - 0.1)
        p["alpha"] = float(p.get("alpha", 0.1)) + 0.2
        return p
    b = a["bindungen"]
    dn = b["denoise"] if isinstance(b["denoise"], list) else [b["denoise"]]
    for x in dn:
        if x.get("quelle") != "wechsel":
            x["min"], x["max"] = round(float(x["min"]) - 0.1, 3), round(float(x["max"]) - 0.1, 3)
    b["denoise"] = dn
    al = b["alpha"] if isinstance(b["alpha"], list) else [b["alpha"]]
    for x in al:
        x["min"], x["max"] = float(x["min"]) + 0.2, float(x["max"]) + 0.1
    if a.get("welt_wechsel"):
        al = al + [{"quelle": "wechsel", "min": float(al[0]["min"]), "max": 0.85, "release": 3}]
    b["alpha"] = al
    return p
