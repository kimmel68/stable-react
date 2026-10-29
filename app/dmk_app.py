"""dmk App: audioreaktive Deforum-Bildwelten aus Bild, Video oder 3D-Pässen.

Zwei Apps, ein Code (Nutzer 27.10.: "2 Anwendungen, selbe Einstellungsmöglichkeiten"):
  app.bat        dmk · SD 1.5      (http://127.0.0.1:7860) – nur offene Gewichte
  app_flux.bat   dmk · Flux Hybrid (http://127.0.0.1:7861) – FLUX.1 Kontext [dev] zeichnet die Welten, SD 1.5 den Fluss
Rechnet lokal über ComfyUI; Aufträge laufen über das Comfy SDK (comfy-api-proxy vor der lokalen ComfyUI).
"""
from __future__ import annotations

import copy
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deforum_mosh"))

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from dmk import audio as audio_mod  # noqa: E402
from dmk import core, hybrid, method_a, proc  # noqa: E402

# Engine der App: "sd15" (Standard) oder "flux" (Flux Hybrid) – per Startdatei (DMK_ENGINE) oder --engine flux
ENGINE = "flux" if (os.environ.get("DMK_ENGINE", "").lower() == "flux" or "--engine=flux" in sys.argv
                    or ("--engine" in sys.argv and sys.argv[sys.argv.index("--engine") + 1:][:1] == ["flux"])) else "sd15"
CFG = ROOT / "configs"
OUT = ROOT / "out" / "app"
VENV = ROOT / ".venv" / "Scripts"
COMFY, PROXY = "127.0.0.1:8188", "http://127.0.0.1:8189"
BASIS_FPS = 30

NEGATIV = ("people, person, figures, text, watermark, color, blurry, lowres, flat")
# Look "Film colour" (Nutzer 21.10.: Variante A des Look-Tests): fotografische Prompts statt "crisp, high contrast",
# CFG 5,5–7 statt 9, kaum Nachschärfung, Farben des Ausgangsbilds, Film-Finish – am wenigsten KI-Look (docs/LOOK_TEST.md)
FOTO = (", photograph, shot on 35mm film, Kodak Portra 400, natural light, soft contrast, subtle film grain, "
        "slight lens imperfections, documentary photography")
NEGATIV_FOTO = ("people, person, figures, text, watermark, blurry, lowres, cgi, 3d render, illustration, painting, "
                "digital art, hdr, oversaturated, glossy, plastic, overly detailed, oversharpened")
ARCH_SCHWANZ = ", strong one-point perspective, vaulted interior, crisp architectural detail, high contrast"


def foto(prompt: str) -> str:
    return prompt.replace(ARCH_SCHWANZ, "").replace(", crisp detail, high contrast", "").rstrip(", ") + FOTO


def farbigkeit(quelle) -> float:
    """Mittlere Sättigung × Helligkeit des Ausgangsmaterials (0..1): Bild oder erster Frame eines Frame-Ordners."""
    q = Path(quelle)
    f = q if q.is_file() else next(iter(core.list_images(q)), None) if q.is_dir() else None
    img = cv2.imread(str(f)) if f is not None else None
    if img is None:
        return 1.0
    hsv = cv2.cvtColor(cv2.resize(img, (256, 144)), cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    return float((hsv[..., 1] * hsv[..., 2]).mean())


PALETTEN = ["Auto", "From the footage", "Per world"]


def _look_farbe(p: dict, a: dict | None, quelle=None, palette: str = "Auto") -> None:
    # Farben vom Ausgangsmaterial – außer es ist grau oder dunkel (Probeclips 21.10.: Morgennebel und Nacht blieben grau
    # bzw. schwarz): dann bringt jede Welt ihre eigene Palette mit (gen0 pro Welt, Look-Test Variante B)
    cm = "gen0" if quelle is not None and farbigkeit(quelle) < 0.09 else "input"
    cm = {"From the footage": "input", "Per world": "gen0"}.get(palette, cm)   # Fluss 0,035, Nacht 0,051, Park 0,072 | Skyline 0,119, Oberbaum 0,277
    p.update(color_match=cm, negative=NEGATIV_FOTO, antiblur=0.08, schaerfe_halten=0.35, prompt=foto(p["prompt"]))
    if a:
        a["bindungen"]["cfg"] = {"quelle": "huelle", "min": 5.5, "max": 7.0, "glatt": 3, "lokal": True}
        if a.get("welt_wechsel"):
            a["welt_wechsel"]["welten"] = [foto(w) for w in a["welt_wechsel"]["welten"]]


def _audio_teil(name: str) -> dict:
    return json.loads((CFG / name).read_text(encoding="utf-8"))["audio"]


REZEPTE = {
    "4/4 bar grammar – 1 zoom, 2 blur, 3 source pull, 4 transform": "recipe_bar_grammar.json",
    "4/4 bar grammar – stronger, melt on count 1": "recipe_bar_melt.json",
    "Flow – calm, one pulse per beat": "recipe_flow.json",
    "Hard – blur and melt on every hit": "recipe_hard.json",
}
STANDARD_REZEPT = next(iter(REZEPTE))
WELTEN = json.loads((CFG / "welten.json").read_text(encoding="utf-8"))
# Weltenfolgen (Nutzer 27.10.: "Naturaufnahmen funktionieren sehr gut"): Natur zuerst, Architektur mit klar
# unterscheidbaren Welten; die Auswahl füllt das Welten-Feld, das man frei ändern kann
WELTENSETS = json.loads((CFG / "weltensets.json").read_text(encoding="utf-8"))
STANDARD_SET = "Coast"
WELTEN_MAX = 10   # Nutzer 29.10.: ein Prompt-Feld je Welt, weitere per "+ Add world", die Welten rotieren


def welten_aus_feldern(n, felder) -> str:
    """Die sichtbaren, nicht leeren Prompt-Felder als Weltenliste (eine Welt je Zeile, in dieser Reihenfolge)."""
    return "\n".join(" ".join(str(t).split()) for t in list(felder)[:int(n or 0)] if t and str(t).strip())
MODELLE = [
    ("SD 1.5 DreamShaper 8", "image model", "CreativeML OpenRAIL-M", "open"),
    ("ControlNet v1.1 depth (control_v11f1p_sd15_depth)", "keeps the room", "CreativeML OpenRAIL-M", "open"),
    ("Depth-Anything-V2-Small", "depth estimation", "Apache-2.0", "open"),
    ("RAFT (torchvision)", "optical flow for video", "BSD-3-Clause", "open"),
]
MODELLE_FLUX = [
    ("FLUX.1 Kontext [dev] fp8 (Comfy-Org)", "draws the worlds on the beat", "FLUX.1 [dev] Non-Commercial", "open"),
    ("T5-XXL fp8 + CLIP-L, Flux VAE", "text encoders, VAE", "Apache-2.0 / MIT / FLUX.1 [dev] NC", "open"),
]
FLUX_KEYS = list(hybrid.KEYS)          # Every beat / Every bar / World change only
SD_S_PRO_FRAME = {512: 1.0, 640: 1.5, 768: 2.15}   # RTX 3070 Ti, Cadence 3; 768: gemessen 27.10. (2,1–2,3 s/Frame)

BASIS = {   # Kern der Produktionseinstellung, ohne projektspezifische Maske, IDs und Prompt
    "zoom": "0:(1.0)", "stride": 1, "controlnet": "depth", "control_strength": 1.0, "hf": 0.0, "alpha": 0.1,
    "denoise": 0.4, "first_denoise": 0.75, "seed_mode": "iter", "color_match": "gen0", "cadence": 3, "licht": 0.8,
    "wandel": "none", "cfg": 9.0, "antiblur": 0.25, "noise": 0.0, "control_end": 1.0, "quellzoom": "0:(1.003)",
    "gedaechtnis": 0.7, "ged_beta": 0.3, "pump_fix": True, "quellzoom_zentrum": (0.5, 0.5), "max_side": 768,
    # Schärfe (Nutzer 20.10.): Unschärfe nur im ausgegebenen Bild, Loop bleibt scharf; jedes Sample auf die Schärfe des
    # ersten Bildes zurück. Gemessen: Schärfe bleibt 2000-2600 statt Absturz auf 9.
    "weich_modus": "anzeige", "schaerfe_halten": 0.8,
    "beat_hart": True, "referenz_bloecke": "output",   # harter Schnitt auf den Beat; Referenz nur in Ausgangsblöcken
}
# Glättung gegen Brodeln in ALLEN Modi (20.10.): Kamerafahrt Cadence 3 scharf + Glättung = Bestwert
# (Brodeln 7.8 -> 4.7, Ruck95 0.31 -> 0.20, Schärfe bleibt, Beat/Ruhe 3.0 -> 5.3; docs/PLAN_DREI_WEGE.md)
# Glättung auf dem Beat ganz aus (Nachtsuche 20./21.10.: Kombi Ruhe 3,33, Schlag 78 %, Struktur 0,85)
GLAETTUNG = {"quelle": "beat", "min": 0.5, "max": 0.0, "release": 2}
# "Frame to frame" (Cadence 1) bleibt in lauf_params für Skripte, aber nicht mehr in der App: neigt zum Mosaik und kam
# in keinem guten Clip vor (Berater-Review 27.10.)
MODI = ["Camera move", "Rigid frame"]
BEAT_QUELLEN = ["Track", "BPM only"]
WECHSEL = {"Off": 0, "Every bar": 1, "Every 2 bars": 2, "Every 4 bars": 4}   # Weltwechsel auf der Eins (21.10.)
FOLGE_STANDARD = ["stich", "beton", "myzel", "kristall"]
# Fahrt-Tempo 1–5 (Nutzer 21.10.: "beides hat was, kommt auf die Stimmung an", "darf noch schneller, noch wilder sein"):
# Grundgeschwindigkeit pro Frame und Faktor auf die Beat-/Takt-Schübe; 0 = nur die kurzen Zoom-Pulse
TEMPO = {1: (1.002, 1.0), 2: (1.003, 1.0), 3: (1.0045, 1.2), 4: (1.0065, 1.6), 5: (1.009, 2.2)}


def tempo_werte(t: float) -> tuple[float, float]:
    """Zwischenstufen (z. B. 2,5) linear zwischen den ganzen Stufen."""
    t = float(np.clip(t, 1.0, 5.0))
    a, b = int(np.floor(t)), int(np.ceil(t))
    f = t - a
    return (TEMPO[a][0] + f * (TEMPO[b][0] - TEMPO[a][0]), TEMPO[a][1] + f * (TEMPO[b][1] - TEMPO[a][1]))
# Pfad = Fokuspunkt im Bild über die Zeit; die Kamera zoomt um ihn und driftet so zu seiner Seite
# "Follow the depth" sucht selbst (fernster strukturierter Bereich), "Direction" folgt Richtung + Drift (Nutzer 21.10.:
# Pan-Richtung und Geschwindigkeit in der App festlegen), "Sweep" pendelt, "Straight in" zoomt auf den Fluchtpunkt.
PFADE = {"Follow the depth": None, "Direction": None,
         "Sweep": [[0.15, 0.6], [0.85, 0.4], [0.15, 0.55], [0.85, 0.45], [0.5, 0.5]], "Straight in": None}
RICHTUNGEN = {"Right": (1, 0), "Left": (-1, 0), "Up": (0, -1), "Down": (0, 1), "Up-right": (0.7, -0.7),
              "Up-left": (-0.7, -0.7), "Down-right": (0.7, 0.7), "Down-left": (-0.7, 0.7)}
ANKER_ZOOM = 1.5   # endlose Fahrt: alle 1,5x wird das eigene Bild zur neuen Quelle (kein Sprung, Nutzer 21.10.)
STOPP = threading.Event()   # Cancel-Knopf
A_ = audio_mod.ANZEIGE
WELT_NAMEN = {"koralle": "Coral basilica", "tusche": "Ink wash hall", "beton": "Brutalist concrete", "kristall": "Crystal palace",
              "uhrwerk": "Clockwork hall", "eis": "Ice palace", "gewebe": "Woven textile hall", "papier": "Paper cut diorama",
              "quecksilber": "Mercury hall", "myzel": "Mycelium cathedral", "stich": "Copperplate engraving",
              "chipstadt": "Circuit-board city", "gewaechshaus": "Overgrown greenhouse", "wurzel": "Root hall",
              "knochen": "Bone ossuary"}


# ------------------------------------------------------------------ Dienste

def _lebt(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=3).read()
        return True
    except Exception:  # noqa: BLE001
        return False


COMFY_VERSION = "0.39.1"   # getestete Version (eingefroren bis zur Abgabe)


def _json(url: str):
    try:
        return json.loads(urllib.request.urlopen(url, timeout=5).read())
    except Exception:  # noqa: BLE001
        return None


def fremde_auftraege() -> int:
    """Aufträge in der ComfyUI-Warteschlange, die nicht von dmk stammen (Nacht 21./22.10.: ein anderes Projekt belegte
    die ComfyUI 2 h mit SDXL; unsere Frames warteten bis zum Zeitlimit)."""
    q = _json(f"http://{COMFY}/queue")
    if not q:
        return 0
    n = 0
    for e in q.get("queue_running", []) + q.get("queue_pending", []):
        wf = e[2] if len(e) > 2 and isinstance(e[2], dict) else {}
        eigen = any(str(k).startswith("dmk_") for k in wf) or any(
            "dreamshaper_8" in str(v.get("inputs", {}).get("ckpt_name", "")).lower()
            or "flux1-dev-kontext" in str(v.get("inputs", {}).get("unet_name", "")).lower()
            for v in wf.values() if isinstance(v, dict))
        if not eigen:
            n += 1
    return n


def dienste_status() -> str:
    """Startprüfung: ComfyUI (Version, fremde Instanz?), Proxy, ffmpeg, Modelle."""
    teile = []
    st = _json(f"http://{COMFY}/system_stats")
    if st is None:
        teile.append(f"ComfyUI {COMFY}: off")
    else:
        v = st.get("system", {}).get("comfyui_version", "?")
        teile.append(f"ComfyUI {v}" + ("" if v == COMFY_VERSION else f" (WARNING: tested with {COMFY_VERSION} – another "
                                                                     f"ComfyUI may be using port {COMFY.split(':')[1]})"))
        ck = _json(f"http://{COMFY}/object_info/CheckpointLoaderSimple")
        cn = _json(f"http://{COMFY}/object_info/ControlNetLoader")

        def namen(info, node, feld):
            try:
                spec = info[node]["input"]["required"][feld]
                return spec[0] if isinstance(spec[0], list) else spec[1].get("options", [])
            except Exception:  # noqa: BLE001
                return []
        hat_ck = any("dreamshaper_8" in x.lower() for x in namen(ck, "CheckpointLoaderSimple", "ckpt_name"))
        hat_cn = any("sd15_depth" in x.lower() for x in namen(cn, "ControlNetLoader", "control_net_name"))
        teile.append("DreamShaper 8 " + ("ok" if hat_ck else "MISSING"))
        teile.append("depth ControlNet " + ("ok" if hat_cn else "MISSING"))
        if ENGINE == "flux":
            un = _json(f"http://{COMFY}/object_info/UNETLoader")
            hat_fx = any(hybrid.UNET.lower() in x.lower() for x in namen(un, "UNETLoader", "unet_name"))
            teile.append("FLUX.1 Kontext " + ("ok" if hat_fx else "MISSING (install.ps1 -Flux)"))
    fremd = fremde_auftraege()
    if fremd:
        teile.append(f"ComfyUI BUSY: {fremd} jobs from another program in the queue – renders wait until they are done")
    teile.append("Comfy API v2 (comfy-api-proxy): " + ("running" if _lebt(f"{PROXY}/api/v2/jobs") else "off"))
    teile.append("ffmpeg " + ("ok" if shutil.which("ffmpeg") else "MISSING (install ffmpeg and add it to PATH)"))
    return "  ·  ".join(teile)


def dienste_starten() -> str:
    if not _lebt(f"http://{COMFY}/system_stats"):
        # ohne Konsolenfenster, Ausgabe in eine Logdatei: ein Klick ins Konsolenfenster (Markiermodus) hält sonst jede
        # Ausgabe an und friert ComfyUI ein (passiert am 20.10.)
        logs = ROOT / "out" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        subprocess.Popen([str(VENV / "python.exe"), "main.py", "--listen", "127.0.0.1", "--port", COMFY.split(":")[1],
                          "--disable-auto-launch", "--highvram", "--cache-ram", "8", "8"], cwd=ROOT / "ComfyUI",
                         stdout=open(logs / "comfy.log", "ab"), stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(90):
            time.sleep(2)
            if _lebt(f"http://{COMFY}/system_stats"):
                break
    if not _lebt(f"{PROXY}/api/v2/jobs"):
        subprocess.run([str(VENV / "comfy-api-proxy.exe"), "start", "--comfyui", f"http://{COMFY}", "--port", "8189",
                        "--comfyui-base-dir", str(ROOT / "ComfyUI")], cwd=ROOT, capture_output=True)
        time.sleep(3)
    if not _lebt(f"{PROXY}/api/v2/jobs"):
        # "start" scheitert an einer veralteten PID-Datei (26.10.) → den Proxy direkt im Hintergrund laufen lassen
        logs = ROOT / "out" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        subprocess.Popen([str(VENV / "python.exe"), "-m", "comfy_api_proxy.cli", "run", "--comfyui", f"http://{COMFY}",
                          "--host", "127.0.0.1", "--port", "8189", "--max-upload-mb", "100",
                          "--comfyui-base-dir", str(ROOT / "ComfyUI")], cwd=ROOT,
                         stdout=open(logs / "proxy.log", "ab"), stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(15):
            time.sleep(1)
            if _lebt(f"{PROXY}/api/v2/jobs"):
                break
    return dienste_status()


# ------------------------------------------------------------------ Aufbau eines Laufs

def _rezept(name: str, track: str | None, zoom_staerke: float, beat_lock: bool = True,
            modus: str = "Camera move") -> dict | None:
    if not track:
        return None
    a = copy.deepcopy(_audio_teil(REZEPTE.get(name, REZEPTE[STANDARD_REZEPT])))
    a["datei"] = track
    a["basis_fps"] = BASIS_FPS
    a["offset"] = 0.0
    if beat_lock:   # Sample genau auf jedem Beat erzwingen: sonst wartet ein Beat bei Cadence 3 bis zu 2 Frames (66 ms)
        a["takt_sample"] = {"quelle": "beat", "schwelle": 0.5}
    zb = a["bindungen"].get("quellzoom")
    for b in (zb if isinstance(zb, list) else [zb] if zb else []):
        b["min"] = 1.0 + (b["min"] - 1.0) * zoom_staerke
        b["max"] = 1.0 + (b["max"] - 1.0) * zoom_staerke
    bd = a["bindungen"]
    if isinstance(bd.get("weich"), list):          # nur das Verschwimmen auf dem Schlag, keine Dauer-Weichheit
        bd["weich"] = bd["weich"][0]
    if bd.get("weich"):                             # setzt mit dem Schlag ein, nach einem Frame wieder scharf
        bd["weich"]["attack"], bd["weich"]["release"] = 0, 1
    if modus == "Rigid frame":                      # keine Kamerafahrt: Verwandlung auf der Stelle
        bd.pop("quellzoom", None)
    for b in (zb if isinstance(zb, list) else [zb] if zb else []):
        # Zoom-Puls mit 2 Frames Anlauf: Gegenprobe 20.10. – ohne Anlauf brodelt es zwischen den Beats 26 % mehr
        # (3,68 -> 4,63), der Schlag sitzt durch den harten Schnitt trotzdem im Beat-Frame (74 %)
        b["attack"] = 3   # Nachtsuche 20./21.10.: 3 Frames Anlauf ruhiger (3,68 -> 3,45) bei gleichem Schlag
        b["release"] = max(float(b.get("release", 3)), 5)
    # Gedächtnis-Dip auf dem Beat verworfen (Gegenprobe: kein Gewinn beim Schlag, etwas mehr Unruhe)
    if beat_lock:   # Referenz-Aufmerksamkeit (Nutzer 20.10.: "deutlich seriöser"): zwischen den Beats an, auf dem Beat aus
        bd["referenz"] = {"quelle": "beat", "min": 1.0, "max": 0.0, "release": 2}
    if modus == "Frame to frame":                   # jeder Frame ein Sample, weniger Neuzeichnen pro Frame
        a["takt_sample"] = None
        for b in bd["denoise"] if isinstance(bd.get("denoise"), list) else []:
            b["min"] = 0.30
        # KEIN Seed pro Takt: der brennt bei Cadence 1 nach wenigen Frames zu einem Mosaik ein (20.10., out/varianten_ruhe)
    # gegen Brodeln: jedes Sample mit dem mitbewegten Vorbild mischen – zwischen den Beats stark, auf dem Beat schwach
    bd["glaettung"] = dict(GLAETTUNG)
    return a


def _video_zu_frames(video: str, ziel: Path, sekunden: float) -> tuple[Path, str | None]:
    ziel.mkdir(parents=True, exist_ok=True)
    proc.run([proc.tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-t", f"{sekunden:.3f}",
              "-vf", f"fps={BASIS_FPS},scale='min(960,iw)':-2", str(ziel / "%05d.png")], "Video zerlegen")
    ton = ziel.parent / "ton_aus_video.wav"
    try:
        proc.run([proc.tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-vn", "-ac", "2",
                  str(ton)], "Ton aus Video")
        return ziel, str(ton)
    except proc.ProcError:
        return ziel, None


def lauf_params(quelle_art, bild, video, ordner, track, rezept, welt, eigener_prompt, sekunden, zoom_staerke,
                zentrum_y, mono, aufloesung, maske, ordner_job: Path, beat_lock: bool = True,
                modus: str = "Camera move", wechsel: str = "Off", folge=None, tempo: int = 0,
                pfad: str = "Follow the depth", versatz_ms: float = 0.0,
                raster: bool = False, film: float = 0.0, richtung: str = "Right",
                drift: float = 0.6, eigene_welten: str = "", palette: str = "Auto", track_start: float = 0.0,
                akzente=None, engine: str = "sd15", beat_quelle: str = "Track",
                bpm: float = 120.0, tiefe_bild=None) -> tuple[dict, str, int, str | None]:
    n = max(12, int(round(sekunden * BASIS_FPS)))
    if beat_quelle == "BPM only":   # Taktspur erzeugen statt Audiodatei – derselbe Analyseweg, das Gitter sitzt exakt
        bpm = float(np.clip(float(bpm or 120), 40, 220))
        track = str(audio_mod.klickspur(Path(ordner_job) / f"takt_{bpm:g}bpm.wav", bpm, sekunden + 1.0))
        track_start = 0.0
    p = dict(BASIS)
    prompt = (eigener_prompt or "").strip() or WELTEN.get(welt, next(iter(WELTEN.values())))
    p["film"] = float(film or 0)   # analoger Abschluss nur fürs Ausgabebild: Halation, Schulter, Vignette, Korn
    p.update(prompt=prompt, negative=NEGATIV, mono=bool(mono), max_side=int(aufloesung),
             quellzoom_zentrum=(0.5, float(zentrum_y)), backend="sdk", sdk_url=PROXY)
    if maske:
        p["mask"] = maske
    if quelle_art == "Image":
        if not bild:
            raise gr.Error("Please choose an image.")
        quelle, p["motion"] = bild, "virtuell"
        if tiefe_bild:   # eigene Tiefenkarte (Blender, Unreal, prozedural): ersetzt die Schätzung
            p["tiefe_bild"] = str(tiefe_bild)
    elif quelle_art == "Video":
        if not video:
            raise gr.Error("Please choose a video.")
        fr, ton = _video_zu_frames(video, ordner_job / "quelle_video", sekunden)
        quelle, p["motion"] = str(fr), "kamera"
        track = track or ton
        n = min(n, len(core.list_images(fr)))
    else:
        if not ordner or not Path(ordner).is_dir():
            raise gr.Error("Please enter a folder with 3D passes (beauty/ and depth/, optional exr/).")
        quelle = ordner
        exr = (Path(ordner) / "exr").is_dir()
        p["motion"] = "vektor" if exr else "kamera"
    if modus == "Rigid frame":
        p["quellzoom"] = "0:(1.0)"
    if modus == "Frame to frame":
        p.update(cadence=1, denoise=0.30)
    a = _rezept(rezept, track, float(zoom_staerke), bool(beat_lock), modus)
    eigene = [z.strip() for z in (eigene_welten or "").splitlines() if z.strip()]
    if eigene:                      # eigene Welten (ein Prompt pro Zeile) ersetzen Welt und Weltenfolge
        p["prompt"] = eigene[0]
    if a:
        a["offset"] = float(track_start or 0)
        _akzente(a, akzente)
        a["bild_versatz_ms"] = float(versatz_ms or 0)
        if beat_quelle == "BPM only":   # festes Gitter ab dem ersten Kick; Tracker-Bereich um das Tempo (sonst halbiert
            a["raster"] = {"bpm": bpm, "t0": 0.5}   # er z. B. 150 bpm auf 75)
            a["bpm"] = (bpm * 0.8, bpm * 1.25)
        elif raster:   # festes BPM-Gitter statt der einzeln erkannten Schläge (dmk.beatraster)
            a["raster"] = "an"
        alle = WECHSEL.get(wechsel, 0)
        if alle:
            _weltwechsel(a, welt, eigener_prompt, folge, alle)
            if eigene:
                a["welt_wechsel"]["welten"] = eigene
        if float(tempo or 0) > 0 and modus != "Rigid frame" and a["bindungen"].get("quellzoom"):
            _fahrt(p, a, float(tempo), pfad, quelle, quelle_art, int(aufloesung), float(zentrum_y), bool(alle),
                   richtung, float(drift))
        p["audio"] = a
    if not mono:
        _look_farbe(p, a, quelle, palette)
    p["_engine"] = "flux" if engine == "flux" else "sd15"
    return p, quelle, n, track


def ausfuehren(p: dict, quelle, n: int, job: Path, engine: str = "sd15", flux_keys: str = "Every beat", log=print,
               abbruch=None) -> dict:
    """Ein Lauf für beide Apps. SD 1.5: die dmk-Schleife direkt auf der Quelle. Flux Hybrid: erst zeichnet FLUX.1
    Kontext die Welten (nach demselben Takt-Fahrplan), dann läuft die SD-Schleife über diesem Flux-Video."""
    p = {k: v for k, v in p.items() if not k.startswith("_")}
    seq = quelle
    if engine == "flux":
        if p.get("motion") == "vektor" or (Path(str(quelle)).is_dir() and (Path(str(quelle)) / "beauty").is_dir()):
            raise ValueError("The Flux Hybrid works with an image or a video, not with 3D passes.")
        pl = hybrid.plan(p, n, flux_keys, job / "flux")
        if Path(str(quelle)).is_file():   # Standbild: Flux zeichnet auf jedem Weltwechsel den aktuellen Blick der Fahrt um
            log(f"[flux] still image: world changes on frames {pl['wechsel']} are drawn over the current view")
            seq, anker_fn = hybrid.bild_hybrid(quelle, pl, FOTO, job / "flux", log=log)
            p = hybrid.sd_stufe(p, quelle)
            p["welt_anker"] = anker_fn
        else:
            log(f"[flux] stage 1: {len(pl['keys'])} keyframes, world changes on frames {pl['wechsel']}")
            seq = hybrid.flux_stufe(quelle, job / "flux", n, pl, FOTO, log=log, abbruch=abbruch)
            p = hybrid.sd_stufe(p, quelle)   # Raum und Bewegung vom Original (Nutzer 27.10.)
        log("[flux] stage 2: dmk loop (SD 1.5) over the Flux video")
    return method_a.run(out_dir=job, count=n, start=0, fps=BASIS_FPS, seq_root=str(seq), depth_cache=job / "_depth",
                        log=log, abbruch=abbruch, **p)   # Tiefen-Cache pro Job: gleich benannte Frames teilen ihn sonst


def schaetzung(engine: str, sekunden: float, aufloesung: int, flux_keys: str = "Every beat", bpm: float = 110.0,
               quelle_art: str = "Image") -> str:
    """Grobe Renderzeit auf einer RTX 3070 Ti (8 GB)."""
    n = sekunden * BASIS_FPS
    sd = n * SD_S_PRO_FRAME.get(int(aufloesung), 2.6) / 60
    if engine != "flux":
        return f"≈ {sd:.0f} min on an RTX 3070 Ti (8 GB)"
    beats = sekunden * bpm / 60
    keys = {"Every beat": beats, "Every bar": beats / 4, "World change only": beats / 4}.get(flux_keys, beats) + 1
    if quelle_art == "Image":
        keys = min(keys, 5)
    fx = keys * hybrid.FLUX_S / 60
    return f"≈ {fx + sd:.0f} min on an RTX 3070 Ti (8 GB): Flux ≈ {fx:.0f} min ({keys:.0f} images) + flow ≈ {sd:.0f} min"


AKZENT_QUELLEN = {0: ("takt", "takt1"), 1: ("takt2",), 2: ("takt3",), 3: ("takt4",)}


def _akzente(a: dict, akzente) -> None:
    """Beat-Akzente: Stärke der Zählzeiten 1–4 (Faktor auf die Spanne jeder Bindung, die an dieser Zählzeit hängt).
    Rezept-Standard: 1 = Zoom-Schub, 2 = Verschwimmen, 3 = Quellzug, 4 = Verwandeln."""
    for k, f in enumerate(akzente or ()):
        if f is None or abs(float(f) - 1.0) < 1e-6:
            continue
        for b in a["bindungen"].values():
            for x in (b if isinstance(b, list) else [b]):
                if isinstance(x, dict) and x.get("quelle") in AKZENT_QUELLEN[k]:
                    x["max"] = float(x["min"]) + (float(x["max"]) - float(x["min"])) * float(f)


def _weltwechsel(a: dict, welt: str, eigener_prompt: str, folge, alle: int) -> None:
    """Prompt springt auf jeder `alle`-ten Eins zur nächsten Welt; im Wechsel stärker neu zeichnen (+0.2) und das
    Gedächtnis loslassen, damit die neue Welt sofort steht (gemessen 21.10.: Struktur 0,84, Schlag 82 %)."""
    namen = [w for w in (folge or []) if w in WELTEN]
    if len(namen) < 2:
        namen = [welt] + [w for w in FOLGE_STANDARD if w != welt][:3]
    prompts = [WELTEN[w] for w in namen]
    if (eigener_prompt or "").strip():
        prompts = [eigener_prompt.strip()] + prompts
    a["welt_wechsel"] = {"quelle": "takt", "alle": alle, "welten": prompts}
    b = a["bindungen"]
    dn = b["denoise"] if isinstance(b.get("denoise"), list) else [b["denoise"]]
    b["denoise"] = dn + [{"quelle": "wechsel", "min": float(dn[0]["min"]), "max": float(dn[0]["min"]) + 0.2, "release": 2}]
    g = b.get("gedaechtnis")
    if g:
        g = g if isinstance(g, list) else [g]
        b["gedaechtnis"] = g + [{"quelle": "wechsel", "min": float(g[0]["min"]), "max": 0.0, "release": 4}]


def _fahrt(p: dict, a: dict, tempo: float, pfad: str, quelle, quelle_art: str, aufloesung: int, zentrum_y: float,
           mit_wechsel: bool, richtung: str = "Right", drift: float = 0.6) -> None:
    """Große Fahrt: stetige Grundgeschwindigkeit, Beat-/Takt-Schübe obendrauf, Fokus wandert den Pfad entlang. Ist die
    Kamera so tief im Bild, wie die Quelle Pixel hat, fliegt sie im eigenen generierten Bild weiter (Anker) – endlos,
    ohne Sprung."""
    grund, schub = tempo_werte(tempo)
    zb = a["bindungen"]["quellzoom"]
    zb = zb if isinstance(zb, list) else [zb]
    for x in zb:
        x["max"] = grund + (float(x["max"]) - float(x["min"])) * schub
        x["min"] = grund
    a["bindungen"]["quellzoom"] = zb
    punkte = PFADE.get(pfad)
    if pfad == "Direction":   # Fokus neben der Mitte: die Kamera zoomt darauf zu und driftet so in diese Richtung
        dx, dy = RICHTUNGEN.get(richtung, (1, 0))
        f = (float(np.clip(0.5 + dx * drift * 0.42, 0.08, 0.92)), float(np.clip(zentrum_y + dy * drift * 0.42, 0.08, 0.92)))
        punkte = [list(f), list(f)]
    p["quellzoom_zentrum"] = punkte if punkte else (0.5, zentrum_y)
    if pfad == "Follow the depth":
        p["fokus_tiefe"] = 1.0
    if quelle_art == "Image":
        img = cv2.imread(str(quelle))
        lang = max(img.shape[:2]) if img is not None else aufloesung
    else:
        lang = 960                                         # Videos werden auf 960 px zerlegt
    p["quellzoom_max"] = float(max(1.3, lang / aufloesung))   # so tief, wie das Quellbild Pixel hat
    p["anker_zoom"] = ANKER_ZOOM


# ------------------------------------------------------------------ Aktionen

def _takt_track(beat_quelle, track, bpm, sekunden, job: Path):
    """(Track, Start) für die Analyse: eigene Datei oder erzeugte Taktspur ("BPM only")."""
    if beat_quelle == "BPM only":
        b = float(np.clip(float(bpm or 120), 40, 220))
        return str(audio_mod.klickspur(job / f"takt_{b:g}bpm.wav", b, float(sekunden) + 1.0)), b
    if not track:
        raise gr.Error("Please load a track – or choose 'BPM only'.")
    return track, None


def kurven_ansehen(beat_quelle, track, bpm, rezept, sekunden, zoom_staerke, track_start=0.0, ak1=1.0, ak2=1.0, ak3=1.0,
                   ak4=1.0):
    n = max(12, int(round(sekunden * BASIS_FPS)))
    job = OUT / "_kurven"
    job.mkdir(parents=True, exist_ok=True)
    track, b = _takt_track(beat_quelle, track, bpm, sekunden, job)
    spec = _rezept(rezept, track, float(zoom_staerke))
    spec["offset"] = 0.0 if b else float(track_start or 0)
    if b:
        spec["raster"], spec["bpm"] = {"bpm": b, "t0": 0.5}, (b * 0.8, b * 1.25)
    _akzente(spec, (ak1, ak2, ak3, ak4))
    _, _, _, info = audio_mod.aufloesen(spec, n, 1, 0, job, log=lambda s: None)
    text = (f"{info['bpm']} bpm · {len(info['beat_frames'])} beats and {len(info['takt_frames'])} bars in {sekunden:.0f} s · "
            + " · ".join(f"{A_.get(k, k)} ← {' + '.join(A_.get(x, x) for x in v['quelle'].split(' + '))} ({v['min']}–{v['max']})"
                         for k, v in info["bindungen"].items()))
    return str(job / "audio_merkmale.png"), str(job / "audio_kurven.png"), text


def rendern(quelle_art, bild, video, ordner, beat_quelle, track, bpm, klick_behalten, sekunden, track_start, weltenset,
            welten_text, wechsel, modus, tempo, pfad, richtung, drift, flux_keys, mono, palette, film, aufloesung,
            rezept, ak1, ak2, ak3, ak4, raster, versatz_ms, zoom_staerke, zentrum_y, maske, tiefe_bild=None,
            progress=gr.Progress()):
    if fremde_auftraege():
        gr.Warning("ComfyUI is busy with jobs from another program – this render starts when they are done.")
    if not _lebt(f"{PROXY}/api/v2/jobs") or not _lebt(f"http://{COMFY}/system_stats"):
        raise gr.Error("ComfyUI or comfy-api-proxy is not running – press 'Start services' first.")
    if ENGINE == "flux" and quelle_art == "3D passes":
        raise gr.Error("The Flux Hybrid works with an image or a video.")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    job = OUT / f"{stamp}_{ENGINE}_{str(weltenset or 'own').split()[0].lower()}"
    job.mkdir(parents=True, exist_ok=True)
    welten = (welten_text or "").strip() or "\n".join(WELTENSETS[STANDARD_SET])
    p, quelle, n, track = lauf_params(quelle_art, bild, video, ordner, track, rezept, "beton", "", sekunden, zoom_staerke,
                                      zentrum_y, mono, aufloesung, maske, job, True, modus, wechsel, None, tempo, pfad,
                                      versatz_ms, raster, film, richtung, drift, welten, palette, track_start,
                                      (ak1, ak2, ak3, ak4), engine=ENGINE, beat_quelle=beat_quelle, bpm=bpm,
                                      tiefe_bild=tiefe_bild if quelle_art == "Image" else None)
    (job / "params.json").write_text(json.dumps(dict(p, quelle=quelle, frames=n, flux_keys=flux_keys), indent=1,
                                                ensure_ascii=False, default=str), encoding="utf-8")
    zeilen: queue.Queue = queue.Queue()
    ergebnis: dict = {}
    STOPP.clear()

    def arbeite():
        try:
            ergebnis["meta"] = ausfuehren(p, quelle, n, job, ENGINE, flux_keys, log=zeilen.put, abbruch=STOPP.is_set)
        except KeyboardInterrupt:
            ergebnis["fehler"] = "Render cancelled. Frames so far are in " + str(job)
        except Exception as e:  # noqa: BLE001
            import traceback
            ergebnis["fehler"] = f"{type(e).__name__}: {e}"
            (job / "fehler.txt").write_text(traceback.format_exc(), encoding="utf-8")
        zeilen.put(None)

    threading.Thread(target=arbeite, daemon=True).start()
    log = []
    while True:
        try:   # jede Sekunde den Fortschritt aus den Frame-Ordnern lesen, nicht nur bei Log-Zeilen
            z = zeilen.get(timeout=1.0)
        except queue.Empty:
            z = ""
        if z is None:
            break
        if z:
            log.append(z)
        fr = len(core.list_images(job / "frames")) if (job / "frames").exists() else 0
        if ENGINE == "flux":
            fx = len(core.list_images(job / "flux" / "frames")) if (job / "flux" / "frames").exists() else 0
            progress(min((fx + fr) / (2 * n), 1.0), desc=f"Flux {fx}/{n} · flow {fr}/{n} frames")
        else:
            progress(min(fr / n, 1.0), desc=f"{fr}/{n} frames")
        if z:
            yield None, None, "\n".join(log[-14:])
    if "fehler" in ergebnis:
        raise gr.Error(ergebnis["fehler"])
    mit_ton = (job / "video_ton.mp4").exists() and not (beat_quelle == "BPM only" and not klick_behalten)
    vid = job / ("video_ton.mp4" if mit_ton else "video.mp4")
    m = ergebnis["meta"]
    log.append(f"done: {m['frames']} frames, {m['sec_per_frame']} s/frame, backend {m.get('backend')} → {job}")
    yield str(vid), str(job / "kontakt.png"), "\n".join(log[-14:])


# ------------------------------------------------------------------ Oberfläche

KOPF = {
    "sd15": ("Stable React · SD 1.5", "# Stable React · SD 1.5 — audio-reactive generative worlds\n"
             "An image or a video plus a track (or just a BPM) → a Deforum-style feedback loop that glides through the "
             "picture without a single cut, keeps space and flow, and plays a **4/4 bar grammar**: count 1 zoom push and "
             "world change, 2 blur snap, 3 source pull, 4 transform. Open-weights models only, running locally through "
             "ComfyUI and the Comfy SDK."),
    "flux": ("Stable React · Flux Hybrid", "# Stable React · Flux Hybrid — audio-reactive generative worlds\n"
             "**FLUX.1 Kontext [dev]** redraws your footage as a new world on the beat – same camera, same shapes – and the "
             "dmk loop (SD 1.5) lets it swell and flow between the beats. Same settings as the SD 1.5 app. Runs locally "
             "through ComfyUI and the Comfy SDK. *The FLUX.1 [dev] weights are under the FLUX.1 [dev] Non-Commercial "
             "License – please read its terms before you use the results.*"),
}


def baue() -> gr.Blocks:
    titel, kopf = KOPF[ENGINE]
    flux = ENGINE == "flux"
    quellen = ["Image", "Video"] if flux else ["Image", "Video", "3D passes"]
    with gr.Blocks(title=titel) as ui:
        gr.Markdown(kopf)
        with gr.Row():
            status = gr.Textbox(value=dienste_status, label="Services", interactive=False, scale=4)
            start_btn = gr.Button("Start services", scale=1)
        with gr.Row():
            with gr.Column(scale=1):
                quelle_art = gr.Radio(quellen, value="Image", label="Source")
                bild = gr.Image(type="filepath", label="Image (photo or render – aerial views and landscapes work great)")
                tiefe_bild = gr.Image(type="filepath", label="Depth map (optional – your own, white = near; "
                                                             "otherwise Depth Anything V2 estimates it)", height=160)
                video = gr.Video(label="Video (drone footage works great)", visible=False)
                ordner = gr.Textbox(label="Folder with 3D passes from any 3D tool: beauty/ and depth/ as numbered "
                                          "PNG frames (depth: white = near, 8 or 16 bit); optional exr/ with Blender "
                                          "multilayer EXR (Depth, Vector, Object Index) for exact motion – without it "
                                          "the motion is measured from the beauty frames", visible=False)
                beat_quelle = gr.Radio(BEAT_QUELLEN, value="Track", label="Beat source")
                track = gr.Audio(type="filepath", label="Track (any audio file – the beats and the bar are detected)")
                with gr.Row():
                    bpm = gr.Number(value=120, minimum=40, maximum=220, label="BPM", visible=False)
                    klick_behalten = gr.Checkbox(value=True, label="Keep the click track in the result", visible=False)
                with gr.Row():
                    sekunden = gr.Slider(2, 60, value=10, step=1, label="Length (s)")
                    track_start = gr.Number(value=0, label="Start in the track (s)", minimum=0)
                zeit = gr.Markdown(schaetzung(ENGINE, 10, 512))
            with gr.Column(scale=1):
                with gr.Accordion("Worlds", open=True):
                    weltenset = gr.Dropdown(list(WELTENSETS) + ["Own"], value=STANDARD_SET,
                                            label="World set (fills the worlds below – edit them freely)")
                    start_welten = WELTENSETS[STANDARD_SET]
                    welt_felder = [gr.Textbox(value=start_welten[k] if k < len(start_welten) else "", lines=1,
                                              max_lines=3, visible=k < len(start_welten),
                                              label=f"World {k + 1}" + (" – start world" if k == 0 else ""),
                                              placeholder="e.g. aerial photograph of the same valley covered in snow")
                                   for k in range(WELTEN_MAX)]
                    welten_n = gr.State(len(start_welten))
                    with gr.Row():
                        welt_plus = gr.Button("+ Add world", size="sm")
                        welt_minus = gr.Button("− Remove last", size="sm")
                    gr.Markdown("On every world change the next world follows; after the last one it starts again "
                                "with world 1.")
                    wechsel = gr.Dropdown(list(WECHSEL), value="Every 2 bars", label="World change on count 1")
                with gr.Accordion("Camera", open=True):
                    modus = gr.Radio(MODI, value=MODI[0], label="Mode (rigid frame = no camera move)")
                    with gr.Row():
                        tempo = gr.Slider(0, 5, value=2, step=0.5, label="Glide speed (0 = short zoom pulses, 5 = wild)")
                        pfad = gr.Dropdown(list(PFADE), value="Follow the depth",
                                           label="Camera path (Follow the depth searches by itself)")
                    with gr.Row():
                        richtung = gr.Dropdown(list(RICHTUNGEN), value="Right", label="Direction", visible=False)
                        drift = gr.Slider(0, 1, value=0.6, step=0.05, label="Sideways drift (0 = straight in)",
                                          visible=False)
                if flux:
                    with gr.Accordion("Flux", open=True):
                        flux_keys = gr.Dropdown(FLUX_KEYS, value="Every beat",
                                                label="Flux draws a new image … (every beat = most precise, every bar = "
                                                      "about 4× faster)")
                else:
                    flux_keys = gr.State("Every beat")
                with gr.Accordion("Look", open=False):
                    with gr.Row():
                        mono = gr.Checkbox(value=False, label="Black and white (off = film colour)")
                        palette = gr.Dropdown(PALETTEN, value="Auto", label="Colour palette")
                    with gr.Row():
                        film = gr.Slider(0, 1, value=1.0, step=0.1, label="Film finish (grain, halation – output only)")
                        aufloesung = gr.Dropdown([512, 640, 768], value=512, label="Resolution (long side)")
                with gr.Accordion("Advanced: beat grammar and timing", open=False):
                    rezept = gr.Dropdown(list(REZEPTE), value=STANDARD_REZEPT, label="Beat recipe")
                    with gr.Row():
                        ak1 = gr.Slider(0, 2, value=1.0, step=0.1, label="Count 1 – zoom push")
                        ak2 = gr.Slider(0, 2, value=1.0, step=0.1, label="Count 2 – blur")
                    with gr.Row():
                        ak3 = gr.Slider(0, 2, value=1.0, step=0.1, label="Count 3 – source pull")
                        ak4 = gr.Slider(0, 2, value=1.0, step=0.1, label="Count 4 – transform")
                    with gr.Row():
                        raster = gr.Checkbox(value=False, label="Snap beats to a fixed grid (steady electronic tracks)")
                        versatz_ms = gr.Slider(0, 200, value=0, step=5, label="Picture delay (ms, e.g. Bluetooth)")
                    with gr.Row():
                        zoom_staerke = gr.Slider(0, 2.5, value=1.0, step=0.1, label="Zoom pulse strength")
                        zentrum_y = gr.Slider(0.3, 0.7, value=0.5, step=0.01, label="Vanishing point (height)")
                    maske = gr.Image(type="filepath", label="Mask (optional: white = transform, black = keep)")
                with gr.Row():
                    kurven_btn = gr.Button("Preview curves")
                    render_btn = gr.Button("Render", variant="primary")
                    stopp_btn = gr.Button("Cancel", variant="stop")
        kurven_text = gr.Markdown()
        with gr.Row():
            merkmale_img = gr.Image(label="Audio features", interactive=False)
            kurven_img = gr.Image(label="Parameter curves", interactive=False)
        with gr.Row():
            ergebnis = gr.Video(label="Result with sound")
            kontakt = gr.Image(label="Contact sheet", interactive=False)
        log = gr.Textbox(label="Log", lines=10)
        with gr.Accordion("Models (open weights)", open=False):
            gr.Dataframe(value=[list(m) for m in (MODELLE_FLUX + MODELLE if flux else MODELLE)],
                         headers=["Model", "Role", "License", "Weights"], interactive=False)

        def quelle_wechsel(art):   # Bild: die Kamera gleitet; Video: das Video bewegt sich selbst, nur Zoom-Pulse
            return (gr.update(visible=art == "Image"), gr.update(visible=art == "Video"),
                    gr.update(visible=art == "3D passes"), gr.update(value=2 if art == "Image" else 0),
                    gr.update(visible=art == "Image"))
        quelle_art.change(quelle_wechsel, quelle_art, [bild, video, ordner, tempo, tiefe_bild])
        beat_quelle.change(lambda q: (gr.update(visible=q == "Track"), gr.update(visible=q == "BPM only"),
                                      gr.update(visible=q == "BPM only"), gr.update(visible=q == "Track")),
                           beat_quelle, [track, bpm, klick_behalten, track_start])
        def set_waehlen(s):   # ein Set füllt die Felder, "Own" lässt sie stehen
            if s not in WELTENSETS:
                return [gr.update()] * (WELTEN_MAX + 1)
            w = WELTENSETS[s][:WELTEN_MAX]
            return [len(w)] + [gr.update(value=w[k] if k < len(w) else "", visible=k < len(w))
                               for k in range(WELTEN_MAX)]

        def welt_dazu(n):
            n = min(WELTEN_MAX, int(n) + 1)
            return [n] + [gr.update(visible=k < n) for k in range(WELTEN_MAX)]

        def welt_weg(n):
            n = max(1, int(n) - 1)
            return [n] + [gr.update(visible=k < n) if k < n else gr.update(visible=False, value="")
                          for k in range(WELTEN_MAX)]

        weltenset.change(set_waehlen, weltenset, [welten_n] + welt_felder)
        welt_plus.click(welt_dazu, welten_n, [welten_n] + welt_felder)
        welt_minus.click(welt_weg, welten_n, [welten_n] + welt_felder)
        pfad.change(lambda p: (gr.update(visible=p == "Direction"), gr.update(visible=p == "Direction")), pfad,
                    [richtung, drift])
        zeit_ein = [sekunden, aufloesung, flux_keys, bpm, quelle_art]
        for c in (sekunden, aufloesung, bpm, quelle_art) + ((flux_keys,) if flux else ()):
            c.change(lambda s, a, k, b, q: schaetzung(ENGINE, s, a, k, b or 110, q), zeit_ein, zeit)
        start_btn.click(dienste_starten, None, status)
        kurven_btn.click(kurven_ansehen, [beat_quelle, track, bpm, rezept, sekunden, zoom_staerke, track_start,
                                          ak1, ak2, ak3, ak4], [merkmale_img, kurven_img, kurven_text])
        stopp_btn.click(lambda: STOPP.set(), None, None)
        def rendern_ui(*a, progress=gr.Progress()):   # Welten-Felder -> Weltenliste (an Position von welten_text)
            a = list(a)
            felder, a = a[-WELTEN_MAX:], a[:-WELTEN_MAX]
            a[11] = welten_aus_feldern(a[11], felder)
            yield from rendern(*a, progress=progress)

        render_btn.click(rendern_ui, [quelle_art, bild, video, ordner, beat_quelle, track, bpm, klick_behalten, sekunden,
                                      track_start, weltenset, welten_n, wechsel, modus, tempo, pfad, richtung, drift,
                                      flux_keys, mono, palette, film, aufloesung, rezept, ak1, ak2, ak3, ak4, raster,
                                      versatz_ms, zoom_staerke, zentrum_y, maske, tiefe_bild] + welt_felder,
                         [ergebnis, kontakt, log])
    return ui


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    baue().queue().launch(server_name="127.0.0.1", server_port=7861 if ENGINE == "flux" else 7860,
                          allowed_paths=[str(ROOT / "out")], show_error=True)
