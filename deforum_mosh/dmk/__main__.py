"""CLI:  python -m dmk discover|synth|preflight|gpu|a|b|c|bench ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _common(p):
    p.add_argument("--seq-root", required=True, help="Ordner mit der Bildsequenz")
    p.add_argument("--out", required=True, help="Ausgabeordner")
    p.add_argument("--count", type=int, default=0, help="Anzahl Frames (0 = alle)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--zoom", default=None, help='Deforum-Keyframes, z. B. "0:(1.01), 60:(1.04)"')
    p.add_argument("--max-side", type=int, default=768, help="längste Kante in Pixeln")
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--border", default="reflect", choices=["reflect", "wrap", "replicate", "zero"])
    p.add_argument("--pattern", default=None, help=r'Regex auf Dateinamen in gemischten Ordnern, z. B. "LOOP_KREUZGANG_LUMEN2\."')
    p.add_argument("--depth-cache", default=None, help="Ordner für geschätzte Tiefenkarten (Standard: neben --out)")


def _args_a(p):
    p.add_argument("--alpha", type=float, default=0.3, help="Anteil des Eingangsframes im Init-Bild")
    p.add_argument("--denoise", default="0.45",
                   help='Zahl oder Schedule "0:(0.4), 40:(0.65), 42:(0.4)" (Motiv-Impuls an Prompt-Keyframes)')
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--seed-mode", default="iter", choices=["iter", "fixed"])
    p.add_argument("--angle", default="0:(0)")
    p.add_argument("--tx", default="0:(0)")
    p.add_argument("--ty", default="0:(0)")
    p.add_argument("--parallax", type=float, default=0.0, help="Parallaxen-Zoom über Depth-Maps (0 = aus)")
    p.add_argument("--depth-invert", action="store_true", help="Tiefenkarten: dunkel = nah")
    p.add_argument("--hf", type=float, default=0.0, help="Hochfrequenzen der Quelle zurückspeisen (0.5–1.0)")
    p.add_argument("--hf-sigma", type=float, default=2.0)
    p.add_argument("--hf-mode", default="replace", choices=["replace", "add"], help="add = akkumulierende Streifen (Streifen-Look)")
    p.add_argument("--stride", type=int, default=1, help="nur jeden n-ten Quellframe (mehr Bewegung pro Frame)")
    p.add_argument("--wrap", action="store_true", help="Loop-Sequenz zyklisch durchlaufen (mit --stride und --count)")
    p.add_argument("--motion", default="zoom", choices=["zoom", "flow", "kamera", "vektor", "virtuell"],
                   help="flow = mit dem optischen Fluss der Quelle weitertragen; kamera = dazu Kameramodell aus "
                        "Tiefe; vektor = exakte Bewegung aus dem Blender-Vector-Pass (exr/ neben beauty/)")
    p.add_argument("--rausch", default="zufall", choices=["zufall", "transport"],
                   help="transport = Startrauschen des Samplers mit der Bewegung mitführen (∫-Noise) statt würfeln")
    p.add_argument("--rausch-frisch", type=float, default=0.0, help="Grundanteil frisch gewürfelter Pixel pro Frame")
    p.add_argument("--rausch-rho", type=float, default=0.0,
                   help="Wandel-Regler für den Transport: Anteil frisches Rauschen pro Frame (0 = klebt/brennt ein, 1 = würfelt)")
    p.add_argument("--mono", action="store_true", help="Schwarzweiß-Ausgabe (vor Farbreferenz und Gedächtnis)")
    p.add_argument("--weich", default=0.0, help="Init vor dem Sampler weichzeichnen (Sigma px), Zahl oder Schedule")
    p.add_argument("--audio", default=None,
                   help="Audio-Bindungen: JSON-Datei oder Inline-JSON (siehe dmk/audio.py); bindet Parameter an den Track")
    p.add_argument("--kein-pump-fix", dest="pump_fix", action="store_false",
                   help="Zwischenframes nicht auf die Schärfe des Samples angleichen")
    p.add_argument("--kein-morph", dest="morph", action="store_false",
                   help="Cadence-Zwischenframes linear überblenden statt per Fluss-Morph")
    p.add_argument("--gedaechtnis", type=float, default=0.0,
                   help="mitbewegtes Gedächtnisbild: Anteil des Grobbands aus der eigenen Geschichte (0.3–0.7)")
    p.add_argument("--ged-beta", type=float, default=0.25, help="Lernrate des Gedächtnisses pro Sample")
    p.add_argument("--ged-sigma", type=float, default=12.0, help="Grobband-Grenze in Pixeln")
    p.add_argument("--kanten", type=float, default=0.0,
                   help="zweites ControlNet auf Geometriekanten (Stärke 0.3–0.7): Blender-Pass 'kanten' oder berechnet")
    p.add_argument("--kanten-end", type=float, default=0.8)
    p.add_argument("--antiblur", type=float, default=0.0, help="Unscharfmaske vor dem Sampler (0.2–0.4) gegen Schärfeverlust")
    p.add_argument("--noise", type=float, default=0.0, help="Korn vor dem Sampler (0.02–0.04)")
    p.add_argument("--flow-backend", default="auto", choices=["auto", "raft", "raft_small", "farneback"])
    p.add_argument("--licht", type=float, default=0.0,
                   help="Lichtkomposition der Quelle übernehmen (0.5–0.8): wo Licht/Schatten liegen, nicht wie hell")
    p.add_argument("--wandel", default="none",
                   help="Wandelkarte: fern/nah (aus der Tiefe) oder ids:6,5 (Blender-Objektindizes, die erhalten bleiben)")
    p.add_argument("--wandel-floor", type=float, default=0.35, help="Mindestanteil Wandel überall")
    p.add_argument("--first-denoise", type=float, default=0.0,
                   help="Denoise nur für Frame 0 (z. B. 0.8): legt Palette/Textur für --color-match gen0 fest")
    p.add_argument("--cadence", type=int, default=1,
                   help="nur jeder n-te Frame durch den Sampler, dazwischen tragen und überblenden (Deforum-Cadence)")
    p.add_argument("--controlnet", default="none", choices=["none", "depth", "canny", "tile"],
                   help="Raumstruktur per ControlNet festhalten (tile = Quellbild selbst als Führung)")
    p.add_argument("--quellzoom", default="0:(1.0)",
                   help='digitale Kamerafahrt in die Quelle, Zoom pro Frame als Schedule, z. B. "0:(1.006)"')
    p.add_argument("--quellzoom-zentrum", default=(0.5, 0.5),
                   type=lambda s: tuple(float(x) for x in s.split(",")), help="Fluchtpunkt als Bildanteil, z. B. 0.5,0.57")
    p.add_argument("--control-strength", type=float, default=0.8)
    p.add_argument("--control-gamma", type=float, default=1.0, help="Gamma auf die Tiefenkarte fürs ControlNet (<1 betont Ferne)")
    p.add_argument("--control-end", type=float, default=1.0,
                   help="ControlNet nur bis zu diesem Anteil des Rauschplans (z. B. 0.8: Grobstruktur fest, Feintextur frei)")
    p.add_argument("--control-net-name", default=None)
    p.add_argument("--mask", default=None, help="Maske (weiß = Bild, schwarz bleibt schwarz), z. B. Projektionskontur")
    p.add_argument("--prompt", default=None, help='Text oder Fahrplan "0: text a | 80: text b" (harter Wechsel am Keyframe)')
    p.add_argument("--negative", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--sampler", default=None)
    p.add_argument("--scheduler", default=None)
    p.add_argument("--server", default="127.0.0.1:8188")
    p.add_argument("--workflow", default=None, help="eigener ComfyUI-Workflow im API-Format")
    p.add_argument("--color-match", default="input", choices=["input", "first", "gen0", "none"],
                   help="gen0 = erster generierter Frame (hält den Kontrast des Modells)")
    p.add_argument("--dry-run", action="store_true", help="nur Warp + Blend, ohne Sampler")


def _args_b(p):
    p.add_argument("--gop", type=int, default=20)
    p.add_argument("--qscale", type=int, default=4)
    p.add_argument("--idrop", default="replace", choices=["replace", "remove", "keep"])
    p.add_argument("--dup-every", type=int, default=0)
    p.add_argument("--dup-times", type=int, default=0)
    p.add_argument("--loop-start", type=int, default=0)
    p.add_argument("--loop-len", type=int, default=0)
    p.add_argument("--loop-times", type=int, default=0)
    p.add_argument("--jitter", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mv-gain", dest="mv_gain_factor", type=float, default=1.0, help="braucht ffedit")


def _args_c(p):
    p.add_argument("--sub", type=int, default=1, help="Zwischenframes pro Quellframe")
    p.add_argument("--trail", type=float, default=0.85)
    p.add_argument("--occl", default="hold", choices=["hold", "fill", "none"])
    p.add_argument("--occl-thresh", type=float, default=1.5)
    p.add_argument("--backend", default="auto", choices=["auto", "raft", "raft_small", "farneback"])
    p.add_argument("--flow-side", type=int, default=768, help="längste Kante für die Flow-Berechnung")
    p.add_argument("--hf", type=float, default=0.0, help="Hochfrequenzen der Quelle zurückspeisen (0.5–1.0)")
    p.add_argument("--hf-sigma", type=float, default=2.0)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Umlaute auch in der cp1252-Konsole
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="dmk", description=f"Deforum / Datamosh / Toolkit {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("discover", help="Taglauf-/B-Sequenz-Ordner suchen")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--max-depth", type=int, default=6)

    p = sub.add_parser("synth", help="synthetische Kamerafahrt mit Tiefenkarten erzeugen")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--zoom", type=float, default=1.02)

    p = sub.add_parser("preflight", help="Umgebung prüfen")
    p.add_argument("--seq-root", default=None)
    p.add_argument("--server", default="127.0.0.1:8188")

    p = sub.add_parser("depth", help="Tiefenkarten einer Sequenz schätzen (Depth-Anything-V2-Small)")
    p.add_argument("--seq-root", required=True)
    p.add_argument("--out", required=True, help="Cache-Ordner")
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--pattern", default=None)

    p = sub.add_parser("gpu", help="auf freien VRAM warten")
    p.add_argument("--min-free", type=int, default=4000)
    p.add_argument("--server", default=None)
    p.add_argument("--timeout", type=float, default=3600)

    for name, adder, text in (("a", _args_a, "Deforum-Feedback-Loop über ComfyUI"),
                              ("b", _args_b, "Datamosh über den MPEG-4-Codec"),
                              ("c", _args_c, "Flow-Advektion")):
        p = sub.add_parser(name, help=text)
        _common(p)
        adder(p)

    p = sub.add_parser("rescore", help="Kennzahlen und Bericht eines Bench-Ordners neu berechnen (ohne Rendern)")
    _common(p)

    p = sub.add_parser("bench", help="alle Wege und Ketten vergleichen")
    _common(p)
    p.add_argument("--server", default="127.0.0.1:8188")
    p.add_argument("--config", default=None, help='JSON mit {"runs": [...]} statt der Standardläufe')
    p.add_argument("--only", nargs="+", default=None, help="nur diese Läufe (Namen)")

    a = ap.parse_args(argv)
    kw = {k: v for k, v in vars(a).items() if k not in ("cmd", "out", "config", "only") and v is not None}
    if isinstance(kw.get("audio"), str):
        s = kw["audio"]
        kw["audio"] = json.loads(Path(s).read_text(encoding="utf-8") if Path(s).is_file() else s)
        if "audio" in kw["audio"]:          # eine komplette Lauf-Konfiguration übergeben: nur den Audio-Teil nehmen
            kw["audio"] = kw["audio"]["audio"]

    if a.cmd == "discover":
        from .core import discover
        hits = discover(a.roots, a.max_depth)
        for h in hits:
            print(f"{h['frames']:6d} Frames  {h['size']:>10}  Tiefe: {'ja' if h['depth'] else 'nein':4}  {h['path']}")
        print(f"{len(hits)} Sequenzordner gefunden.")
        return 0
    if a.cmd == "synth":
        from . import synth
        print(synth.make(a.out, a.frames, zoom=a.zoom))
        return 0
    if a.cmd == "preflight":
        from . import preflight
        return 0 if preflight.check(a.seq_root, a.server) else 1
    if a.cmd == "depth":
        from . import core, depth
        seq = core.load_sequence(a.seq_root, a.count, a.start, a.pattern)
        paths = depth.estimate(seq, a.out)
        print(f"{len(paths)} Tiefenkarten in {paths[0].parent}")
        return 0
    if a.cmd == "gpu":
        from . import gpu
        return 0 if gpu.wait_free(a.min_free, timeout=a.timeout, server=a.server) else 1
    if a.cmd == "rescore":
        from . import bench
        bench.rescore(a.seq_root, a.out, a.count or 90, a.start, a.zoom or "0:(1.02)", a.pattern, a.depth_cache)
        return 0
    if a.cmd == "bench":
        from . import bench
        runs = None
        if a.config:
            runs = json.loads(Path(a.config).read_text(encoding="utf-8"))["runs"]
        if a.only:
            runs = [r for r in (runs or bench.DEFAULT_RUNS) if r["name"] in a.only]
        kw.setdefault("zoom", "0:(1.02)")
        kw.setdefault("count", 90)
        if not kw["count"]:
            kw["count"] = 90
        bench.run(out_dir=a.out, runs=runs, **kw)
        return 0

    from importlib import import_module
    meta = import_module(f".method_{a.cmd}", __package__).run(out_dir=a.out, **kw)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
