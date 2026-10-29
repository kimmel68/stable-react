"""Rohschnitt des Demo-Videos (≤ 2:00) aus den finalen Clips – Schnitte auf der Eins, jeder Abschnitt mit seinem eigenen
Song-Ausschnitt, Texttafeln unten. 1920×1080, 30 fps. Platzhalter für die Bildschirmaufnahme der App.

    python -m dmk.demo_schnitt [--app <aufnahme.mp4>] [--out out/demo/dmk_demo_roh.mp4]
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from . import proc

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "out" / "final"
FONT = "C\\:/Windows/Fonts/segoeui.ttf"
FONT_B = "C\\:/Windows/Fonts/segoeuib.ttf"

# (Quelle, Sekunden ungefähr, Text oben, Text unten); Quelle = Job-Ordner (video_ton.mp4) oder "vorschau:<job>"
ABLAUF = [
    ("h1_kueste_flux", 12, "dmk — the music is a grammar, not a volume meter",
     "every count of the bar does something different · on count 1 the world changes"),
    ("n2_salzseen_flux", 10, "", "Flux Hybrid · FLUX.1 Kontext redraws the footage on the beat, the dmk loop makes it flow"),
    ("APP", 25, "", "two apps, same settings: open-weights SD 1.5 and FLUX.1 Kontext Hybrid · track or BPM only"),
    ("s2_bergsee", 9, "", "SD 1.5 · open weights · the worlds swell out of the footage, space and flow stay"),
    ("n4_flusscanyon", 8, "", "SD 1.5 · lake, frost, red rock, autumn — on count 1"),
    ("vorschau:h1_kueste_flux", 10, "", "source · depth (Depth Anything V2) · FLUX.1 Kontext keyframe · dmk result"),
    ("a2_arkade_flux", 9, "", "architecture · coral, ice, greenhouse, clockwork — the hall stays"),
    ("d2_felsnadeln_flux", 9, "", "depth-rich footage: the rock spire stays in front of the fjord in every world"),
    ("LANGLAUF", 10, "", "endless glide · never a cut — when the source runs out, its own frame becomes the new source"),
    ("n1_felskueste_flux", 8, "", "beats on the onset, change within one frame of count 1"),
    ("ENDE", 7, "dmk", "ComfyUI node pack · Comfy SDK · RTX 3070 Ti 8 GB · MIT + GPL-3.0 node pack · #OpenSourceHiggsfield"),
]


def _esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace(",", "\\,")


def _text(oben: str, unten: str) -> str:
    f = []
    if oben:
        f.append(f"drawtext=fontfile='{FONT_B}':text='{_esc(oben)}':fontsize=54:fontcolor=white:borderw=3:"
                 f"bordercolor=black@0.6:x=(w-text_w)/2:y=70")
    if unten:
        f.append(f"drawtext=fontfile='{FONT}':text='{_esc(unten)}':fontsize=34:fontcolor=white:box=1:boxcolor=black@0.45:"
                 f"boxborderw=14:x=(w-text_w)/2:y=h-110")
    return ("," + ",".join(f)) if f else ""


def _eins_fenster(job: Path, dauer: float) -> tuple[float, float]:
    """Start und Länge eines Ausschnitts, der auf einer Eins beginnt und vor/auf einer späteren Eins endet."""
    info = json.loads((job / "audio.json").read_text(encoding="utf-8")) if (job / "audio.json").exists() else {}
    t = [f / 30 for f in info.get("takt_frames", [])]
    if len(t) < 2:
        return 0.0, dauer
    start = t[0] if t[0] < 3 else 0.0
    ende = max([x for x in t if x - start <= dauer + 0.5] or [start + dauer])
    return start, max(4.0, ende - start)


def segment(i: int, eintrag, tmp: Path, app: Path | None) -> Path:
    quelle, dauer, oben, unten = eintrag
    ziel = tmp / f"{i:02d}.mp4"
    v = "scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps=30"
    ff = [proc.tool("ffmpeg"), "-y", "-v", "error"]
    if quelle in ("APP", "ENDE") and not (quelle == "APP" and app):
        titel = "[ screen recording of the app ]" if quelle == "APP" else ""
        ff += ["-f", "lavfi", "-i", f"color=c=0x101010:s=1920x1080:r=30:d={dauer}", "-f", "lavfi", "-i",
               f"anullsrc=r=48000:cl=stereo", "-t", str(dauer), "-vf",
               (f"drawtext=fontfile='{FONT}':text='{_esc(titel)}':fontsize=40:fontcolor=gray:x=(w-text_w)/2:y=(h-text_h)/2"
                if titel else "null") + _text(oben, unten)]
    else:
        if quelle == "APP":
            src, start = app, 0.0
        elif quelle == "LANGLAUF":
            src, start = ROOT / "out" / "nacht_pexels" / "09_langlauf_60s_farbe" / "video_ton.mp4", 0.0
        elif quelle.startswith("vorschau:"):
            job = F / quelle.split(":", 1)[1]
            src = job / "vorschau.mp4"
            start, dauer = _eins_fenster(job, dauer)
        else:
            job = F / quelle
            src = job / "video_ton.mp4"
            start, dauer = _eins_fenster(job, dauer)
        tempo = "setpts=PTS/4," if quelle == "LANGLAUF" else ""
        lese = dauer * 4 if quelle == "LANGLAUF" else dauer
        ff += ["-ss", f"{start:.3f}", "-t", f"{lese:.3f}", "-i", str(src), "-vf", tempo + v + _text(oben, unten)]
        ff += ["-t", f"{dauer:.3f}"]
    af = "afade=t=in:d=0.05,aresample=48000"
    if quelle == "LANGLAUF":   # 4× schneller: der Ton läuft mit (atempo), die Schläge bleiben am Bild
        af = "atempo=2,atempo=2," + af
    ff += ["-af", af,
           "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-ar", "48000", "-ac", "2", str(ziel)]
    proc.run(ff, f"Segment {i}")
    return ziel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="")
    ap.add_argument("--out", default=str(ROOT / "out" / "demo" / "dmk_demo_roh.mp4"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        teile = [segment(i, e, tmp, Path(a.app) if a.app else None) for i, e in enumerate(ABLAUF)]
        liste = tmp / "liste.txt"
        liste.write_text("".join(f"file '{p.as_posix()}'\n" for p in teile), encoding="utf-8")
        proc.run([proc.tool("ffmpeg"), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(liste), "-c", "copy",
                  "-movflags", "+faststart", str(out)], "Demo zusammensetzen")
    dauer = proc.run([proc.tool("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                     "Dauer").stdout
    print(out, dauer.decode().strip() if isinstance(dauer, bytes) else dauer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
