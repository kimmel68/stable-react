"""Vorschau-Video mit Tiefenkarte (Nutzer 28.10.: "Beispielvideos am besten mit Depth Map für die Vorschau").
SD 1.5:      Original | Tiefe (Depth Anything V2) | Ergebnis
Flux Hybrid: Original | Tiefe | Flux-Keyframes (Stufe 1) | Ergebnis
Darunter ein Taktstreifen: Beats als Striche, die Eins rot, Weltwechsel weiß. Ton vom Ergebnis.

    python -m dmk.vorschau <job> [<job> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import audio, core, proc

B, H = 640, 360


def _tiefe_ordner(job: Path) -> Path | None:
    kand = [d for d in (job / "_depth").glob("*") if d.is_dir() and any(d.glob("*_depth.png"))]
    return max(kand, key=lambda d: len(list(d.glob("*_depth.png")))) if kand else None


def _feld(img: np.ndarray, titel: str) -> np.ndarray:
    f = cv2.resize(img, (B, H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(f, (0, 0), (B, 26), (0, 0, 0), -1)
    cv2.putText(f, titel, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return f


def _tiefe_farbe(p: Path) -> np.ndarray:
    d = core.read_depth(p, None, False).astype(np.float32)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def bauen(job: Path) -> Path:
    job = Path(job)
    frames = core.list_images(job / "frames")
    n = len(frames)
    # bevorzugt die tatsächlich benutzte Quelle (mit Kamera-Ausschnitt) und ihre Tiefe, sonst das zerlegte Video
    hyb = (job / "flux" / "frames").is_dir()     # Hybrid: "quelle" sind die Flux-Bilder, Original ist das Video
    qdir = next((job / d for d in (("quelle_video",) if hyb else ("quelle", "quelle_video"))
                 if (job / d).is_dir() and core.list_images(job / d)), None)
    quellen = core.list_images(qdir) if qdir else []
    if (job / "quelle_depth").is_dir() and any((job / "quelle_depth").iterdir()):
        tiefen = sorted(p for p in (job / "quelle_depth").iterdir() if p.suffix.lower() == ".png")
    else:
        td = _tiefe_ordner(job)
        tiefen = sorted(td.glob("*_depth.png")) if td else []
    flux = core.list_images(job / "flux" / "frames") if (job / "flux" / "frames").is_dir() else []
    params = json.loads((job / "params.json").read_text(encoding="utf-8")) if (job / "params.json").exists() else {}
    if not quellen and params.get("quelle") and Path(params["quelle"]).is_file():
        quellen = [Path(params["quelle"])] * n
    info = json.loads((job / "audio.json").read_text(encoding="utf-8")) if (job / "audio.json").exists() else {}
    beats, takte = set(info.get("beat_frames", [])), set(info.get("takt_frames", []))
    wechsel = set(info.get("welt_wechsel", []))
    spalten = 4 if flux else 3
    out = job / "_vorschau"
    out.mkdir(exist_ok=True)
    streifen_h = 22
    for i in range(n):
        teile = [_feld(core.imread(quellen[min(i, len(quellen) - 1)]), "source") if quellen else np.zeros((H, B, 3), np.uint8),
                 _feld(_tiefe_farbe(tiefen[min(i, len(tiefen) - 1)]), "depth (Depth Anything V2)") if tiefen
                 else np.zeros((H, B, 3), np.uint8)]
        if flux:
            teile.append(_feld(core.imread(flux[min(i, len(flux) - 1)]), "FLUX.1 Kontext keyframes"))
        teile.append(_feld(core.imread(frames[i]), "dmk result"))
        oben = np.hstack(teile[:2]) if spalten == 4 else np.hstack(teile)
        bild = np.vstack([oben, np.hstack(teile[2:])]) if spalten == 4 else oben
        s = np.zeros((streifen_h, bild.shape[1], 3), np.uint8)
        fenster = 150                                  # 5 s um den aktuellen Frame
        for k in range(max(0, i - fenster // 2), min(n, i + fenster // 2)):
            x = int((k - (i - fenster // 2)) / fenster * bild.shape[1])
            if k in wechsel:
                cv2.line(s, (x, 0), (x, streifen_h), (255, 255, 255), 3)
            elif k in takte:
                cv2.line(s, (x, 2), (x, streifen_h - 2), (60, 60, 255), 2)
            elif k in beats:
                cv2.line(s, (x, 6), (x, streifen_h - 6), (200, 200, 200), 1)
        cv2.line(s, (bild.shape[1] // 2, 0), (bild.shape[1] // 2, streifen_h), (0, 220, 255), 1)
        core.imwrite(out / f"{i:05d}.png", np.vstack([bild, s]))
    ziel = job / "vorschau.mp4"
    core.frames_to_mp4(out, job / "_vorschau_stumm.mp4", 30)
    ton = job / "video_ton.mp4"
    if ton.exists():
        proc.run([proc.tool("ffmpeg"), "-y", "-v", "error", "-i", str(job / "_vorschau_stumm.mp4"), "-i", str(ton), "-map",
                  "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart", str(ziel)], "Vorschau")
    else:
        (job / "_vorschau_stumm.mp4").replace(ziel)
    return ziel


def main() -> int:
    for j in sys.argv[1:]:
        print(bauen(Path(j)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
