"""Starke Abschnitte in einem Song finden (Nutzer 27.10.: "starke 15-Sekunden-Abschnitte mit gutem Spannungsverlauf").
Jeder Kandidat beginnt auf einer Eins. Wertung: klarer Beat (Anschlag auf dem Beat gegen dazwischen), genug Energie und ein
Spannungsverlauf – die Energie steigt im Abschnitt oder es gibt einen Einsatz (Drop) im mittleren Teil, auf den ein
Weltwechsel fallen kann. Schreibt eine Übersicht (PNG), Hörproben und abschnitte.json.

    python -m dmk.song_abschnitte <song> [<song> ...] --sek 15 --out out/material/musik
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import audio, proc

FPS = 30


def analyse(song: Path, sek: float, anzahl: int = 3) -> dict:
    y = audio.dekodieren(song)
    dauer = len(y) / audio.SR
    n = int(dauer * FPS)
    m, info = audio.merkmale(y, audio.SR, n, 1.0 / FPS, log=lambda *_: None, bpm=(70.0, 180.0))
    e = cv2.GaussianBlur(m["rms"].reshape(-1, 1), (0, 0), FPS * 0.5).ravel()          # Energie, 1 s geglättet
    on = m["onset"]
    beat = np.asarray(info["beat_frames"])
    takt = [t for t in info["takt_frames"]]
    L = int(sek * FPS)
    kand = []
    for s in takt:
        if s + L > n:
            break
        w = e[s:s + L]
        b = beat[(beat >= s) & (beat < s + L)] - s
        if len(b) < 8:
            continue
        o = on[s:s + L]
        maske = np.ones(L, bool)
        for k in b:
            maske[max(0, k - 2):k + 3] = False
        klarheit = float(o[b].mean() / (o[maske].mean() + 1e-3))
        energie = float(w.mean())
        anstieg = float(w[int(L * 0.6):].mean() - w[:int(L * 0.3)].mean())
        # Einsatz: größter Sprung der Energie (2 s danach gegen 2 s davor) im mittleren Teil 25–75 %
        sp = [(float(w[i:i + 2 * FPS].mean() - w[i - 2 * FPS:i].mean()), i)
              for i in range(int(L * 0.25), int(L * 0.75), 3)]
        einsatz, wo = max(sp)
        score = 0.35 * min(klarheit / 4.0, 1.0) + 0.25 * energie + 0.25 * max(einsatz, 0) * 3 + 0.15 * max(anstieg, 0) * 2
        kand.append({"start": round(s / FPS, 3), "score": round(score, 3), "klarheit": round(klarheit, 2),
                     "energie": round(energie, 3), "einsatz": round(einsatz, 3), "einsatz_bei_s": round(wo / FPS, 2),
                     "anstieg": round(anstieg, 3)})
    kand.sort(key=lambda k: -k["score"])
    wahl = []
    for k in kand:
        if all(abs(k["start"] - w["start"]) >= sek for w in wahl):
            wahl.append(k)
        if len(wahl) == anzahl:
            break
    return {"song": song.name, "dauer_s": round(dauer, 1), "bpm": info["bpm"], "abschnitte": wahl, "_e": e, "_n": n}


def bild(erg: list[dict], ziel: Path, sek: float) -> None:
    h, b = 140, 1400
    tafeln = []
    for r in erg:
        t = np.full((h, b, 3), 20, np.uint8)
        e = r["_e"] / (r["_e"].max() + 1e-9)
        xs = np.linspace(0, b - 1, len(e)).astype(int)
        for k, w in enumerate(r["abschnitte"]):
            x0, x1 = int(w["start"] * FPS / r["_n"] * b), int((w["start"] + sek) * FPS / r["_n"] * b)
            cv2.rectangle(t, (x0, 0), (x1, h), (60, 90, 40), -1)
            cv2.putText(t, f"{k + 1}", (x0 + 4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
        pts = np.stack([xs, (h - 8 - e * (h - 40)).astype(int)], 1)
        cv2.polylines(t, [pts], False, (240, 200, 120), 1)
        cv2.putText(t, f"{r['song']}  {r['bpm']} bpm  {r['dauer_s']} s", (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        tafeln.append(t)
    cv2.imwrite(str(ziel), np.vstack(tafeln))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("songs", nargs="+")
    ap.add_argument("--sek", type=float, default=15)
    ap.add_argument("--out", default="out/material/musik")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    erg = [analyse(Path(s), a.sek) for s in a.songs]
    bild(erg, out / "abschnitte.png", a.sek)
    for r, s in zip(erg, a.songs):
        for k, w in enumerate(r["abschnitte"]):
            proc.run([proc.tool("ffmpeg"), "-y", "-v", "error", "-ss", str(w["start"]), "-t", str(a.sek), "-i", s,
                      "-c:a", "aac", "-b:a", "192k", str(out / f"{Path(s).stem}_{k + 1}_{w['start']:.1f}s.m4a")], "Hörprobe")
    sauber = [{k: v for k, v in r.items() if not k.startswith("_")} for r in erg]
    (out / "abschnitte.json").write_text(json.dumps(sauber, indent=1), encoding="utf-8")
    print(json.dumps(sauber, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
