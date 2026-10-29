"""dmk.beatraster: BPM-Raster eines Tracks prüfen und als exaktes Beat-Gitter nutzen.

Elektronische Musik (Drum-Machine, DAW, TouchDesigner) läuft auf einem festen Tempo. Dann ist ein angepasstes Gitter
t_k = t0 + k·T genauer als jeder einzelne erkannte Schlag: es hat keinen Erkennungsfehler pro Beat, nur den Fehler der
Anpassung über alle Beats (< 1 ms bei 50 Beats). Die Prüfung sagt, ob der Track wirklich ein festes Raster hat
(Streuung der Schläge um das Gitter), und vergleicht mit einer mitgelieferten Analyse (MIDI, CSV/TXT, BPM-Angabe).

    python -m dmk.beatraster <track> [<ordner mit analyse>]
"""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

AUDIO = (".wav", ".mp3", ".m4a", ".flac", ".aif", ".aiff", ".ogg", ".opus")


def raster_fit(t: np.ndarray, periode: float | None = None, iter_: int = 4) -> dict:
    """Gitter t0 + k·T an Beat-Zeiten anpassen (robust: Ausreißer > 40 ms zählen in der letzten Runde nicht)."""
    t = np.sort(np.asarray(t, float))
    if len(t) < 4:
        return {}
    T = float(periode or np.median(np.diff(t)))
    t0 = float(t[0])
    ok = np.ones(len(t), bool)
    for _ in range(iter_):
        k = np.round((t - t0) / T)
        A = np.stack([np.ones(ok.sum()), k[ok]], 1)
        (t0, T), *_ = np.linalg.lstsq(A, t[ok], rcond=None)
        res = t - (t0 + k * T)
        ok = np.abs(res) < max(0.04, 3 * np.std(res[ok]))
    k = np.round((t - t0) / T)
    res = t - (t0 + k * T)
    return {"bpm": round(60.0 / T, 3), "periode": float(T), "t0": float(t0), "rest_std_ms": round(float(np.std(res[ok]) * 1000), 2),
            "rest_max_ms": round(float(np.abs(res[ok]).max() * 1000), 1), "ausreisser": int((~ok).sum()), "n": int(len(t)),
            "residuen_ms": np.round(res * 1000, 1).tolist()}


def gitter(fit: dict, t_von: float, t_bis: float) -> np.ndarray:
    T, t0 = fit["periode"], fit["t0"]
    k0, k1 = int(np.ceil((t_von - t0) / T)), int(np.floor((t_bis - t0) / T))
    return t0 + np.arange(k0, k1 + 1) * T


# ------------------------------------------------------------------ externe Analyse

def midi_noten(path) -> tuple[list[float], float | None]:
    """Minimaler SMF-Leser: Zeiten (s) aller Note-on-Ereignisse und das erste Tempo (bpm)."""
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        return [], None
    _, ntrk, div = struct.unpack(">HHH", data[8:14])
    pos, ereignisse, tempi = 14, [], []
    for _ in range(ntrk):
        if data[pos:pos + 4] != b"MTrk":
            break
        ln = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        p, end, tick, status = pos + 8, pos + 8 + ln, 0, 0
        while p < end:
            d = 0
            while True:
                b = data[p]; p += 1
                d = (d << 7) | (b & 0x7F)
                if b < 0x80:
                    break
            tick += d
            b = data[p]
            if b == 0xFF:
                typ, p = data[p + 1], p + 2
                ln2 = 0
                while True:
                    c = data[p]; p += 1
                    ln2 = (ln2 << 7) | (c & 0x7F)
                    if c < 0x80:
                        break
                if typ == 0x51:
                    tempi.append((tick, int.from_bytes(data[p:p + 3], "big")))
                p += ln2
                continue
            if b in (0xF0, 0xF7):
                p += 1
                ln2 = 0
                while True:
                    c = data[p]; p += 1
                    ln2 = (ln2 << 7) | (c & 0x7F)
                    if c < 0x80:
                        break
                p += ln2
                continue
            if b & 0x80:
                status, p = b, p + 1
            typ = status & 0xF0
            n_daten = 1 if typ in (0xC0, 0xD0) else 2
            if typ == 0x90 and data[p + 1] > 0:
                ereignisse.append(tick)
            p += n_daten
        pos = end
    tempi = sorted(tempi) or [(0, 500000)]

    def sek(t):
        s, last_tick, us = 0.0, 0, tempi[0][1]
        for tt, u in tempi:
            if tt >= t:
                break
            s += (tt - last_tick) * us / div / 1e6
            last_tick, us = tt, u
        return s + (t - last_tick) * us / div / 1e6
    return sorted(sek(t) for t in ereignisse), round(60e6 / tempi[0][1], 3)


def externe_analyse(ordner) -> dict:
    """Sucht im Ordner nach Beat-Daten: MIDI (Note-on), CSV/TXT/JSON (Zeiten in s), BPM-Angaben in Namen und Texten."""
    ordner = Path(ordner)
    out: dict = {"dateien": [], "bpm_angaben": []}
    for f in sorted(ordner.rglob("*")):
        if not f.is_file():
            continue
        out["dateien"].append(str(f.relative_to(ordner)))
        for m in re.finditer(r"(\d{2,3}(?:[.,]\d+)?)\s*-?\s*bpm|bpm\s*[:=]?\s*(\d{2,3}(?:[.,]\d+)?)", f.name, re.I):
            out["bpm_angaben"].append((f.name, float((m.group(1) or m.group(2)).replace(",", "."))))
        suf = f.suffix.lower()
        if suf in (".mid", ".midi"):
            zeiten, bpm = midi_noten(f)
            out.setdefault("midi", []).append({"datei": f.name, "noten": len(zeiten), "bpm": bpm, "zeiten": zeiten[:4000]})
        elif suf in (".csv", ".txt", ".json", ".tsv", ".dat") and f.stat().st_size < 5_000_000:
            txt = f.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"bpm\D{0,4}(\d{2,3}(?:[.,]\d+)?)", txt, re.I):
                out["bpm_angaben"].append((f.name, float(m.group(1).replace(",", "."))))
            zahlen = [float(x) for x in re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", txt)[:20000]]
            if len(zahlen) >= 8:
                z = np.array(zahlen)
                if np.all(np.diff(z) >= 0):   # aufsteigende Zeitreihe
                    out.setdefault("zeitreihen", []).append({"datei": f.name, "werte": len(z), "zeiten": z[:4000].tolist()})
    return out


def analyse(track, ordner=None, log=print) -> dict:
    from . import audio
    y = audio.dekodieren(track, audio.SR)
    dauer = len(y) / audio.SR
    _, info = audio.merkmale(y, audio.SR, int(dauer * 30), 1 / 30, 0.0, log=lambda s: None)
    tb = np.array(info["beat_zeiten"])
    fit = raster_fit(tb)
    r = {"track": str(track), "dauer_s": round(dauer, 2), "tracker_bpm": info["bpm"], "beats": len(tb), "raster": fit}
    log(f"[raster] {Path(track).name}: tracker {info['bpm']} bpm, grid fit {fit.get('bpm')} bpm, beats scatter "
        f"{fit.get('rest_std_ms')} ms (max {fit.get('rest_max_ms')} ms, outliers {fit.get('ausreisser')})")
    if ordner:
        ext = externe_analyse(ordner)
        r["extern"] = {k: v for k, v in ext.items() if k not in ("midi", "zeitreihen")}
        for quelle in ext.get("midi", []) + ext.get("zeitreihen", []):
            z = np.array(quelle["zeiten"])
            if len(z) < 4:
                continue
            # Vergleich: Abstand jedes erkannten Beats zum nächsten externen Ereignis
            d = np.array([z[np.argmin(np.abs(z - t))] - t for t in tb])
            nah = np.abs(d) < 0.08
            e = {"datei": quelle["datei"], "ereignisse": len(z), "bpm_datei": quelle.get("bpm"),
                 "beats_mit_partner": int(nah.sum()),
                 "versatz_median_ms": round(float(np.median(d[nah]) * 1000), 1) if nah.any() else None,
                 "versatz_std_ms": round(float(np.std(d[nah]) * 1000), 1) if nah.any() else None,
                 "raster_datei": {k: v for k, v in raster_fit(z).items() if k != "residuen_ms"}}
            r.setdefault("vergleich", []).append(e)
            log(f"[raster] vs {quelle['datei']}: {e['beats_mit_partner']}/{len(tb)} beats matched, offset "
                f"{e['versatz_median_ms']} ms ± {e['versatz_std_ms']} ms")
    return r


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    r = analyse(a[0], a[1] if len(a) > 1 else None)
    print(json.dumps({k: v for k, v in r.items()}, indent=1, ensure_ascii=False, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
