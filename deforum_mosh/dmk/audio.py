"""Audioreaktive Steuerung (wie Deforums Audio-Schedules / Parseq, aber im Treiber selbst).

Ein Track wird per ffmpeg dekodiert und in Merkmale pro VIDEO-Frame zerlegt (alle 0..1 normiert):
  rms      Lautheit                         huelle   Lautheit, über ~0,5 s geglättet
  bass     20–150 Hz    (Kick, Sub)         tief     150–400 Hz
  mitten   400–2000 Hz                      hoehen   2–8 kHz (Hi-Hats, Rauschen)
  onset    spektraler Fluss (Anschläge)     beat     1 auf Beat-Frames (DP-Beat-Tracker, Ellis 2007)
  takt     1 auf jedem 4. Beat (Zählzeit 1) hell     spektraler Schwerpunkt (Helligkeit des Klangs)
  flach    spektrale Flachheit (Rauschanteil)

Zeitbasis: Frame i der Ausgabe ist Quellframe start + i·stride; dessen Zeit im Track ist
offset + (start + i·stride) / basis_fps  (Lille: Frame 1 = 0 s bei 30 fps). Das Video darf eine andere
Bildrate haben; `ton_anlegen` zieht den Tonausschnitt dann per atempo auf die Videolänge.

Bindung (JSON, Schlüssel "audio" in den Parametern):
  {"datei": "...m4a", "basis_fps": 30, "offset": 0.0,
   "bindungen": {"denoise": {"quelle": "onset", "min": 0.40, "max": 0.68, "release": 4},
                 "quellzoom": {"quelle": "beat", "min": 1.0015, "max": 1.01, "release": 3}, ...},
   "takt_sample": {"quelle": "beat", "schwelle": 0.5}}        # erzwingt ein Sample auf dem Beat
Jede Bindung: quelle, min, max, kurve (Gamma), schwelle, invert, attack/release (Hüllkurve in Frames),
glatt (Gauß-Sigma in Frames), lokal (0..1 über den gerenderten Ausschnitt statt über den ganzen Track).
Ergebnis ist ein Wert pro Frame. "bpm": [lo, hi] grenzt die Temposuche ein (Standard 60–140; auf kurzen Ausschnitten rastet
der Tracker sonst auf Achtel ein, z. B. 161 statt 86 bpm). Für schnelle Musik z. B. [100, 200] setzen.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import proc

SR = 22050
N_FFT = 2048
HOP = 512
# Anzeigenamen (englisch) für die internen Parameter- und Merkmalsschlüssel
ANZEIGE = {"weich": "blur", "gedaechtnis": "memory", "quellzoom": "zoom", "glaettung": "smoothing", "referenz": "reference",
           "control_strength": "depth guidance", "denoise": "transform", "alpha": "source pull", "noise": "grain",
           "cfg": "prompt strength", "schaerfe_halten": "sharpness hold", "licht": "light", "huelle": "envelope",
           "hoehen": "highs", "tief": "low-mids", "mitten": "mids", "takt": "bar (count 1)", "takt2": "count 2",
           "takt3": "count 3", "takt4": "count 4", "takt_phase": "bar phase", "beat_phase": "beat phase",
           "hell": "brightness", "flach": "noisiness"}
BAENDER = {"bass": (20, 150), "tief": (150, 400), "mitten": (400, 2000), "hoehen": (2000, 8000)}
MERKMALE = ("rms", "huelle", "bass", "tief", "mitten", "hoehen", "onset", "beat", "takt", "takt2", "takt3", "takt4",
            "offbeat", "takt_phase", "beat_phase", "hell", "flach")


def dekodieren(path, sr: int = SR, start: float = 0.0, dauer: float | None = None) -> np.ndarray:
    """Mono-Float-Signal per ffmpeg (liest alles, was ffmpeg kennt: m4a/opus, mp3, wav …)."""
    cmd = [proc.tool("ffmpeg"), "-v", "error", "-ss", f"{start:.3f}", "-i", str(path)]
    if dauer:
        cmd += ["-t", f"{dauer:.3f}"]
    cmd += ["-ac", 1, "-ar", sr, "-f", "f32le", "-"]
    r = proc.run(cmd, "Audio dekodieren")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def klickspur(pfad, bpm: float, sekunden: float, sr: int = 44100, seed: int = 7) -> Path:
    """Taktspur für "BPM only" (Nutzer 27.10.: wählen zwischen BPM und eigener Audiodatei): Kick auf der Eins (55 Hz,
    die Analyse erkennt die Eins am meisten Bass), Klick auf 2–4 (2 kHz, liegt im Band "hoehen"), leises Rosa-Rauschen
    darunter, damit Hüllkurven-Bindungen nicht auf null fallen. Erster Schlag bei 0,5 s. Schreibt 16-bit-WAV."""
    import wave
    n = int((sekunden + 1.0) * sr)
    rng = np.random.default_rng(seed)
    w = np.fft.rfft(rng.standard_normal(n))
    w /= np.sqrt(np.maximum(np.arange(len(w)), 1))
    y = np.fft.irfft(w, n)
    y *= 10 ** (-36 / 20) / (np.abs(y).max() + 1e-9)
    t_k = np.arange(int(0.09 * sr)) / sr
    kick = np.sin(2 * np.pi * (55 + 90 * np.exp(-t_k * 40)) * t_k) * np.exp(-t_k * 28) * 10 ** (-3 / 20)
    t_c = np.arange(int(0.015 * sr)) / sr
    klick = np.sin(2 * np.pi * 2000 * t_c) * np.exp(-t_c * 300) * 10 ** (-9 / 20)
    periode, t0, k = 60.0 / float(bpm), 0.5, 0
    while t0 + k * periode < sekunden + 0.9:
        s = int((t0 + k * periode) * sr)
        z = kick if k % 4 == 0 else klick
        e = min(n, s + len(z))
        y[s:e] += z[:e - s]
        k += 1
    pfad = Path(pfad)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(pfad), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return pfad


def _stft_power(y: np.ndarray) -> np.ndarray:
    pad = N_FFT // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    n = 1 + (len(yp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = yp[idx] * np.hanning(N_FFT)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2          # (t, f)


def _normiert(x: np.ndarray, lo_p: float = 2, hi_p: float = 98) -> np.ndarray:
    lo, hi = np.percentile(x, lo_p), np.percentile(x, hi_p)
    return np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(x)


def _beats(onset: np.ndarray, fps_env: float, bpm_min: float = 60, bpm_max: float = 180,
           straff: float = 400.0) -> tuple[np.ndarray, float]:
    """Beat-Frames (Index in der Onset-Hüllkurve) und Tempo. Tempo aus der Autokorrelation mit
    Log-Normal-Prior um 120 bpm; Beats per dynamischer Programmierung (Ellis 2007)."""
    o = onset - onset.mean()
    ac = np.correlate(o, o, "full")[len(o) - 1:]
    lmin, lmax = int(fps_env * 60 / bpm_max), int(fps_env * 60 / bpm_min)
    lags = np.arange(lmin, min(lmax, len(ac) - 1) + 1)
    bpm = 60.0 * fps_env / lags
    prior = np.exp(-0.5 * ((np.log2(bpm / 120.0)) / 1.0) ** 2)
    period = int(lags[np.argmax(ac[lags] * prior)])
    tempo = 60.0 * fps_env / period

    n = len(onset)
    score = onset.copy()
    back = np.full(n, -1, dtype=int)
    lo, hi = int(round(period * 0.5)), int(round(period * 2.0))
    cands = np.arange(lo, hi + 1)
    strafe = -straff * (np.log(cands / period)) ** 2
    for i in range(lo, n):
        js = i - cands
        ok = js >= 0
        if not ok.any():
            continue
        vals = score[js[ok]] + strafe[ok]
        k = int(np.argmax(vals))
        if vals[k] > 0:
            score[i] = onset[i] + vals[k]
            back[i] = js[ok][k]
    # Ende: bestes Score im letzten Periodenfenster, dann zurücklaufen
    tail = max(1, n - period)
    i = int(tail + np.argmax(score[tail:]))
    beats = [i]
    while back[i] >= 0:
        i = back[i]
        beats.append(i)
    return np.array(beats[::-1], dtype=int), float(tempo)


def fein_anschlag(y: np.ndarray, sr: int, t_grob: np.ndarray, fenster: float = 0.04) -> np.ndarray:
    """Beat-Zeiten auf den Anschlag verfeinern: spektraler Fluss mit kurzem Fenster (512, Hop 64 ≈ 3 ms) im Bereich
    ±fenster um den Tracker-Beat. Der Tracker rechnet im 23-ms-Raster; gemessen 21.10. lag der Anschlag 0–23 ms nach
    dem Tracker-Beat (Median 6 ms). Ein größeres Fenster greift auf Nachbarschläge über (±60 ms: Ausreißer bis 58 ms)."""
    nfft, hop = 512, 64
    yp = np.pad(y, (nfft // 2, nfft // 2), mode="reflect")
    n = 1 + (len(yp) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    L = np.log1p(np.abs(np.fft.rfft(yp[idx] * np.hanning(nfft)[None, :], axis=1)) ** 2 * 1e3)
    f = np.concatenate([[0.0], np.maximum(L[1:] - L[:-1], 0).sum(1)])
    t = np.arange(n) * hop / sr
    out = np.asarray(t_grob, dtype=np.float64).copy()
    for k, g in enumerate(out):
        s = (t >= g - fenster) & (t <= g + fenster)
        if s.any():
            out[k] = t[s][int(np.argmax(f[s]))]
    return out


def merkmale(y: np.ndarray, sr: int, n_frames: int, sek_pro_frame: float, offset: float = 0.0,
             log=print, bpm=(60.0, 140.0), vorlauf: float = 0.25, raster=None) -> tuple[dict[str, np.ndarray], dict]:
    """Merkmale pro Video-Frame (0..1). Kontinuierliche Größen werden über das Frame-Fenster gemittelt,
    Anschläge (onset) per Maximum, damit ein kurzer Schlag nicht im Mittel verschwindet."""
    P = _stft_power(y)                                  # (t, f)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    fps_env = sr / HOP
    te = np.arange(P.shape[0]) * HOP / sr               # Zeit jeder Analyse-Spalte

    env: dict[str, np.ndarray] = {}
    tot = P.sum(1) + 1e-10
    env["rms"] = np.sqrt(P.mean(1))
    for name, (f0, f1) in BAENDER.items():
        sel = (freqs >= f0) & (freqs < f1)
        env[name] = np.log1p(P[:, sel].sum(1) * 1e3)
    env["hell"] = (P * freqs[None, :]).sum(1) / tot
    logP = np.log(P + 1e-10)
    env["flach"] = np.exp(logP.mean(1)) / (P.mean(1) + 1e-10)
    # spektraler Fluss auf log-komprimiertem Spektrum (halbwellen-gleichgerichtet)
    L = np.log1p(P * 1e3)
    flux = np.maximum(L[1:] - L[:-1], 0).sum(1)
    onset = np.concatenate([[0.0], flux])
    onset = onset - np.convolve(onset, np.ones(int(fps_env * 0.5)) / int(fps_env * 0.5), "same") * 0.5
    onset = np.maximum(onset, 0)
    env["onset"] = onset
    beats_e, bpm = _beats(onset / (onset.max() + 1e-10), fps_env, float(bpm[0]), float(bpm[1]))

    # Beat-Phase mit dem meisten Bass = Zählzeit 1
    bass_e = env["bass"]
    takt_phase = int(np.argmax([bass_e[beats_e[p::4]].mean() if len(beats_e[p::4]) else 0 for p in range(4)]))
    takt_e = beats_e[takt_phase::4]

    # ---- auf Video-Frames abbilden
    t_frame = offset + np.arange(n_frames + 1) * sek_pro_frame
    idx = np.searchsorted(t_frame, te, side="right") - 1          # Frame jeder Analyse-Spalte, -1/n = außerhalb
    innen = (idx >= 0) & (idx < n_frames)
    fehlt = n_frames - len(np.unique(idx[innen]))
    if fehlt > 0:
        log(f"[Audio] {fehlt} video frames lie beyond the end of the track ({te[-1]:.1f} s) → 0")
    cnt = np.bincount(idx[innen], minlength=n_frames).astype(np.float64)
    out: dict[str, np.ndarray] = {}
    for name, e in env.items():
        # Normierung über den GANZEN dekodierten Track, nicht nur über den gerenderten Ausschnitt: so
        # bedeutet 0..1 bei einem 40-Frame-Test dasselbe wie bei 500 Frames
        e = _normiert(e, 2, 99.5 if name == "onset" else 98)
        if name == "onset":
            m = np.zeros(n_frames)
            np.maximum.at(m, idx[innen], e[innen])
        else:
            m = np.bincount(idx[innen], weights=e[innen], minlength=n_frames) / np.maximum(cnt, 1)
        out[name] = np.clip(m, 0, 1)
    k = max(1, int(round(0.5 / sek_pro_frame)))
    out["huelle"] = np.clip(cv2.GaussianBlur(out["rms"].reshape(-1, 1), (0, 0), k / 2.0).ravel(), 0, 1)

    # Beat-Zeiten fein auf den Anschlag; ein Ereignis gehört zum ersten Frame, der höchstens `vorlauf` Frames vor dem
    # Anschlag beginnt. Vorher: Tracker-Raster (23 ms) und Abrunden auf den Frame-Anfang – der Schnitt kam im Mittel
    # 20–40 ms, bis 80 ms vor dem Ton (gemessen 21.10. mit dmk.sync_echt; Nutzer: "rennt dem Sound davon")
    tb = fein_anschlag(y, sr, te[beats_e])
    # BPM-Raster (dmk.beatraster): bei festem Tempo ein angepasstes Gitter statt der einzelnen Schläge – kein Fehler
    # pro Beat. "auto": nur wenn die Schläge eng am Gitter liegen (Streuung < 12 ms); "an": immer;
    # {"zeiten": [...]}: Beat-Zeiten von außen (MIDI/Analyse), {"bpm": x, "t0": s}: festes Gitter
    raster_info = None
    if raster:
        from . import beatraster
        if isinstance(raster, dict) and raster.get("zeiten"):
            tb, raster_info = np.sort(np.asarray(raster["zeiten"], float)), {"quelle": "extern"}
        else:
            # Gitter lokal um den gerenderten Ausschnitt anpassen (±8 s): ein ganzer Song hält das Tempo nicht immer
            fen = (tb >= offset - 8.0) & (tb <= offset + n_frames * sek_pro_frame + 8.0)
            fit = ({"periode": 60.0 / float(raster["bpm"]), "t0": float(raster.get("t0", tb[0])), "bpm": float(raster["bpm"]),
                    "rest_std_ms": 0.0} if isinstance(raster, dict) and raster.get("bpm")
                   else beatraster.raster_fit(tb[fen] if fen.sum() >= 8 else tb))
            if fit and (raster != "auto" or fit["rest_std_ms"] < 12.0):
                tb = beatraster.gitter(fit, max(0.0, tb[0] - 4 * fit["periode"]), te[-1])
                raster_info = {k: v for k, v in fit.items() if k != "residuen_ms"}
        if raster_info is not None:   # Zählzeit 1 neu bestimmen: Beat-Phase mit dem meisten Bass
            be = np.clip(np.round(tb * sr / HOP).astype(int), 0, len(bass_e) - 1)
            takt_phase = int(np.argmax([bass_e[be[q::4]].mean() if len(be[q::4]) else 0 for q in range(4)]))
            log(f"[Audio] beat grid: {raster_info}")

    def impulse(zeiten):
        s = np.zeros(n_frames)
        f = np.ceil((np.asarray(zeiten) - offset) / sek_pro_frame - vorlauf).astype(int)
        f = np.unique(f[(f >= 0) & (f < n_frames)])
        s[f] = 1.0
        return s, f
    out["beat"], beat_frames = impulse(tb)
    out["takt"], takt_frames = impulse(tb[takt_phase::4])
    # 4/4-Muster: Zählzeit 1..4 einzeln (takt1 = takt), Offbeat (Achtel zwischen den Beats) und Phasen 0..1
    # innerhalb von Takt und Beat (Sägezahn) für Rampen über den Takt
    for k in range(4):
        out[f"takt{k + 1}"], _ = impulse(tb[(takt_phase + k) % 4::4])
    off = ((tb[:-1] + tb[1:]) / 2) if len(tb) > 1 else np.array([])
    out["offbeat"], _ = impulse(off)
    t_mitte = offset + (np.arange(n_frames) + 0.5) * sek_pro_frame

    def saegezahn(tm):
        if len(tm) < 2:
            return np.zeros(n_frames)
        j = np.clip(np.searchsorted(tm, t_mitte, side="right") - 1, 0, len(tm) - 2)
        return np.clip((t_mitte - tm[j]) / (tm[j + 1] - tm[j]), 0, 1)
    out["takt_phase"] = saegezahn(tb[takt_phase::4])
    out["beat_phase"] = saegezahn(tb)
    info = {"bpm": round(bpm, 1), "beats": int(len(beats_e)), "beat_frames": beat_frames.tolist(),
            "beat_zeiten": [round(float(x), 4) for x in tb], "takt_phase_index": takt_phase, "raster": raster_info,
            "takt_frames": takt_frames.tolist(), "sek_pro_frame": sek_pro_frame, "offset": offset,
            "track_sek": round(float(te[-1]), 2)}
    log(f"[Audio] {bpm:.1f} bpm, {len(beats_e)} beats, {len(beat_frames)} of them in the clip; "
        f"frame window {sek_pro_frame * 1000:.1f} ms")
    return out, info


def _huelle(x: np.ndarray, attack: float, release: float) -> np.ndarray:
    """Hüllkurvenfolger (einpolig): attack/release als Zeitkonstante in Frames; 0 = sofort."""
    if attack <= 0 and release <= 0:
        return x
    ka = 1.0 - np.exp(-1.0 / attack) if attack > 0 else 1.0
    kr = 1.0 - np.exp(-1.0 / release) if release > 0 else 1.0
    e = np.empty_like(x)
    v = x[0]
    for i, s in enumerate(x):
        v += (s - v) * (ka if s > v else kr)
        e[i] = v
    return e


def bindung(sig: np.ndarray, min: float, max: float, kurve: float = 1.0, schwelle: float = 0.0,
            invert: bool = False, attack: float = 0.0, release: float = 0.0, glatt: float = 0.0,
            lokal: bool = False, **_) -> np.ndarray:
    """lokal=True: Signal über den gerenderten Ausschnitt neu auf 0..1 spreizen (sonst gilt die Normierung
    über den ganzen Track, und ein leiser Abschnitt erreicht `max` nie)."""
    s = np.asarray(sig, dtype=np.float64)
    if lokal:
        s = _normiert(s)
    if schwelle > 0:
        s = np.clip((s - schwelle) / (1.0 - schwelle), 0, 1)
    s = _huelle(s, attack, release)
    if glatt > 0:
        s = cv2.GaussianBlur(s.reshape(-1, 1), (0, 0), glatt).ravel()
    s = np.clip(s, 0, 1) ** kurve
    if invert:
        s = 1.0 - s
    return min + (max - min) * s


def aufloesen(spec: dict, n: int, stride: int = 1, start: int = 0, out_dir=None, log=print):
    """Spezifikation → (Parameterkurven pro Frame, erzwungene Sample-Frames | None, Merkmale, Info)."""
    basis_fps = float(spec.get("basis_fps", 30))
    spf = stride / basis_fps
    offset = float(spec.get("offset", 0.0)) + start / basis_fps
    datei = Path(spec["datei"])
    if not datei.exists() and (Path(__file__).resolve().parents[2] / datei).exists():
        datei = Path(__file__).resolve().parents[2] / datei      # relative Pfade in Rezepten gelten ab dem Repo
    y = dekodieren(datei, SR)               # ganzer Track: Normierung und Beat-Raster gelten trackweit
    # bild_versatz_ms > 0: Bild reagiert so viel später (Ausgleich z. B. für Bluetooth-Kopfhörer); nur die Analyse
    # verschiebt sich, die Tonspur bleibt, wo sie ist
    versatz = float(spec.get("bild_versatz_ms", 0.0)) / 1000.0
    mm, info = merkmale(y, SR, n, spf, offset - versatz, log, bpm=tuple(spec.get("bpm", (60, 140))),
                        raster=spec.get("raster"))
    kurven: dict[str, np.ndarray] = {}
    # Weltwechsel auf dem Takt: {"quelle": "takt", "alle": 1, "welten": [prompt, prompt, ...]} – auf jedem `alle`-ten
    # Ereignis der Quelle springt der Prompt zur nächsten Welt. Merkmal "wechsel" (1 auf dem Wechsel-Frame) steht den
    # Bindungen zur Verfügung, z. B. um im Wechsel stärker neu zu zeichnen und das Gedächtnis loszulassen.
    ww = spec.get("welt_wechsel")
    mm["wechsel"] = np.zeros(n)
    if ww and ww.get("welten"):
        ev = np.flatnonzero(mm[ww.get("quelle", "takt")] >= 0.5)
        alle = max(1, int(ww.get("alle", 1)))
        sprung = ev[::alle] if ww.get("erster_sofort") else ev[alle - 1::alle]
        sprung = sprung[sprung > 0]          # Eins auf Frame 0: die Startwelt bleibt stehen (27.10.: Clip begann mit Welt 2)
        welt = np.zeros(n, int)
        for f in sprung:
            welt[f:] += 1
        mm["wechsel"][sprung] = 1.0
        kurven["_prompt"] = [ww["welten"][k % len(ww["welten"])] for k in welt]
        info["welt_wechsel"] = sprung.tolist()
        log(f"[Audio] world change on frames {sprung.tolist()} ({len(ww['welten'])} worlds)")

    def eine(b):
        q = b.get("quelle", "rms")
        if q not in mm:
            raise ValueError(f"Unknown audio feature {q!r}; available: {', '.join(MERKMALE)}")
        return bindung(mm[q], **{k: v for k, v in b.items() if k != "quelle"})
    for param, b in (spec.get("bindungen") or {}).items():
        if isinstance(b, list):
            # mehrere Bindungen auf einen Parameter: die erste setzt den Ruhewert (ihr min), jede weitere addiert
            # ihren Wert abzüglich dieses Ruhewerts; begrenzt auf den Bereich aller min/max. Zwei Pulse auf demselben
            # Frame verstärken sich; eine invertierte Bindung (min > max) hebt den Ruhepegel an, wo sie "an" ist.
            ks = [eine(x) for x in b]
            ruhe = float(b[0]["min"])
            k = ks[0] + sum(kx - ruhe for kx in ks[1:])
            lo = min(min(float(x["min"]), float(x["max"])) for x in b)
            hi = max(max(float(x["min"]), float(x["max"])) for x in b)
            kurven[param] = np.clip(k, lo, hi)
        else:
            kurven[param] = eine(b)
    # Prompt-Zusätze pro Zählzeit: {"takt1": ", monumental", "takt2": ", dissolving", ...} mit "release" Frames
    pz = spec.get("prompt_zusatz")
    if pz:
        rel = int(pz.get("release", 4))
        zus = [""] * n
        for q, text in pz.items():
            if q == "release" or q not in mm:
                continue
            akt = _huelle(mm[q], 0, rel) > 0.2
            for i in np.flatnonzero(akt):
                zus[i] += text
        kurven["_prompt_zusatz"] = zus
    # Seed-Sprung auf einem Signal ("seed": {"quelle": "takt"}): innerhalb des Takts derselbe Seed (kohärente
    # Zeichnung), auf der Eins ein neuer – statt Seed pro Frame
    sd = spec.get("seed")
    if sd:
        kurven["_seed"] = np.cumsum(mm[sd.get("quelle", "takt")] >= float(sd.get("schwelle", 0.5))).astype(int)
    force = None
    ts = spec.get("takt_sample")
    if ts:
        sig = mm[ts.get("quelle", "beat")]
        force = sig >= float(ts.get("schwelle", 0.5))
        info["takt_sample"] = int(force.sum())
    def _q(b):
        return " + ".join(x.get("quelle", "rms") for x in b) if isinstance(b, list) else b.get("quelle", "rms")
    info["bindungen"] = {p: {"quelle": _q(b), "min": round(float(kurven[p].min()), 4), "max": round(float(kurven[p].max()), 4),
                             "mittel": round(float(kurven[p].mean()), 4)} for p, b in (spec.get("bindungen") or {}).items()}
    for p, b in (spec.get("bindungen") or {}).items():
        log(f"[Audio] {p} <- {_q(b)}: {kurven[p].min():.4g}..{kurven[p].max():.4g}, mean {kurven[p].mean():.4f}")
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "audio_kurven.npz", **{f"m_{k}": v for k, v in mm.items()},
                            **{f"p_{k}": v for k, v in kurven.items() if not k.startswith("_")})
        (out_dir / "audio.json").write_text(json.dumps(dict(info, spec=spec), indent=2, ensure_ascii=False),
                                            encoding="utf-8")
        zeichnen({**{k: mm[k] for k in MERKMALE if k in mm}}, out_dir / "audio_merkmale.png",
                 marks=info["beat_frames"], takte=info["takt_frames"])
        num = {p: v for p, v in kurven.items() if not p.startswith("_")}
        if num:
            zeichnen({p: (num[p] - num[p].min()) / (np.ptp(num[p]) or 1) for p in num},
                     out_dir / "audio_kurven.png", marks=info["beat_frames"], takte=info["takt_frames"],
                     labels={p: f"{ANZEIGE.get(p, p)} {num[p].min():.3g}..{num[p].max():.3g}" for p in num})
    return kurven, force, mm, info


def zeichnen(kurven: dict[str, np.ndarray], path, marks=None, takte=None, labels=None, breite: int = 1400,
             zeile: int = 70) -> None:
    """Kurvenblatt: eine Zeile pro Kurve (0..1), Beat-Striche grau, Takt-Striche weiß."""
    n = len(next(iter(kurven.values())))
    h = zeile * len(kurven) + 20
    img = np.full((h, breite, 3), 24, np.uint8)
    xs = (np.arange(n) * (breite - 1) / max(1, n - 1)).astype(int)
    for f in (marks or []):
        cv2.line(img, (xs[f], 0), (xs[f], h), (70, 70, 70), 1)
    for f in (takte or []):
        cv2.line(img, (xs[f], 0), (xs[f], h), (140, 140, 140), 1)
    for r, (name, k) in enumerate(kurven.items()):
        y0, y1 = r * zeile + 15, (r + 1) * zeile + 5
        k = np.clip(np.asarray(k, dtype=np.float64), 0, 1)
        pts = np.stack([xs, (y1 - k * (y1 - y0)).astype(int)], 1)
        cv2.fillPoly(img, [np.concatenate([[[xs[0], y1]], pts, [[xs[-1], y1]]]).astype(np.int32)], (90, 140, 200))
        cv2.polylines(img, [pts.astype(np.int32)], False, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, (labels or {}).get(name, name), (6, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    for f in range(0, n, max(1, n // 10)):
        cv2.putText(img, str(f), (xs[f], h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    Path(path).write_bytes(buf.tobytes())


def ton_anlegen(video, audio, out, start_sek: float, quell_sek: float, video_sek: float) -> None:
    """Tonausschnitt [start, start+quell_sek] des Tracks per atempo auf die Videolänge ziehen und muxen."""
    tempo = quell_sek / video_sek
    kette = []
    t = tempo                        # atempo erlaubt 0.5..100; kleinere Faktoren verketten
    while t < 0.5:
        kette.append("atempo=0.5")
        t /= 0.5
    kette.append(f"atempo={t:.6f}")
    proc.run([proc.tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
              "-i", str(video), "-ss", f"{start_sek:.3f}", "-t", f"{quell_sek:.3f}", "-i", str(audio),
              # apad + -t statt -shortest: der Ton wird bis zur Videolänge aufgefüllt, -shortest hatte die letzten
              # 1-2 Videoframes abgeschnitten, weil AAC die Tonspur um einige ms kürzt
              "-filter:a", ",".join(kette + ["apad"]), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
              "-b:a", "192k", "-t", f"{video_sek:.3f}", "-movflags", "+faststart", str(out)], "Ton anlegen")


def kopplung(frames_dir, kurven_npz, quelle: str = "onset", max_lag: int = 6) -> dict:
    """Wie stark folgt das Bild dem Ton? Korrelation zwischen Bildänderung (1 − SSIM zum Vorframe)
    und dem Audiosignal, bestes Lag in Frames (positiv = Bild hinkt nach)."""
    from . import core
    fs = core.list_images(frames_dir)
    d = np.load(kurven_npz)
    sig = d[f"m_{quelle}"]
    prev = None
    ch = []
    for p in fs:
        g = cv2.resize(core.imread(p, gray=True), (320, 180), interpolation=cv2.INTER_AREA)
        ch.append(0.0 if prev is None else 1.0 - core.ssim(prev, g))
        prev = g
    ch = np.array(ch)
    m = min(len(ch), len(sig))
    ch, sig = ch[:m], sig[:m]
    best = (0.0, 0)
    for lag in range(0, max_lag + 1):
        a, b = ch[lag:], sig[:m - lag]
        if a.std() > 0 and b.std() > 0:
            r = float(np.corrcoef(a, b)[0, 1])
            if r > best[0]:
                best = (r, lag)
    return {"r": round(best[0], 3), "lag": best[1], "aenderung_mittel": round(float(ch.mean()), 4),
            "aenderung_beat": round(float(ch[sig > 0.5].mean()), 4) if (sig > 0.5).any() else None,
            "aenderung_ruhe": round(float(ch[sig < 0.2].mean()), 4) if (sig < 0.2).any() else None}
