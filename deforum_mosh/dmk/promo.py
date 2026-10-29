"""Promo-Clips für Stable React (Nutzer 29.10.: Name "Stable React" (wie Stable Diffusion + audio-reaktiv); "erste Test-Promo-Clips mit passender professioneller Motion-Grafik").
Beat-synchrone Motion-Grafik über einem fertigen Clip: Logo-Enthüllung durch einen Säbelhieb, auf jedem Weltwechsel ein
Hieb quer durchs Bild und ein hartes Wort, Takt-Anzeige [1][2][3][4], BPM, Welt-Name, Engine-Label, Endkarte mit
Wettbewerb und Hashtags. Harte Mono-Schrift (Consolas Bold), weiß, kein Schnörkel. Ton: durchgehendes Bett aus dem
Original-Song ab dem Startpunkt des Clips (läuft unter der Endkarte weiter).

    python -m dmk.promo <name> [...]      (Konfigurationen unten in PROMOS)
"""
from __future__ import annotations

import bisect
import json
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import proc

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "out" / "final"
MUSIK = ROOT / "out" / "material" / "musik"
FONT_B = "C:/Windows/Fonts/consolab.ttf"
FONT = "C:/Windows/Fonts/consola.ttf"
W, H, FPS = 1920, 1080, 30
NAME = "STABLE REACT"

PROMOS = {
    "promo_flux_allee": dict(job="d4_baumallee_flux", song="Coffin_Bass.m4a", start=102.2, clip_s=15.0, ende_s=4.0,
                             action_ab=14.0,   # Nutzer 29.10.: "bis Sekunde 14 auf die Wiese", dann Supercut
                             engine="FLUX.1 KONTEXT HYBRID", welten=["SUMMER", "SNOW", "AUTUMN", "BLOSSOM"],
                             woerter=["THE BAR", "PLAYS", "THE WORLD", "NO CUT"]),
    "promo_sd_bergsee": dict(job="s2_bergsee", song="Signal_Fracture.m4a", start=2.833, clip_s=15.0, ende_s=4.0,
                             engine="SD 1.5 · 8 GB VRAM · OPEN WEIGHTS", welten=["LAKE", "GLACIER", "CRATER", "LARCHES"],
                             woerter=["OPEN WEIGHTS", "8 GB", "ONE CLIP", "NO CUT"]),
    "promo_flux_salzseen": dict(job="n2_salzseen_flux", song="Coffin_Bass.m4a", start=35.233, clip_s=15.0, ende_s=4.0,
                                engine="FLUX.1 KONTEXT HYBRID", welten=["PINK", "SNOW", "TURQUOISE", "GOLD"],
                                woerter=["FLUX DRAWS", "SD FLOWS", "EVERY BEAT", "NO CUT"]),
    "promo_brutal_beton": dict(job="b2_betonstruktur_flux", song="Coffin_Bass.m4a", start=35.233, clip_s=15.0, ende_s=4.0,
                               engine="FLUX.1 KONTEXT HYBRID", welten=["CONCRETE", "MOSS", "ICE", "LAVA"],
                               woerter=["OVERGROWN", "FROZEN", "MOLTEN", "RAW"]),
}


def _font(p, s):
    return ImageFont.truetype(p, s)


def _text(d: ImageDraw.ImageDraw, xy, t, f, fill=(255, 255, 255, 255), spacing=0):
    if not spacing:
        d.text(xy, t, font=f, fill=fill)
        return d.textlength(t, font=f)
    x = xy[0]
    for ch in t:
        d.text((x, xy[1]), ch, font=f, fill=fill)
        x += d.textlength(ch, font=f) + spacing
    return x - xy[0]


def _breite(t, f, spacing=0):
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    return sum(dummy.textlength(ch, font=f) + spacing for ch in t) - spacing if spacing else dummy.textlength(t, font=f)


def _hieb(img: Image.Image, t: float, winkel: float = -12.0):
    """Säbelhieb: eine helle, scharfe Linie, die in ~5 Frames quer durchs Bild fährt und kurz nachglüht (t 0..1)."""
    if not 0 <= t <= 1:
        return
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    k = math.tan(math.radians(winkel))
    x_kopf = -200 + t * (W + 400)
    x_schwanz = max(-200, x_kopf - 900 * (1 - t * 0.6))
    for breite, alpha in ((18, 40), (8, 110), (3, 255)):
        d.line([(x_schwanz, H * 0.55 + k * (x_schwanz - W / 2)), (x_kopf, H * 0.55 + k * (x_kopf - W / 2))],
               fill=(255, 255, 255, int(alpha * (1 - max(0, t - 0.7) / 0.3))), width=breite)
    lay = lay.filter(ImageFilter.GaussianBlur(1.2))
    img.alpha_composite(lay)


def overlay_frames(cfg: dict, out: Path) -> int:
    job = F / cfg["job"]
    info = json.loads((job / "audio.json").read_text(encoding="utf-8"))
    beats, takte = sorted(info["beat_frames"]), sorted(info["takt_frames"])
    wechsel = sorted(info.get("welt_wechsel", []))
    bpm = info.get("bpm", 0)
    n_clip = int(cfg["clip_s"] * FPS)
    n = n_clip + int(cfg["ende_s"] * FPS)
    f_logo, f_klein, f_mitte, f_wort, f_zahl = (_font(FONT_B, 120), _font(FONT, 26), _font(FONT_B, 30),
                                                _font(FONT_B, 150), _font(FONT_B, 34))

    def zaehlzeit(i):
        vor = [b for b in beats if b <= i]
        if not vor:
            return 0, 99
        b = vor[-1]
        t1 = [t for t in takte if t <= b]
        k = (beats.index(b) - beats.index(t1[-1])) % 4 + 1 if t1 and t1[-1] in beats else ((len(vor) - 1) % 4) + 1
        return k, i - b

    def welt(i):
        return sum(1 for w in wechsel if w <= i) % len(cfg["welten"])

    for i in range(n):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if i < n_clip:
            # Rahmen-Ecken
            # Raster (Nutzer 29.10.: "Spacing unten ist off"): Eckwinkel 48 px vom Rand, Inhalt 96 px, untere Textzeile
            # endet 96 px über dem Rand (frei von den Winkeln), Takt-Kästchen 20 px darüber, rechts auf gleicher Höhe
            _ecken(d)
            zeile_u = H - M - 26                      # obere Kante der unteren Textzeile (Schrift 26 px)
            kasten, luecke = 48, 12
            k_y = zeile_u - 20 - kasten               # obere Kante der Takt-Kästchen
            # Kopfzeile: Name + Engine, gemeinsame Oberkante
            _text(d, (M, M - 10), NAME, f_mitte, spacing=6)
            ew = _breite(cfg["engine"], f_klein, 2)
            _text(d, (W - M - ew, M - 6), cfg["engine"], f_klein, fill=(255, 255, 255, 220), spacing=2)
            # Takt-Anzeige [1][2][3][4] unten links
            k, seit = zaehlzeit(i)
            for j in range(4):
                x0 = M + j * (kasten + luecke)
                an = (j + 1 == k)
                blitz = max(0.0, 1 - seit / 6) if an else 0
                z = str(j + 1)
                bb = d.textbbox((0, 0), z, font=f_zahl)
                zx = x0 + (kasten - (bb[2] - bb[0])) / 2 - bb[0]
                zy = k_y + (kasten - (bb[3] - bb[1])) / 2 - bb[1]
                if an:
                    d.rectangle([x0, k_y, x0 + kasten, k_y + kasten], fill=(255, 255, 255, int(120 + 135 * blitz)))
                    d.text((zx, zy), z, font=f_zahl, fill=(0, 0, 0, 255))
                else:
                    d.rectangle([x0, k_y, x0 + kasten, k_y + kasten], outline=(255, 255, 255, 160), width=2)
                    d.text((zx, zy), z, font=f_zahl, fill=(255, 255, 255, 160))
            _text(d, (M, zeile_u), f"{bpm:.0f} BPM  ·  4/4  ·  COUNT 1 = NEW WORLD", f_klein,
                  fill=(255, 255, 255, 200), spacing=1)
            # Welt unten rechts, mittig zur Kästchenreihe; Timecode auf der unteren Textzeile
            wn = f"WORLD {welt(i) + 1:02d}  /  {cfg['welten'][welt(i)]}"
            ww = _breite(wn, f_mitte, 3)
            bb = d.textbbox((0, 0), wn, font=f_mitte)
            _text(d, (W - M - ww, k_y + (kasten - (bb[3] - bb[1])) / 2 - bb[1]), wn, f_mitte, spacing=3)
            tc = f"{i // FPS:02d}:{i % FPS:02d}"
            _text(d, (W - M - _breite(tc, f_klein), zeile_u), tc, f_klein, fill=(255, 255, 255, 160))
            # Intro: Logo groß, durch einen Hieb enthüllt (erste ~1,3 s)
            if i < 40:
                a = 255 if i >= 6 else int(255 * i / 6)
                lw = _breite(NAME, f_logo, 18)
                _text(d, ((W - lw) / 2, H / 2 - 90), NAME, f_logo, fill=(255, 255, 255, a if i < 32 else
                                                                          int(255 * (40 - i) / 8)), spacing=18)
                _hieb(img, (i - 2) / 6)
            # Weltwechsel: Hieb + Wort
            for wi, w in enumerate(wechsel):
                if w <= i < w + 24:
                    _hieb(img, (i - w) / 5, winkel=-12 if wi % 2 else 10)
                    wort = cfg["woerter"][wi % len(cfg["woerter"])]
                    s = 1.0 + max(0, 4 - (i - w)) * 0.03
                    fw = _font(FONT_B, int(150 * s))
                    bw = _breite(wort, fw, 10)
                    a = 255 if i - w < 18 else int(255 * (24 - (i - w)) / 6)
                    d2 = ImageDraw.Draw(img)
                    _text(d2, ((W - bw) / 2, H / 2 - 95 * s), wort, fw, fill=(255, 255, 255, a), spacing=10)
        else:
            # Endkarte
            j = i - n_clip
            img = Image.new("RGBA", (W, H), (13, 13, 13, 255))
            d = ImageDraw.Draw(img)
            lw = _breite(NAME, f_logo, 18)
            _text(d, ((W - lw) / 2, 330), NAME, f_logo, spacing=18)
            _hieb(img, j / 6, winkel=-6)
            zeilen = ["AUDIO-REACTIVE GENERATIVE VIDEO · COMFYUI", cfg["engine"],
                      "COMFYUI OPEN SOURCE HIGGSFIELD", "#OpenSourceHiggsfield  #ComfyUI"]
            for z, t in enumerate(zeilen):
                if j >= 4 + z * 4:
                    fz = f_mitte if z != 2 else f_mitte
                    bw = _breite(t, fz, 3)
                    _text(d, ((W - bw) / 2, 540 + z * 62), t, fz, fill=(255, 255, 255, 255 if z != 1 else 190),
                          spacing=3)
        img.save(out / f"{i:05d}.png")
    return n


# Abschluss (Nutzer 29.10.: Schrift "klar lesbar", Supercut "im Takt", Infos "dazwischen eingeblendet"): ab einer Eins
# zwei Takte Supercut, Schnitte nur auf Beat-Frames – Bild auf 1 und 3, Aussage auf 2 und 4 über dem weiterlaufenden
# Bild (nie angeschnitten, ruhig stehend); Knall auf der nächsten Eins, dann baut sich die Endkarte auf den Schlägen auf.
AUSSAGEN = {
    "promo_flux_allee": ["THE BAR PLAYS|THE WORLD", "ONE CLIP|NO CUT", "FLUX KONTEXT|+ SD 1.5", "RUNS ON|8 GB VRAM"],
    "promo_sd_bergsee": ["OPEN WEIGHTS", "SD 1.5|8 GB VRAM", "COUNT 1 =|NEW WORLD", "ONE CLIP|NO CUT"],
    "promo_flux_salzseen": ["FLUX DRAWS", "SD FLOWS", "ON EVERY|BEAT", "ONE CLIP|NO CUT"],
    "promo_brutal_beton": ["SAME SPACE", "NEW MATTER", "ON THE|BEAT", "ONE CLIP|NO CUT"],
}
CLAIM = "THE MUSIC IS A GRAMMAR"
GRUND = (13, 13, 13, 255)
C, L, M = 48, 36, 96                                        # Raster wie im HUD: Eckwinkel, Winkellänge, Inhaltsrand


def _raster(info: dict, n: int) -> tuple[list[int], list[int], float]:
    """Beats und Einsen über das Clip-Ende hinaus fortsetzen (die Musik läuft unter Aussagen und Endkarte weiter)."""
    beats, takte = sorted(info["beat_frames"]), sorted(info["takt_frames"])
    p = float(np.median(np.diff(beats))) if len(beats) > 2 else 15.0
    while beats[-1] < n:
        beats.append(int(round(beats[-1] + p)))
    while takte and takte[-1] < n:
        takte.append(int(round(takte[-1] + 4 * p)))
    return beats, takte, p


def _ecken(d: ImageDraw.ImageDraw, a: int = 200):
    for (x, y, dx, dy) in ((C, C, 1, 1), (W - C, C, -1, 1), (C, H - C, 1, -1), (W - C, H - C, -1, -1)):
        d.line([(x, y), (x + dx * L, y)], fill=(255, 255, 255, a), width=3)
        d.line([(x, y), (x, y + dy * L)], fill=(255, 255, 255, a), width=3)


def _zeilen(d: ImageDraw.ImageDraw, zeilen: list, y0: float):
    """Zeilen zentriert stapeln, gemessen an der Versalhöhe (alles Versalien): gleiche Abstände, egal welche Glyphen.
    zeilen: (text, font, sperrung, alpha, abstand_danach)."""
    y = y0
    for t, f, sp, a, nach in zeilen:
        bb = d.textbbox((0, 0), "H", font=f)
        if t:
            _text(d, ((W - _breite(t, f, sp)) / 2, y - bb[1]), t, f, fill=(255, 255, 255, a), spacing=sp)
        y += bb[3] - bb[1] + nach


def _hoehe(zeilen: list) -> float:
    d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    return sum(d.textbbox((0, 0), "H", font=f)[3] - d.textbbox((0, 0), "H", font=f)[1] + nach
               for _, f, _, _, nach in zeilen) - zeilen[-1][4]


def _tafel(text: str, grund: bool = True) -> Image.Image:
    """Eine Aussage, groß, aber innerhalb des Satzspiegels (max. 1400 px breit), ruhig stehend – lesbar."""
    img = Image.new("RGBA", (W, H), GRUND if grund else (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    _ecken(d)
    teile = text.split("|")
    gr = 200
    while gr > 60:
        f = _font(FONT_B, gr)
        sp = int(gr * 0.06)
        if max(_breite(t, f, sp) for t in teile) <= 1400:
            break
        gr -= 6
    zeilen = [(t, f, sp, 255, int(gr * 0.32)) for t in teile]
    _zeilen(d, zeilen, (H - _hoehe(zeilen)) / 2)
    return img


def _endkarte(cfg: dict, jj: int, einsatz: list) -> Image.Image:
    """Endkarte: Name, Claim, Engine, Wettbewerb + Hashtags – in dieser Reihenfolge (nach Größe) je auf einem Schlag,
    harter Einsatz ohne Skalieren, danach stehend."""
    img = Image.new("RGBA", (W, H), GRUND)
    d = ImageDraw.Draw(img)
    _ecken(d)
    stufe = [0, 1, 2, 3, 3]                                 # Wettbewerb und Hashtags gleich groß -> gemeinsam
    zeilen = [(NAME, _font(FONT_B, 170), 18, 255, 60), (CLAIM, _font(FONT_B, 52), 8, 255, 92),
              (cfg["engine"], _font(FONT_B, 34), 4, 190, 38), ("COMFYUI · OPEN SOURCE HIGGSFIELD", _font(FONT_B, 32), 4,
              255, 34), ("#OpenSourceHiggsfield   #ComfyUI", _font(FONT_B, 32), 2, 255, 0)]
    y0 = (H - _hoehe(zeilen)) / 2
    sichtbar = [(t if jj >= einsatz[s] else "", f, sp, a, n) for (t, f, sp, a, n), s in zip(zeilen, stufe)]
    _zeilen(d, sichtbar, y0)
    if jj >= einsatz[2]:                                    # kurze Linie trennt Marke und Angaben
        ys = y0 + _hoehe(zeilen[:2]) + 92 / 2
        d.line([(W / 2 - 80, ys), (W / 2 + 80, ys)], fill=(255, 255, 255, 140), width=2)
    if jj < 6:
        _hieb(img, jj / 5, winkel=-6)
    return img


def _clipbild(frames: list, i: int) -> np.ndarray:
    img = cv2.imread(str(frames[min(i, len(frames) - 1)]))
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)


def _glitch(a: np.ndarray, b: np.ndarray, t: float, seed: int) -> np.ndarray:
    """Glitch von a nach b (t 0..1): Streifen springen nach b, Versatz und RGB-Split klingen ab – Übergang in den
    ersten Frame, damit das Video auf Instagram nahtlos im Loop läuft (Nutzer 29.10.)."""
    rng = np.random.default_rng(seed)
    out = a.copy()
    y = 0
    while y < H:
        hgt = int(rng.integers(6, 70))
        quelle = b if rng.random() < t else a
        dx = int(rng.integers(-1, 2) * (1 - t) * rng.integers(10, 140))
        out[y:y + hgt] = np.roll(quelle[y:y + hgt], dx, axis=1)
        y += hgt
    k = int((1 - t) * 24)
    if k:
        out[..., 2] = np.roll(out[..., 2], k, axis=1)
        out[..., 0] = np.roll(out[..., 0], -k, axis=1)
    for _ in range(int((1 - t) * 14)):                       # Blockrauschen
        bx, by = int(rng.integers(0, W - 160)), int(rng.integers(0, H - 60))
        out[by:by + int(rng.integers(8, 60)), bx:bx + int(rng.integers(40, 160))] = rng.integers(0, 255)
    return out


def _supercut_bild(frames: list, segmente: list, i: int, b0: int, k: int, beat: int, text: str | None) -> Image.Image:
    """Supercut im Takt: Schnitt nur auf Beat-Frames. Schläge 1 und 3 zeigen eine Welt (kurzer Zoom-Punch auf dem
    Schnitt), auf 2 und 4 steht die Aussage über dem abgedunkelten, weiterlaufenden Bild – Info und Bild im selben Fluss."""
    s0, s1 = segmente[(k // 2) % len(segmente)]
    idx = min(s1 - 1, s0 + 8 + (i - b0) + (beat if k % 2 else 0))
    img = _clipbild(frames, idx)
    seit = i - b0
    if not text:
        z = 1.0 + max(0, 3 - seit) * 0.02
        if z > 1.0:
            cw, ch = int(W / z), int(H / z)
            x0, y0 = (W - cw) // 2, (H - ch) // 2
            img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
    bild = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    if text:
        bild = Image.blend(bild, Image.new("RGBA", (W, H), GRUND), 0.72)
        bild.alpha_composite(_tafel(text, grund=False))
    else:
        _ecken(ImageDraw.Draw(bild))
    return bild.convert("RGB")


def bauen(name: str) -> Path:
    cfg = PROMOS[name]
    job = F / cfg["job"]
    ziel = ROOT / "out" / "promo" / f"{name}.mp4"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    info = json.loads((job / "audio.json").read_text(encoding="utf-8"))
    frames = sorted((job / "frames").glob("*.png"))
    n_clip = min(int(cfg["clip_s"] * FPS), len(frames))
    beats, takte, p = _raster(info, n_clip + 900)
    a0 = min([t for t in takte if t >= cfg.get("action_ab", 13.0) * FPS] or [n_clip])   # Supercut ab einer Eins
    bi = bisect.bisect_left(beats, a0)
    aussagen = AUSSAGEN.get(name, ["ONE CLIP|NO CUT"])
    n_sc = 2 * len(aussagen)                                # je Aussage: ein Schlag Bild, ein Schlag Aussage
    sc = beats[bi:bi + n_sc + 1]
    knall = sc[-1]                                          # Knall auf der Eins nach dem Supercut
    grenzen = [0] + sorted(info.get("welt_wechsel", [])) + [n_clip]
    segmente = [(a, b) for a, b in zip(grenzen[:-1], grenzen[1:]) if b - a > 2 * p] or [(0, n_clip)]
    ende0 = knall + 1
    einsatz = [0] + [beats[bi + n_sc + m] - ende0 for m in range(1, 4)]   # Endkarte baut sich auf den Schlägen auf
    ende_n = einsatz[3] + int(2.4 * FPS)                    # fertige Endkarte steht 2,4 s
    n_glitch = 8
    n = ende0 + ende_n + n_glitch
    with tempfile.TemporaryDirectory() as d:
        ov, fin = Path(d) / "ov", Path(d) / "fin"
        ov.mkdir()
        fin.mkdir()
        overlay_frames(dict(cfg, ende_s=0), ov)             # HUD und Weltwechsel-Wörter für die Clip-Frames
        for i in range(n):
            if i >= ende0 + ende_n:                         # Loop-Glitch in den ersten Frame
                t = (i - (ende0 + ende_n) + 1) / (n_glitch + 1)
                a_ = np.array(_endkarte(cfg, ende_n - 1, einsatz).convert("RGB"))
                b_ = np.array(Image.open(fin / "00000.png").convert("RGB"))
                bild = Image.fromarray(_glitch(a_, b_, t * t, seed=i))
            elif i >= ende0:
                bild = _endkarte(cfg, i - ende0, einsatz).convert("RGB")
            elif i == knall:                                # Knall: ein weißer Frame
                bild = Image.new("RGB", (W, H), (255, 255, 255))
            elif i >= a0:
                k = max(j for j, b in enumerate(sc[:-1]) if b <= i)
                bild = _supercut_bild(frames, segmente, i, sc[k], k, sc[k + 1] - sc[k],
                                      aussagen[k // 2] if k % 2 else None)
            else:
                b = _clipbild(frames, i)
                bild = Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)).convert("RGBA")
                o = ov / f"{i:05d}.png"
                if o.exists():
                    bild.alpha_composite(Image.open(o).convert("RGBA"))
                bild = bild.convert("RGB")
            bild.save(fin / f"{i:05d}.png")
        dauer = n / FPS
        ms = int(knall / FPS * 1000)
        # Ton: Bett aus dem Song + ein tiefer Impact auf dem Knall (Sinus-Sweep 110 -> 38 Hz, kurzer Rauschstoß)
        boom = ("aevalsrc=exprs=0.9*sin(2*PI*(38+72*exp(-9*t))*t)*exp(-4.5*t)+0.25*(random(0)*2-1)*exp(-30*t)"
                ":s=48000:d=1.2")
        filt = (f"[1:a]aresample=48000,afade=t=in:d=0.03,afade=t=out:st={dauer - 0.3:.2f}:d=0.3[bett];"
                f"[2:a]adelay={ms}|{ms},aformat=channel_layouts=stereo,volume=0.8[b];"
                "[bett][b]amix=inputs=2:normalize=0:duration=first[a]")
        proc.run([proc.tool("ffmpeg"), "-y", "-v", "error", "-framerate", str(FPS), "-i", str(fin / "%05d.png"),
                  "-ss", f"{cfg['start']:.3f}", "-t", f"{dauer:.3f}", "-i", str(MUSIK / cfg["song"]),
                  "-f", "lavfi", "-i", boom, "-filter_complex", filt,
                  "-map", "0:v", "-map", "[a]", "-t", f"{dauer:.3f}", "-c:v", "libx264", "-crf", "18",
                  "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                  str(ziel)], "Promo")
    return ziel


def main() -> int:
    for name in (sys.argv[1:] or list(PROMOS)):
        print(bauen(name), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
