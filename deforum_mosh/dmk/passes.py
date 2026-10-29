"""dmk.passes: Render-Pässe aus Blender lesen (EXR MultiLayer und PNG-Kontrollbilder).

Liefert pro Frame: Tiefe (linear, in Szeneneinheiten), Disparität (1/Z, fest normiert), Bewegungsvektoren
(Pixel, zum vorigen und zum nächsten Frame, in unserer Konvention: bwd[y,x] = Position des Pixels im VORIGEN
Frame minus (x,y), also direkt für cv2.remap als Rück-Abbildung), Normalen (Kameraraum), Licht, IDs.

EXR: bevorzugt das Paket OpenEXR (liest alle Kanäle mit Namen); ohne das Paket nur Einzel-Part-RGBA über OpenCV.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np

try:
    import OpenEXR  # type: ignore
    import Imath  # type: ignore
    _HAVE_OPENEXR = True
except Exception:  # pragma: no cover
    _HAVE_OPENEXR = False


def exr_kanaele(path) -> list[str]:
    if not _HAVE_OPENEXR:
        return []
    f = OpenEXR.File(str(path))
    return sorted(n for part in f.parts for n in part.channels.keys())


def lese_exr(path, kanaele: list[str] | None = None, teile: list[str] | None = None) -> dict[str, np.ndarray]:
    """Alle (oder die genannten) Kanäle als float32-Arrays, Schlüssel = Kanalname wie in der Datei.
    Blender 5 schreibt MultiPART-EXR (ein Part je Pass, z. B. 'ViewLayer.Vector'); gelesen wird über die
    OpenEXR-3-API (OpenEXR.File → parts → channels). RGB(A)-Pässe liegen als ein Kanal (h, w, 3|4) vor,
    Datenpässe als einzelne Kanäle 'ViewLayer.Vector.X' usw. `teile` schränkt auf Part-Namen ein (Substring)."""
    path = str(path)
    if _HAVE_OPENEXR:
        f = OpenEXR.File(path)
        out = {}
        for part in f.parts:
            pname = part.name()
            if teile and not any(t.lower() in pname.lower() for t in teile):
                continue
            for n, ch in part.channels.items():
                if kanaele and n not in kanaele:
                    continue
                arr = np.asarray(ch.pixels)
                out[n] = arr if arr.dtype == np.uint32 else arr.astype(np.float32, copy=False)
        return out
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"EXR nicht lesbar (ohne OpenEXR-Paket nur Einzel-Part-RGBA): {path}")
    if img.ndim == 2:
        return {"Y": img.astype(np.float32)}
    names = ["B", "G", "R", "A"][:img.shape[2]]
    return {n: img[..., i].astype(np.float32) for i, n in enumerate(names)}


def finde_kanal(kanaele: dict[str, np.ndarray], *muster: str) -> np.ndarray | None:
    """Ersten Kanal, dessen Name (ohne View-Layer-Präfix) auf eines der Regex-Muster passt."""
    for m in muster:
        rx = re.compile(m, re.I)
        for name, arr in kanaele.items():
            kurz = name.split(".", 1)[1] if "." in name and not name.split(".")[0] in ("R", "G", "B", "A") else name
            if rx.search(kurz) or rx.search(name):
                return arr
    return None


def tiefe_aus_exr(kanaele: dict[str, np.ndarray]) -> np.ndarray | None:
    z = finde_kanal(kanaele, r"^Depth\.Z$", r"Depth\.Z", r"\.Z$", r"^Z$")
    if z is None:
        return None
    z = z.copy()
    z[~np.isfinite(z)] = np.nan
    return z


def disparitaet(z: np.ndarray, z_nah: float, z_fern: float) -> np.ndarray:
    """1/Z, mit FESTER Normierung über die ganze Sequenz: z_nah → 1 (hell = nah), z_fern → 0."""
    zc = np.clip(np.nan_to_num(z, nan=z_fern), z_nah, z_fern)
    d = (1.0 / zc - 1.0 / z_fern) / (1.0 / z_nah - 1.0 / z_fern)
    return d.astype(np.float32)


def vektoren_aus_exr(kanaele: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Blender Vector-Pass (X, Y, Z, W): (X, Y) Bewegung zum vorigen, (Z, W) zum nächsten Frame, in Pixeln.
    Rückgabe (zum_vorigen, zum_naechsten) als (h, w, 2)-Felder in Bildkoordinaten (x rechts, y runter).
    Blenders Y zeigt nach oben → Vorzeichen der Y-Komponente wird gedreht. Vorzeichen/Zuordnung gegen das
    Testobjekt aus BLENDER_PASSES_PROMPT prüfen (meta.json)."""
    x = finde_kanal(kanaele, r"Vector\.X$")
    y = finde_kanal(kanaele, r"Vector\.Y$")
    z = finde_kanal(kanaele, r"Vector\.Z$")
    w = finde_kanal(kanaele, r"Vector\.W$")
    if x is None or y is None:
        return None
    vor = np.stack([x, -y], -1).astype(np.float32)
    nach = np.stack([z, -w], -1).astype(np.float32) if z is not None and w is not None else None
    return vor, nach


def normalen_aus_exr(kanaele: dict[str, np.ndarray]) -> np.ndarray | None:
    n = [finde_kanal(kanaele, rf"Normal\.{c}$") for c in "XYZ"]
    if any(v is None for v in n):
        return None
    return np.stack(n, -1).astype(np.float32)


class PassSequenz:
    """Ordner nach BLENDER_PASSES_PROMPT: <seq>/beauty, exr, depth, normal, licht, kanten, ids, albedo, meta.json.
    Ordnet alles über die Framenummer zu und hält die feste Tiefennormierung aus meta.json."""

    def __init__(self, root):
        self.root = Path(root)
        self.meta = {}
        mp = self.root / "meta.json"
        if mp.exists():
            try:
                self.meta = json.loads(mp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.meta = {}
        self.exr = self._nummern(self.root / "exr", (".exr",))
        self.png = {d.name: self._nummern(d, (".png", ".tif", ".tiff", ".jpg"))
                    for d in self.root.iterdir() if d.is_dir() and d.name != "exr"}
        # meta.json aus dem Blender-Prompt: normalisation.depth_near_m / depth_far_m
        norm = self.meta.get("normalisation") or {}
        z = self.meta.get("tiefe") or self.meta.get("depth") or {}
        self.z_nah = float(norm.get("depth_near_m", z.get("nah", z.get("near", 0.1))))
        self.z_fern = float(norm.get("depth_far_m", z.get("fern", z.get("far", 100.0))))
        self.fps = float(self.meta.get("fps", 24.0))
        self.resolution = tuple(self.meta.get("resolution", ()))

    @staticmethod
    def _nummern(d: Path, exts) -> dict[int, Path]:
        if not d.is_dir():
            return {}
        out = {}
        for p in d.iterdir():
            if p.suffix.lower() in exts:
                m = re.findall(r"\d+", p.stem)
                if m:
                    out[int(m[-1])] = p
        return out

    def frames(self) -> list[int]:
        return sorted(self.png.get("beauty", self.exr).keys())

    def exr_frame(self, nr: int, teile: list[str] | None = None) -> dict[str, np.ndarray] | None:
        p = self.exr.get(nr)
        return lese_exr(p, teile=teile) if p else None

    def bewegung_rueck(self, nr: int, size: tuple[int, int] | None = None) -> np.ndarray | None:
        """Rück-Abbildung für cv2.remap: für jeden Pixel von Frame nr der Versatz zur Position im Frame nr−1."""
        k = self.exr_frame(nr, ["Vector"])
        if not k:
            return None
        v = vektoren_aus_exr(k)
        if v is None:
            return None
        vor = v[0]
        if size is not None and (vor.shape[1], vor.shape[0]) != size:
            sx, sy = size[0] / vor.shape[1], size[1] / vor.shape[0]
            vor = cv2.resize(vor, size, interpolation=cv2.INTER_LINEAR) * np.array([sx, sy], np.float32)
        return vor

    def rueckabbildung(self, nr_von: int, nr_zu: int, size: tuple[int, int], start=None):
        """Verkettete Rück-Abbildung über mehrere Quellframes (stride): für jeden Pixel von Frame nr_von die
        Position im Frame nr_zu (< nr_von), als (mx, my) für cv2.remap, plus Gültigkeit (im Bild geblieben und
        Objekt-ID unverändert = nicht verdeckt/aufgedeckt). Vector-Pass: (X, Y) = Versatz zum vorigen Frame,
        Y nach oben → gedreht; gegen DIS geprüft (vector_check, 2026-10-09).
        start: (sx, sy) Startpositionen in Quellpixeln statt des Identitätsgitters (digitale Kamerafahrt:
        das gezoomte Pixel liegt in der Quelle bei origin + p/z). Ergebnis dann ebenfalls in Quellpixeln."""
        w, h = size
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        if start is None:
            mx, my = xs.copy(), ys.copy()
        else:
            mx, my = start[0].astype(np.float32).copy(), start[1].astype(np.float32).copy()
        ok = np.ones((h, w), bool)
        ids0 = self.objekt_ids(nr_von, size)
        if ids0 is not None and start is not None:
            ids0 = cv2.remap(ids0.astype(np.float32), mx, my, cv2.INTER_NEAREST)
        for nr in range(nr_von, nr_zu, -1):
            v = self.bewegung_rueck(nr, size)
            if v is None:
                return None
            vx = cv2.remap(v[..., 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            vy = cv2.remap(v[..., 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mx, my = mx + vx, my + vy
            ok &= (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
        ids1 = self.objekt_ids(nr_zu, size)
        if ids0 is not None and ids1 is not None:
            at = cv2.remap(ids1.astype(np.float32), mx, my, cv2.INTER_NEAREST)
            ok &= np.abs(at - ids0) < 0.5
        return mx, my, ok

    def referenz_bewegung(self, frame_nrs: list[int], qz=None, zentrum=(0.5, 0.5)):
        """Für anker.messen: ref_bewegung(i, size) -> (bwd, gueltig) zwischen den benutzten Quellframes
        frame_nrs[i] → frame_nrs[i+1] (stride wird über die Verkettung abgedeckt).
        qz: kumulativer Zoomfaktor je Ausgabeframe (digitale Kamerafahrt, wie in method_a: Pixel p des gezoomten
        Frames liegt in der Quelle bei zoom_origin + p/qz); dann gilt die Bewegung zwischen den GEZOOMTEN Frames."""
        from . import core

        def fn(i, size):
            w, h = size
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            if qz is None:
                rb = self.rueckabbildung(frame_nrs[i + 1], frame_nrs[i], size)
                if rb is None:
                    return None
                mx, my, ok = rb
            else:
                z1, z0 = float(qz[i + 1]), float(qz[i])
                ox, oy = core.zoom_origin(w, h, z1, *zentrum)
                rb = self.rueckabbildung(frame_nrs[i + 1], frame_nrs[i], size, start=(ox + xs / z1, oy + ys / z1))
                if rb is None:
                    return None
                mx, my, ok = rb
                px, py = core.zoom_origin(w, h, z0, *zentrum)
                mx, my = (mx - px) * z0, (my - py) * z0
                ok = ok & (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
            return np.stack([mx - xs, my - ys], -1), ok
        return fn

    def objekt_ids(self, nr: int, size: tuple[int, int] | None = None) -> np.ndarray | None:
        k = self.exr_frame(nr, ["Object Index"])
        if not k:
            return None
        ids = next(iter(k.values()))
        if size is not None:
            ids = cv2.resize(ids, size, interpolation=cv2.INTER_NEAREST)
        return ids

    def tiefe(self, nr: int, size: tuple[int, int] | None = None) -> np.ndarray | None:
        """Disparität 0..1 (hell = nah), fest normiert; aus EXR-Z, sonst aus depth/<nr>.png (bereits normiert)."""
        k = self.exr_frame(nr, ["Depth"])
        z = tiefe_aus_exr(k) if k else None
        if z is not None:
            d = disparitaet(z, self.z_nah, self.z_fern)
        else:
            p = self.png.get("depth", {}).get(nr)
            if not p:
                return None
            raw = cv2.imread(str(p), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            d = raw.astype(np.float32) / (65535.0 if raw.dtype == np.uint16 else 255.0)
        if size is not None:
            d = cv2.resize(d, size, interpolation=cv2.INTER_LINEAR)
        return d
