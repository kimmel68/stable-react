"""dmk.core: Schedule-Parser, Warp, Sequenz-IO, Metriken, Kontaktbögen."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import proc

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
BORDERS = {
    "reflect": cv2.BORDER_REFLECT_101,
    "wrap": cv2.BORDER_WRAP,
    "replicate": cv2.BORDER_REPLICATE,
    "zero": cv2.BORDER_CONSTANT,
}
_DEPTH_PAT = re.compile(r"depth|tiefe", re.I)
_SEQ_PAT = re.compile(r"taglauf|b[\s_\-]*sequenz", re.I)
_SKIP_DIRS = {"$recycle.bin", "windows", "node_modules", ".git", "site-packages", "__pycache__",
              "system volume information", "program files", "program files (x86)", "appdata"}


# ---------------------------------------------------------------- Schedules

_KEY = re.compile(r"(\d+)\s*:\s*\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)")


def parse_schedule(text) -> list[tuple[int, float]]:
    """Deforum-Keyframes: "0:(1.01), 60:(1.04)". Eine nackte Zahl gilt als konstanter Wert."""
    if isinstance(text, (int, float)):
        return [(0, float(text))]
    text = str(text).strip()
    keys = [(int(f), float(v)) for f, v in _KEY.findall(text)]
    if not keys:
        try:
            return [(0, float(text))]
        except ValueError:
            raise ValueError(f"Schedule nicht lesbar: {text!r}") from None
    return sorted(keys)


def schedule(text, n: int) -> np.ndarray:
    """Wert pro Frame: linear interpoliert, an den Rändern gehalten."""
    keys = parse_schedule(text)
    return np.interp(np.arange(n), [k for k, _ in keys], [v for _, v in keys])


# ---------------------------------------------------------------- Bild-IO

def imread(path, max_side: int = 0, gray: bool = False) -> np.ndarray:
    # np.fromfile statt cv2.imread: funktioniert auch mit Umlauten im Pfad
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Bild nicht lesbar: {path}")
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img


def imwrite(path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_PNG_COMPRESSION, 1] if path.suffix.lower() == ".png" else []
    ok, buf = cv2.imencode(path.suffix, img, params)
    if not ok:
        raise IOError(f"Bild nicht schreibbar: {path}")
    buf.tofile(str(path))


def zoom_origin(w: int, h: int, z: float, cx: float = 0.5, cy: float = 0.5) -> tuple[float, float]:
    """Ursprung (x0, y0) des Ausschnitts 1/z um (cx, cy) in Originalpixeln. Gezoomtes Pixel p ↔ Original
    s = origin + p / z (gleiche Formel wie zoom_crop, damit Bild, Tiefe, Vektoren und IDs exakt zusammenpassen)."""
    cw, ch = w / z, h / z
    return float(np.clip(cx * w - cw / 2, 0, w - cw)), float(np.clip(cy * h - ch / 2, 0, h - ch))


def zoom_crop(img, z: float, cx: float = 0.5, cy: float = 0.5, interp=cv2.INTER_LANCZOS4):
    """Digitale Kamerafahrt: Ausschnitt 1/z um (cx, cy) (Anteile der Bildgröße) auf volle Größe ziehen.
    Subpixelgenau, damit ein Zoom von wenigen Promille pro Frame gleichmäßig fließt."""
    if z <= 1.0 + 1e-6:
        return img
    h, w = img.shape[:2]
    x0, y0 = zoom_origin(w, h, z, cx, cy)
    m = np.float32([[1 / z, 0, x0], [0, 1 / z, y0]])
    return cv2.warpAffine(img, m, (w, h), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)


def read_depth(path, size: tuple[int, int] | None, invert: bool = False) -> np.ndarray:
    """Tiefenkarte als float32 0..1 in Zielgröße (w, h), None = Originalgröße. Konvention: hell = nah."""
    data = np.fromfile(str(path), dtype=np.uint8)
    d = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if d is None:
        raise IOError(f"Tiefenkarte nicht lesbar: {path}")
    d = d.astype(np.float32)
    if size is not None:
        d = cv2.resize(d, size, interpolation=cv2.INTER_LINEAR)
    lo, hi = float(d.min()), float(d.max())
    d = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    return 1.0 - d if invert else d


def prep(img: np.ndarray, max_side: int = 0, multiple: int = 1) -> np.ndarray:
    """Auf max_side verkleinern und mittig auf ein Vielfaches von `multiple` beschneiden."""
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    nh, nw = h - h % multiple, w - w % multiple
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    return img[y0:y0 + nh, x0:x0 + nw]


# ---------------------------------------------------------------- Sequenzen

def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(p).name)]


def list_images(folder: Path) -> list[Path]:
    try:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]
    except OSError:
        return []
    return sorted(files, key=natural_key)


# Render-Pässe aus Blender (BLENDER_PASSES_PROMPT.md): Unterordner neben dem Beauty, gleiche Framenummern
PASS_NAMES = {"normal": ("normal", "normalen"), "licht": ("licht", "light", "diffuse_light"),
              "kanten": ("kanten", "edges", "lineart", "line"), "ids": ("ids", "index", "objectindex"),
              "albedo": ("albedo", "diffcol", "diffuse_color"), "mist": ("mist", "nebel", "volume")}
_BEAUTY_NAMES = ("beauty", "frames", "combined", "image")


def _pass_kind(name: str) -> str | None:
    n = name.lower()
    if _DEPTH_PAT.search(n):
        return "depth"
    for kind, keys in PASS_NAMES.items():
        if any(n == k or n.startswith(k) for k in keys):
            return kind
    return "exr" if n == "exr" else None


@dataclass
class Sequence:
    root: Path
    frames: list[Path]
    depth: list[Path] | None = None
    passes: dict | None = None   # {"licht": [Pfade], "kanten": [...], ...} deckungsgleich mit frames

    def __len__(self):
        return len(self.frames)

    @property
    def name(self) -> str:
        """Kurzname für Cache-Ordner: gemeinsames Präfix der Frames oder der Ordnername."""
        stems = [p.stem for p in self.frames[:2]]
        pre = re.sub(r"[._-]?\d*$", "", os.path.commonprefix(stems))  # Zähler abtrennen, Name behalten
        return pre or self.root.name


def _frames_dir(root: Path) -> Path:
    own = [p for p in list_images(root) if not _DEPTH_PAT.search(p.stem)]
    if len(own) >= 2:
        return root
    for name in _BEAUTY_NAMES:  # Pass-Ordnerstruktur: das Beauty hat Vorrang vor allen Datenpässen
        d = root / name
        if d.is_dir() and len(list_images(d)) >= 2:
            return d
    best, best_n = None, 0
    for dirpath, dirnames, _ in os.walk(root):
        d = Path(dirpath)
        if len(d.relative_to(root).parts) >= 3:
            dirnames[:] = []
        dirnames[:] = [x for x in dirnames if _pass_kind(x) is None]
        if d == root:
            continue
        n = len(list_images(d))
        # "frames" gewinnt bei Gleichstand, sonst der Ordner mit den meisten Bildern
        if n > best_n or (n == best_n and n and d.name.lower() == "frames"):
            best, best_n = d, n
    if best is None:
        raise FileNotFoundError(f"Keine Bildsequenz unter {root} gefunden.")
    return best


def _find_depth(frames_dir: Path, root: Path) -> list[Path] | None:
    inline = [p for p in list_images(frames_dir) if _DEPTH_PAT.search(p.stem)]
    if len(inline) >= 2:
        return inline
    for base in dict.fromkeys([frames_dir, frames_dir.parent, root]):
        try:
            subs = [d for d in base.iterdir() if d.is_dir() and _DEPTH_PAT.search(d.name)]
        except OSError:
            continue
        for d in sorted(subs):
            files = list_images(d)
            if len(files) >= 2:
                return files
    return None


def load_sequence(seq_root, count: int = 0, start: int = 0, pattern: str | None = None,
                  depth_dir=None, stride: int = 1, wrap: bool = False) -> Sequence:
    """pattern: Regex auf den Dateinamen, für gemischte Ordner (z. B. "LOOP_KREUZGANG_LUMEN2\\.").
    depth_dir: Ordner mit passenden Tiefenkarten (z. B. aus dmk.depth), sonst wird gesucht.
    stride: nur jeden n-ten Quellframe nehmen (mehr Bewegung pro Ausgabeframe); count zählt Ausgabeframes."""
    root = Path(seq_root)
    if root.is_file() and root.suffix.lower() in IMG_EXT:
        # Einzelbild als Quelle (App: Foto/Render + Track): dasselbe Bild für jeden Frame, die Bewegung kommt
        # aus der virtuellen Kamera (motion="virtuell"), die Tiefe aus der Schätzung
        n = max(2, int(count or 120))
        return Sequence(root=root.parent, frames=[root] * n, depth=None, passes={})
    if not root.is_dir():
        raise FileNotFoundError(f"Sequenzordner fehlt: {root}")
    if pattern:
        pat = re.compile(pattern, re.I)
        fdir = root
        frames = [p for p in list_images(root) if pat.search(p.name) and not _DEPTH_PAT.search(p.stem)]
    else:
        fdir = _frames_dir(root)
        frames = [p for p in list_images(fdir) if not _DEPTH_PAT.search(p.stem)]
    if depth_dir:
        depth = [Path(depth_dir) / (p.stem + "_depth.png") for p in frames]
        depth = depth if all(d.exists() for d in depth) else None
    else:
        depth = _find_depth(fdir, root) if not pattern else None
    stride = max(1, int(stride))
    if wrap and count and frames:
        # Loop-Sequenz zyklisch durchlaufen: mit großem stride mehrfach rundherum (mehr Bewegung pro Frame)
        idx = [(start + i * stride) % len(frames) for i in range(count)]
        depth = [depth[j] for j in idx] if depth and len(depth) >= len(frames) else None
        frames = [frames[j] for j in idx]
    else:
        end = start + count * stride if count else None
        frames = frames[start:end:stride]
        if depth:
            depth = depth[start:end:stride]
            if len(depth) < len(frames):
                depth = None
    if len(frames) < 2:
        raise ValueError(f"Zu wenige Frames in {fdir} (start={start}, count={count}, stride={stride}, pattern={pattern!r}).")
    return Sequence(root=fdir, frames=frames, depth=depth, passes=_find_passes(fdir, frames))


def _find_passes(fdir: Path, frames: list[Path]) -> dict:
    """Pass-Ordner neben dem Beauty suchen und über die Framenummer den Frames zuordnen (stride/wrap-fest)."""
    num = lambda p: (re.findall(r"\d+", Path(p).stem) or [""])[-1].lstrip("0") or "0"
    out = {}
    if fdir.name.lower() not in _BEAUTY_NAMES:  # nur in einer echten Pass-Struktur, sonst sind Nachbarordner fremd
        return out
    try:
        subs = [d for d in fdir.parent.iterdir() if d.is_dir() and d != fdir]
    except OSError:
        return out
    for d in subs:
        kind = _pass_kind(d.name)
        if kind in (None, "depth") or kind in out:
            continue
        files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT | {".exr"}]
        by_num = {num(p): p for p in files}
        matched = [by_num.get(num(f)) for f in frames]
        if matched and all(m is not None for m in matched):
            out[kind] = matched
    return out


def discover(roots, max_depth: int = 6, min_frames: int = 8) -> list[dict]:
    """Tolerante Suche nach Taglauf-/B-Sequenz-Ordnern mit Bildsequenzen."""
    hits = []
    for root in roots:
        root = Path(root)
        for dirpath, dirnames, _ in os.walk(root):
            d = Path(dirpath)
            depth = len(d.relative_to(root).parts)
            dirnames[:] = [x for x in dirnames if x.lower() not in _SKIP_DIRS and not x.startswith(".")]
            if depth >= max_depth:
                dirnames[:] = []
            if not any(_SEQ_PAT.search(part) for part in d.parts):
                continue
            imgs = list_images(d)
            if len(imgs) < min_frames or _DEPTH_PAT.search(d.name):
                continue
            try:
                h, w = imread(imgs[0]).shape[:2]
            except IOError:
                continue
            hits.append({"path": str(d), "frames": len(imgs), "size": f"{w}x{h}",
                         "depth": _find_depth(d, d) is not None})
    return hits


# ---------------------------------------------------------------- Warp

def warp(img, zoom=1.0, angle=0.0, tx=0.0, ty=0.0, border="reflect", depth=None, parallax=0.0):
    """Zoom/Rotation um die Bildmitte plus Verschiebung. Mit depth+parallax zoomt Nahes stärker."""
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    mode = BORDERS[border]
    if depth is None or not parallax:
        m = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
        m[0, 2] += tx
        m[1, 2] += ty
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=mode)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    z = 1.0 + (zoom - 1.0) * (1.0 + parallax * (2.0 * depth - 1.0))
    a = np.deg2rad(angle)
    dx, dy = xs - cx - tx, ys - cy - ty
    mx = (np.cos(a) * dx - np.sin(a) * dy) / z + cx
    my = (np.sin(a) * dx + np.cos(a) * dy) / z + cy
    return cv2.remap(img, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LANCZOS4, borderMode=mode)


def hf_inject(base, src, k: float, sigma: float = 2.0, mode: str = "replace"):
    """Hochfrequenzen (Kanten, Textur) des Quellframes in das weitergetragene Bild zurückspeisen.
    Resampling und Blending wirken als Tiefpass; so bleibt die Schärfe der Quelle erhalten,
    während Farbe und Form aus dem Feedback kommen.

    mode="replace": Hochband des Bildes wird zu k durch das der Quelle ersetzt; beschränkt, Schärfe ≈ Quelle.
    mode="add":     Hochband wird addiert und schaukelt sich im Feedback-Loop auf (det_ratio 8+): Streifen
                    entlang der Bewegung, der "Streifen-Look". Bewusst wählbar, nicht Standard."""
    if k <= 0:
        return base
    b = base.astype(np.float32)
    s = src.astype(np.float32)
    hp_src = s - cv2.GaussianBlur(s, (0, 0), sigma)
    if mode == "add":
        return np.clip(b + float(k) * hp_src, 0, 255).astype(np.uint8)
    k = min(float(k), 1.0)
    lo_base = cv2.GaussianBlur(b, (0, 0), sigma)
    out = lo_base + (1.0 - k) * (b - lo_base) + k * hp_src
    return np.clip(out, 0, 255).astype(np.uint8)


def load_mask(path, size: tuple[int, int]) -> np.ndarray:
    """Maske als float32 0..1 in Zielgröße (w, h): weiß = Bildinhalt, schwarz = bleibt schwarz
    (Projektionsmapping: außerhalb der Gebäudekontur darf nichts entstehen)."""
    m = imread(path, gray=True)
    m = cv2.resize(m, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return m[..., None]


def apply_mask(img, mask) -> np.ndarray:
    if mask is None:
        return img
    return (img.astype(np.float32) * mask).astype(np.uint8)


def licht_transfer(img, src, k: float, sigma: float = 14.0):
    """Lichtkomposition der Quelle übernehmen: WO Licht und Schatten liegen (Tiefpass der Helligkeit), nicht
    wie hell das Bild insgesamt ist. Das Tiefband wird z-normiert übertragen, Palette und Kontrast des
    generierten Bildes bleiben. Textur und Inhalt (Hochband) werden nicht angefasst."""
    if k <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ls = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo_i = cv2.GaussianBlur(lab[..., 0], (0, 0), sigma)
    lo_s = cv2.GaussianBlur(ls, (0, 0), sigma)
    target = lo_i.mean() + (lo_s - lo_s.mean()) / (lo_s.std() + 1e-3) * lo_i.std()
    lab[..., 0] = np.clip(lab[..., 0] + float(k) * (target - lo_i), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def licht_korrelation(gray, ref_gray) -> float:
    """Liegt das Licht noch an derselben Stelle? Pearson-Korrelation der tiefpassgefilterten Helligkeit."""
    s = max(gray.shape) / 22.0
    a = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), s)
    b = cv2.GaussianBlur(ref_gray.astype(np.float32), (0, 0), s)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def lap_var(img) -> float:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def schaerfe_angleichen(img, ref, max_gain: float = 2.0, sigma: float = 1.0):
    """Schärfe (Laplace-Varianz) von img an ref angleichen – gegen das Pumpen bei Cadence: die Zwischenframes sind
    Mischungen zweier Samples und verlieren dabei etwa die Hälfte der Hochfrequenz-Varianz, der Sample-Frame nicht.
    Unscharfmaske mit Verstärkung aus dem Varianzverhältnis (Laplace-Varianz skaliert ~ (1+a)²)."""
    lv_i, lv_r = lap_var(img), lap_var(ref)
    if lv_i <= 1e-6 or lv_i >= 0.95 * lv_r:
        return img
    a = float(np.clip(np.sqrt(lv_r / lv_i) - 1.0, 0.0, max_gain))
    soft = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + a, soft, -a, 0)


def edge_map(gray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    return np.clip(m / (np.percentile(m, 99) + 1e-6) * 255, 0, 255).astype(np.uint8)


def match_color(img, ref):
    """Mittelwert/Streuung pro LAB-Kanal an ref angleichen (gegen Farbdrift im Feedback-Loop)."""
    a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    ma, sa = a.mean((0, 1)), a.std((0, 1)) + 1e-6
    mb, sb = b.mean((0, 1)), b.std((0, 1))
    out = (a - ma) / sa * sb + mb
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- Metriken

def farneback(a_gray, b_gray):
    return cv2.calcOpticalFlowFarneback(a_gray, b_gray, None, 0.5, 4, 21, 3, 7, 1.5, 0)


def blockiness(gray: np.ndarray, n: int = 16) -> float:
    """Gradient an den Makroblock-Kanten relativ zum Rest. ~1.0 = keine Blockstruktur."""
    g = gray.astype(np.float32)
    dx, dy = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    bx = (np.arange(dx.shape[1]) % n) == n - 1
    by = (np.arange(dy.shape[0]) % n) == n - 1
    if not bx.any() or not by.any():
        return 1.0
    edge = (dx[:, bx].mean() + dy[by].mean()) / 2
    inner = (dx[:, ~bx].mean() + dy[~by].mean()) / 2
    return float(edge / (inner + 1e-6))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float32), b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    va, vb = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def _small(gray, side):
    h, w = gray.shape[:2]
    s = side / max(h, w)
    return gray if s >= 1 else cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def analyse(frame_paths, ref_paths=None, sub: int = 1, side: int = 320) -> dict:
    """Kennzahlen einer Framefolge. ref_paths: Eingangssequenz für SSIM_ref (Index n/sub)."""
    paths = list(frame_paths)
    refs = list(ref_paths) if ref_paths else None
    prev = None
    mags, zooms, werr, det, blk, sim, still, ratio, raum, prev_sim = [], [], [], [], [], [], [], [], [], []
    licht, hell = [], []
    grid = None
    for i, p in enumerate(paths):
        full = imread(p, gray=True)
        blk.append(blockiness(full))
        g = _small(full, side)
        lap = float(cv2.Laplacian(g, cv2.CV_32F).var())
        det.append(lap)
        if refs:
            r = imread(refs[min(round(i / sub), len(refs) - 1)], gray=True)
            r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
            sim.append(ssim(g, r))
            # Schärfe relativ zur Quelle und Erhalt der Raumstruktur (Kantenbild gegen Quelle)
            ratio.append(lap / (float(cv2.Laplacian(r, cv2.CV_32F).var()) + 1e-6))
            raum.append(ssim(edge_map(g), edge_map(r)))
            licht.append(licht_korrelation(g, r))
        hell.append(float(g.mean()))
        if prev is not None and prev.shape == g.shape:
            f = farneback(prev, g)
            if grid is None:
                h, w = g.shape
                ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
                rx, ry = xs - (w - 1) / 2, ys - (h - 1) / 2
                grid = (xs, ys, rx, ry, float((rx * rx + ry * ry).sum()))
            xs, ys, rx, ry, rr = grid
            mags.append(float(np.linalg.norm(f, axis=2).mean()))
            # reiner Zoom z um die Mitte: f(p) = (z-1)(p-c) -> Kleinste-Quadrate-Fit des Radialanteils
            zooms.append(1.0 + float((f[..., 0] * rx + f[..., 1] * ry).sum()) / rr)
            back = cv2.remap(g, xs + f[..., 0], ys + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            werr.append(float(np.abs(back.astype(np.float32) - prev).mean()))
            still.append(float((np.abs(g.astype(np.int16) - prev) <= 1).mean()))
            prev_sim.append(ssim(g, prev))  # Wandel pro Frame: 1 = eingefroren
        prev = g
    third = max(1, len(det) // 3)
    m = np.array(mags) if mags else np.zeros(1)
    return {
        "frames": len(paths),
        "warp_error": float(np.mean(werr)) if werr else 0.0,
        "jerk": float(np.std(np.diff(m, 2)) / (m.mean() + 1e-3)) if len(m) > 2 else 0.0,
        "zoom": float(np.mean(zooms)) if zooms else 1.0,
        "zoom_std": float(np.std(zooms)) if zooms else 0.0,
        "det_drift": float(np.mean(det[-third:]) / (np.mean(det[:third]) + 1e-6)),
        "blockiness": float(np.mean(blk)),
        "ssim_ref": float(np.mean(sim)) if sim else None,
        "det_ratio": float(np.mean(ratio)) if ratio else None,
        "raum": float(np.mean(raum)) if raum else None,
        "smear": float(np.mean(still)) if still else 0.0,
        "ssim_prev": float(np.mean(prev_sim)) if prev_sim else None,
        "flow_px": float(m.mean()) if mags else 0.0,
        "licht": float(np.mean(licht)) if licht else None,
        "textur": float(np.mean(det)),          # absolut; Deforum-Referenz 1275
        "helligkeit": float(np.mean(hell)),     # Deforum-Referenz 108
    }


def ssim_map(a: np.ndarray, b: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Lokale SSIM-Karte (größeres Fenster als ssim(): bewertet Bereiche, nicht Pixel)."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), sigma)
    mu_a, mu_b = blur(a), blur(b)
    va, vb = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))


def mitfluss(frame_paths, soll_path, side: int = 320):
    """Wandert der Bildinhalt so, wie die Kamera es verlangt? Vergleicht die im ERGEBNIS gemessene Bewegung
    mit der Soll-Bewegung aus dem Kameramodell der Quelle (soll_fluss.npz des Laufs).
    1 = Inhalt fließt exakt mit der Szene, 0 = Inhalt steht auf dem Bildschirm, während die Kamera fährt."""
    try:
        soll = np.load(soll_path)["bwd"].astype(np.float32)
    except (OSError, KeyError):
        return None
    paths = list(frame_paths)
    grays = [_small(imread(p, gray=True), side) for p in paths[:len(soll) + 1]]
    vals = []
    for i in range(1, len(grays)):
        h, w = grays[i].shape
        ist = farneback(grays[i], grays[i - 1])                       # Frame i -> i-1, im Ergebnis gemessen
        s = cv2.resize(soll[i - 1], (w, h), interpolation=cv2.INTER_LINEAR) * w
        vals.append(1.0 - float(np.linalg.norm(ist - s, axis=2).mean()) / (float(np.linalg.norm(s, axis=2).mean()) + 0.05))
    return float(np.clip(np.mean(vals), -1, 1)) if vals else None


def punkt_leben(frame_paths, k: int = 8, side: int = 480, step: int = 4) -> dict:
    """Identitätssprünge, wie das Auge sie sieht: Anteil markanter Punkte (ORB), die k Frames später an der
    bewegungskompensierten Stelle mit ähnlichem Aussehen wiedergefunden werden. `erscheinen` zählt Flecken, die
    pro Frame neu auftauchen oder verschwinden (bewegungskompensierte Differenz, Schwelle, Komponenten)."""
    paths = list(frame_paths)
    grays = [_small(imread(p, gray=True), side) for p in paths]
    if len(grays) < k + 2:
        return {}
    h, w = grays[0].shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    orb = cv2.ORB_create(nfeatures=600, fastThreshold=10)
    flows = [farneback(grays[i], grays[i + 1]) for i in range(len(grays) - 1)]
    leben, ersch = [], []
    for a in range(0, len(grays) - k, step):
        kp, des = orb.detectAndCompute(grays[a], None)
        if des is None or len(kp) < 20:
            continue
        pts = np.array([p.pt for p in kp], np.float32)
        for j in range(k):  # Punkte entlang der Bahn vorwärts tragen
            f = flows[a + j]
            fx = cv2.remap(f[..., 0], pts[:, 0], pts[:, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fy = cv2.remap(f[..., 1], pts[:, 0], pts[:, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            pts = pts + np.stack([fx.ravel(), fy.ravel()], 1)
        p0 = np.array([p.pt for p in kp], np.float32)
        r = 7
        inside = ((pts[:, 0] > r) & (pts[:, 0] < w - r - 1) & (pts[:, 1] > r) & (pts[:, 1] < h - r - 1)
                  & (p0[:, 0] > r) & (p0[:, 0] < w - r - 1) & (p0[:, 1] > r) & (p0[:, 1] < h - r - 1))
        if inside.sum() < 10:
            continue
        ga, gb = grays[a].astype(np.float32), grays[a + k].astype(np.float32)
        ok = 0
        for (x0, y0), (x1, y1) in zip(p0[inside], pts[inside]):
            pa = ga[int(y0) - r:int(y0) + r + 1, int(x0) - r:int(x0) + r + 1]
            pb = cv2.getRectSubPix(gb, (2 * r + 1, 2 * r + 1), (float(x1), float(y1)))
            pa, pb = pa - pa.mean(), pb - pb.mean()
            ncc = float((pa * pb).sum() / (np.sqrt((pa * pa).sum() * (pb * pb).sum()) + 1e-6))
            ok += ncc > 0.6  # derselbe Fleck ist an der vorhergesagten Stelle noch wiederzuerkennen
        leben.append(ok / int(inside.sum()))
    for i in range(len(grays) - 1):
        f = flows[i]
        back = cv2.remap(grays[i + 1], xs + f[..., 0], ys + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        diff = cv2.GaussianBlur(np.abs(back.astype(np.float32) - grays[i]), (0, 0), 2)
        bw = (diff > 40).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(bw)
        ersch.append(int((stats[1:, cv2.CC_STAT_AREA] > (h * w) * 0.0005).sum()))
    return {"punkt_leben": float(np.mean(leben)) if leben else None,
            "erscheinen": float(np.mean(ersch)) if ersch else None}


def fluss_gewinn(m: dict):
    """Wie viel der Änderung von Frame zu Frame ist Bewegung statt Neuzeichnen?
    erhalt_1 (bewegungskompensiert) minus ssim_prev (unkompensiert). Deforum-Referenz: 0.31.
    ~0 heißt: entweder steht das Bild, oder es flackert auf der Stelle."""
    if m.get("erhalt_1") is None or m.get("ssim_prev") is None:
        return None
    return float(m["erhalt_1"] - m["ssim_prev"])


def analyse_fluss(frame_paths, lags=(1, 4, 8), side: int = 320, anchor_step: int = 4) -> dict:
    """Fluss und Erhalt wie bei Deforum: bleibt ein Bildbereich bestehen und wandert mit der Bewegung?

    erhalt_k         SSIM zwischen Frame n und Frame n+k, nachdem n+k entlang der verketteten Bewegungsbahn
                     auf n zurückgetragen wurde. 1 = der Bereich ist derselbe geblieben, nur gewandert.
    erhalt_flaeche   Anteil der Bildfläche, der über den größten Lag erhalten bleibt (lokale SSIM > 0.5).
    flow_jitter      mittlere Änderung des Flussfelds von Frame zu Frame relativ zur Bewegung:
                     klein = die Bewegung fließt, groß = das Feld zappelt (Flackern täuscht Bewegung vor).
    flow_global      Anteil der Bewegungsenergie, den eine globale Kamerabewegung (Affinmodell) erklärt.
    """
    paths = list(frame_paths)
    grays = [_small(imread(p, gray=True), side) for p in paths]
    n = len(grays)
    if n < 3 or any(g.shape != grays[0].shape for g in grays):
        return {}
    h, w = grays[0].shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    flows = [farneback(grays[i], grays[i + 1]) for i in range(n - 1)]

    jit, glob = [], []
    A = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w, np.float32)], 1)
    pinv = np.linalg.pinv(A)
    for i, f in enumerate(flows):
        mag = float(np.linalg.norm(f, axis=2).mean())
        if i:
            jit.append(float(np.linalg.norm(f - flows[i - 1], axis=2).mean()) / (mag + 0.05))
        fx, fy = f[..., 0].ravel(), f[..., 1].ravel()
        res = (fx - A @ (pinv @ fx)) ** 2 + (fy - A @ (pinv @ fy)) ** 2
        tot = fx ** 2 + fy ** 2
        glob.append(1.0 - float(res.sum()) / (float(tot.sum()) + 1e-6))

    max_lag = max(lags)
    acc = {k: [] for k in lags}
    area = []
    for a in range(0, n - max_lag, anchor_step):
        mx, my = xs.copy(), ys.copy()  # Position des Punktes aus Frame a im Frame a+j
        for j in range(1, max_lag + 1):
            f = flows[a + j - 1]
            fx = cv2.remap(f[..., 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fy = cv2.remap(f[..., 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mx, my = mx + fx, my + fy
            if j in acc:
                back = cv2.remap(grays[a + j], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
                if valid.mean() < 0.2:
                    continue
                s = ssim_map(grays[a], back)
                acc[j].append(float(s[valid].mean()))
                if j == max_lag:
                    area.append(float((s[valid] > 0.5).mean()))
    out = {f"erhalt_{k}": float(np.mean(v)) if v else None for k, v in acc.items()}
    out["erhalt_flaeche"] = float(np.mean(area)) if area else None
    out["flow_jitter"] = float(np.mean(jit)) if jit else None
    out["flow_global"] = float(np.mean(glob)) if glob else None
    return out


# ---------------------------------------------------------------- Ausgabe

def contact_sheet(frame_paths, out_path, cols: int = 8, rows: int = 2, thumb_w: int = 256) -> None:
    paths = list(frame_paths)
    idx = np.unique(np.linspace(0, len(paths) - 1, cols * rows).round().astype(int))
    thumbs = []
    for i in idx:
        img = imread(paths[i])
        s = thumb_w / img.shape[1]
        t = cv2.resize(img, (thumb_w, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(i), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, str(i), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)
    th = thumbs[0].shape[0]
    while len(thumbs) % cols:
        thumbs.append(np.zeros((th, thumb_w, 3), np.uint8))
    sheet = np.vstack([np.hstack(thumbs[r:r + cols]) for r in range(0, len(thumbs), cols)])
    imwrite(out_path, sheet)


def frames_to_mp4(frames_dir, out_path, fps: float = 24) -> None:
    proc.run([proc.tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
              "-framerate", fps, "-start_number", 0, "-i", Path(frames_dir) / "%05d.png",
              "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-crf", 16,
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], "MP4-Export")   # Index vorn: spielt im Browser/Handy sofort


def finish_run(out_dir, fps: float, meta: dict, t0: float) -> dict:
    """Kontaktbogen, MP4 und run.json schreiben; gibt meta ergänzt zurück."""
    out_dir = Path(out_dir)
    frames = list_images(out_dir / "frames")
    meta = dict(meta, frames=len(frames), frames_dir=str(out_dir / "frames"),
                seconds=round(time.time() - t0, 2))
    meta["sec_per_frame"] = round(meta["seconds"] / max(1, len(frames)), 3)
    contact_sheet(frames, out_dir / "kontakt.png")
    try:
        frames_to_mp4(out_dir / "frames", out_dir / "video.mp4", fps)
    except proc.ProcError as e:
        meta["mp4_error"] = str(e).splitlines()[0]
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def clear_frames(out_dir) -> Path:
    """Leert out_dir/frames, damit keine Reste eines längeren Vorlaufs mitgezählt werden."""
    fdir = Path(out_dir) / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    for p in list_images(fdir):
        p.unlink()
    return fdir


def film_finish(img, i: int, staerke: float = 1.0, mono: bool = False, seed: int = 0):
    """Analoger Abschluss NUR fürs ausgegebene Bild (nie in die Rückkopplung): gegen den glatten KI-Look.
    Halation (helle Bereiche strahlen weich und warm aus), sanfte Schulter in den Lichtern, leichte Vignette und
    Filmkorn, das in den Mitteltönen am stärksten ist und pro Frame neu fällt (wie echtes Korn, kein Rauschmuster)."""
    if staerke <= 0:
        return img
    f = img.astype(np.float32) / 255.0
    h, w = f.shape[:2]
    lum = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    # Halation: Lichter weich ausstrahlen, bei Farbe warm (rot/orange wie bei Kinofilm ohne Lichthofschutz)
    hell = np.clip((lum - 0.7) / 0.3, 0, 1)[..., None]
    glow = cv2.GaussianBlur(f * hell, (0, 0), max(3.0, w / 120))
    tint = np.array([0.35, 0.6, 1.0], np.float32) if not mono else np.array([1.0, 1.0, 1.0], np.float32)
    f = f + 0.35 * staerke * glow * tint
    # Schulter: Lichter weich abrollen statt hart zu clippen
    f = np.where(f > 0.75, 0.75 + 0.25 * np.tanh((f - 0.75) / 0.25), f)
    # Vignette
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    f = f * (1.0 - 0.12 * staerke * r2)[..., None]
    # Korn: luminanzabhängig, leicht weichgezeichnet (Korngröße > 1 Pixel), pro Frame neu
    rng = np.random.default_rng(seed * 100003 + i)
    g = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.7)
    mitte = 4.0 * lum * (1.0 - lum)
    korn = (0.028 * staerke * (0.3 + mitte))[..., None] * g[..., None]
    if not mono:
        korn = korn + 0.005 * staerke * cv2.GaussianBlur(rng.normal(0, 1, (h, w, 3)).astype(np.float32), (0, 0), 0.7)
    f = f + korn
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)
