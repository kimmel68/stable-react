"""dmk.preflight: prüft die Umgebung, bevor ein langer Lauf startet."""
from __future__ import annotations

import argparse
import sys

from . import core, gpu, proc


def check(seq_root=None, server: str = "127.0.0.1:8188", log=print) -> bool:
    rows: list[tuple[str, str, str]] = []
    add = lambda state, name, info: rows.append((state, name, str(info)))

    add("OK", "Python", sys.version.split()[0])
    try:
        import cv2
        import numpy
        add("OK", "numpy / opencv", f"{numpy.__version__} / {cv2.__version__}")
    except ImportError as e:
        add("FEHLT", "numpy / opencv", e)

    try:
        add("OK", "ffmpeg", proc.ffmpeg_version())
        enc = proc.run([proc.tool("ffmpeg"), "-hide_banner", "-encoders"]).stdout.decode("utf-8", "replace")
        add("OK" if " mpeg4 " in enc else "FEHLT", "ffmpeg mpeg4-Encoder (Weg B)", "vorhanden" if " mpeg4 " in enc else "fehlt")
        add("OK", "ffmpeg Sync-Option", " ".join(proc.sync_args()))
    except proc.ProcError as e:
        add("FEHLT", "ffmpeg (Pflicht für Weg B)", str(e).splitlines()[0])

    add("OK" if proc.tool("ffedit", required=False) else "OPTIONAL", "ffedit/ffglitch (MV-Gain)",
        proc.tool("ffedit", required=False) or "nicht installiert, --mv-gain wird übersprungen")

    try:
        import torch
        import torchvision
        if torch.cuda.is_available():
            add("OK", "torch / CUDA (RAFT)", f"{torch.__version__}, {torch.cuda.get_device_name(0)}")
        else:
            add("WARNUNG", "torch ohne CUDA", f"{torch.__version__}: Weg C nimmt Farnebäck")
        add("OK", "torchvision", torchvision.__version__)
    except ImportError as e:
        add("WARNUNG", "torch fehlt", f"{e}: Weg C nimmt Farnebäck")

    mem = gpu.memory_mb()
    if mem:
        add("OK" if mem[0] >= 3000 else "WARNUNG", "VRAM frei / gesamt", f"{mem[0]} / {mem[1]} MiB")

    from .method_a import Comfy
    comfy = Comfy(server)
    if comfy.alive():
        try:
            ck = comfy.checkpoints()
            ver = comfy.stats().get("system", {}).get("comfyui_version", "?")
            add("OK" if ck else "WARNUNG", f"ComfyUI {server}", f"v{ver}, {len(ck)} Checkpoints")
        except Exception as e:
            add("WARNUNG", f"ComfyUI {server}", f"erreichbar, aber {type(e).__name__}: {e}")
    else:
        add("WARNUNG", f"ComfyUI {server}", "nicht erreichbar: Weg A nur mit --dry-run (start_comfy.bat)")

    if seq_root:
        try:
            seq = core.load_sequence(seq_root)
            h, w = core.imread(seq.frames[0]).shape[:2]
            add("OK", "Sequenz", f"{len(seq)} Frames {w}x{h}, Tiefe: {'ja' if seq.depth else 'nein'}  ({seq.root})")
        except Exception as e:
            add("FEHLT", "Sequenz", e)

    for state, name, info in rows:
        log(f"  [{state:^8}] {name:34} {info}")
    return not any(s == "FEHLT" for s, _, _ in rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dmk.preflight", description=__doc__)
    ap.add_argument("--seq-root", default=None)
    ap.add_argument("--server", default="127.0.0.1:8188")
    a = ap.parse_args(argv)
    return 0 if check(a.seq_root, a.server) else 1


if __name__ == "__main__":
    raise SystemExit(main())
