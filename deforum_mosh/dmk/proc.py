"""dmk.proc: geprüfte Subprozess-Schicht (ffmpeg, ffedit, nvidia-smi)."""
from __future__ import annotations

import functools
import shutil
import subprocess


class ProcError(RuntimeError):
    pass


def tool(name: str, required: bool = True) -> str | None:
    path = shutil.which(name)
    if path is None and required:
        raise ProcError(f"'{name}' nicht im PATH gefunden.")
    return path


def run(cmd, what: str = "", timeout: float | None = None, input_bytes: bytes | None = None):
    """Führt cmd aus und wirft ProcError mit dem Ende von stderr, wenn der Exit-Code nicht 0 ist."""
    cmd = [str(c) for c in cmd]
    r = subprocess.run(cmd, capture_output=True, timeout=timeout, input=input_bytes)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace")[-1500:]
        raise ProcError(f"{what or cmd[0]} fehlgeschlagen (Exit {r.returncode}):\n  {' '.join(cmd)}\n{err}")
    return r


@functools.lru_cache(maxsize=1)
def ffmpeg_version() -> str:
    r = run([tool("ffmpeg"), "-hide_banner", "-version"], "ffmpeg -version")
    return r.stdout.decode("utf-8", "replace").splitlines()[0]


@functools.lru_cache(maxsize=1)
def sync_args() -> tuple[str, str]:
    """-vsync ist in neueren ffmpeg-Versionen abgekündigt; -fps_mode nehmen, wenn vorhanden."""
    r = run([tool("ffmpeg"), "-hide_banner", "-h", "full"], "ffmpeg -h full")
    if b"-fps_mode" in r.stdout:
        return ("-fps_mode", "passthrough")
    return ("-vsync", "0")
