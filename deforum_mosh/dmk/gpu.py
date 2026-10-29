"""dmk.gpu: auf freien VRAM warten (nvidia-smi), optional bis die ComfyUI-Queue leer ist."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

from . import proc


def ram_frei_gb() -> float:
    """Verfügbarer Arbeitsspeicher in GB (inkl. freigebbarem Cache)."""
    import psutil
    return psutil.virtual_memory().available / 1024 ** 3


def ram_waechter(min_gb: float = 4.0, ziel_gb: float = 6.0, max_warten: float = 600, log=print) -> None:
    """Schützt vor dem Einfrieren: ist weniger als min_gb RAM frei, wird gewartet, bis wieder ziel_gb frei sind.
    Bleibt es nach max_warten Sekunden knapp, bricht der Lauf mit MemoryError ab (statt Windows auslagern zu lassen)."""
    frei = ram_frei_gb()
    if frei >= min_gb:
        return
    log(f"[RAM] only {frei:.1f} GB free (< {min_gb} GB), pausing until {ziel_gb} GB are free")
    t0 = time.time()
    import gc
    gc.collect()
    while time.time() - t0 < max_warten:
        time.sleep(5)
        frei = ram_frei_gb()
        if frei >= ziel_gb:
            log(f"[RAM] {frei:.1f} GB free again, continuing")
            return
    raise MemoryError(f"Only {frei:.1f} GB RAM free after waiting {max_warten:.0f} s – render stopped so the PC does not "
                      f"freeze. Close other programs and start again.")


def memory_mb() -> tuple[int, int] | None:
    """(frei, gesamt) in MiB der ersten GPU, None ohne nvidia-smi."""
    smi = proc.tool("nvidia-smi", required=False)
    if smi is None:
        return None
    r = proc.run([smi, "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"], "nvidia-smi")
    free, total = r.stdout.decode().splitlines()[0].split(",")
    return int(free), int(total)


def comfy_idle(server: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://{server}/queue", timeout=3) as r:
            q = json.loads(r.read())
        return not q.get("queue_running") and not q.get("queue_pending")
    except OSError:
        return True  # nicht erreichbar = belegt die GPU nicht über die Queue


def wait_free(min_free_mb: int = 4000, stable: int = 3, interval: float = 5.0, timeout: float = 3600,
              server: str | None = None, log=print) -> bool:
    """Wartet, bis `stable` Messungen in Folge genug freien VRAM zeigen."""
    t0, ok = time.time(), 0
    while True:
        mem = memory_mb()
        if mem is None:
            log("[gpu] nvidia-smi fehlt, warte nicht.")
            return True
        good = mem[0] >= min_free_mb and (server is None or comfy_idle(server))
        ok = ok + 1 if good else 0
        log(f"[gpu] frei {mem[0]} / {mem[1]} MiB  ({ok}/{stable})")
        if ok >= stable:
            return True
        if time.time() - t0 > timeout:
            return False
        time.sleep(interval)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dmk.gpu", description=__doc__)
    ap.add_argument("--min-free", type=int, default=4000, help="benötigter freier VRAM in MiB")
    ap.add_argument("--stable", type=int, default=3)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--server", default=None, help="ComfyUI host:port, dessen Queue leer sein muss")
    a = ap.parse_args(argv)
    return 0 if wait_free(a.min_free, a.stable, a.interval, a.timeout, a.server) else 1


if __name__ == "__main__":
    raise SystemExit(main())
