"""dmk.fluss: Fluss- und Erhalt-Kennzahlen für beliebige Laufordner als Tabelle (CPU, kein Rendern).

    python -m dmk.fluss seq\\ref_bond out\\bench_flow_kreuzgang\\* out\\ab_flow\\iter_hf0_gen0
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

from . import core

LAGS = (1, 4, 8, 16, 32)
COLS = ("fluss_gewinn", "erhalt_1", "erhalt_4", "erhalt_8", "erhalt_16", "erhalt_32", "erhalt_flaeche",
        "flow_jitter", "flow_global")


def frames_of(path: Path) -> list[Path]:
    return core.list_images(path / "frames") if (path / "frames").is_dir() else core.list_images(path)


def table(paths, max_frames: int = 120) -> list[tuple[str, dict]]:
    rows = []
    for p in paths:
        fs = frames_of(Path(p))
        if len(fs) < 12:
            continue
        fs = fs[:max_frames]
        m = core.analyse_fluss(fs, tuple(k for k in LAGS if k < len(fs) - 4))
        base = core.analyse(fs)
        m.update(flow_px=base["flow_px"], ssim_prev=base["ssim_prev"], jerk=base["jerk"])
        m["fluss_gewinn"] = core.fluss_gewinn(m)
        rows.append((f"{Path(p).parent.name}/{Path(p).name}", m))
    return rows


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    args = argv if argv is not None else sys.argv[1:]
    paths = [q for a in args for q in (sorted(glob.glob(a)) or [a]) if Path(q).is_dir()]
    cols = COLS + ("flow_px", "ssim_prev", "jerk")
    print("| Lauf | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, m in table(paths):
        print(f"| {name} | " + " | ".join("–" if m.get(c) is None else f"{m[c]:.3f}" for c in cols) + " |", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
