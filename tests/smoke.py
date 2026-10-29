"""Smoke tests for dmk – no GPU, no ComfyUI needed (runs in about a minute).

    .venv\\Scripts\\python tests\\smoke.py

Checks: audio analysis on the example track, the portable recipe, workflow building (incl. reference attention),
both app modes, BPM-only mode, the Flux Hybrid schedule, and a dry run of the full loop on an example image.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deforum_mosh"))
sys.path.insert(0, str(ROOT / "app"))

E = ROOT / "examples"
ERG: list[tuple[str, bool, str]] = []


def test(name):
    def deko(fn):
        t0 = time.time()
        try:
            info = fn() or ""
            ERG.append((name, True, f"{info} ({time.time() - t0:.1f} s)"))
        except Exception as e:  # noqa: BLE001
            ERG.append((name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))
        return fn
    return deko


@test("audio analysis: tempo and bar grid on the example track")
def _():
    from dmk import audio
    spec = json.loads((ROOT / "configs" / "recipe_bar44.json").read_text(encoding="utf-8"))["audio"]
    kurven, force, mm, info = audio.aufloesen(spec, 240, 1, 0, None, log=lambda s: None)
    assert 80 <= info["bpm"] <= 92, f"tempo {info['bpm']} (expected about 86)"
    assert len(info["takt_frames"]) >= 2 and force is not None and force.sum() >= 8
    for k in ("quellzoom", "weich", "denoise", "glaettung"):
        assert k in kurven and len(kurven[k]) == 240, k
    return f"{info['bpm']} bpm, {int(force.sum())} beat samples"


@test("workflow: depth ControlNet + reference attention graph")
def _():
    from dmk import method_a
    wf = json.loads(Path(method_a.DEPTH_WORKFLOW).read_text(encoding="utf-8"))
    w = method_a.add_referenz(wf)
    ks = next(v for v in w.values() if v["class_type"] == "KSampler")
    assert ks["inputs"]["model"] == ["dmk_ref_attn", 0] and ks["inputs"]["latent_image"] == ["dmk_ref_attn", 1]
    prev = next(v for v in w.values() if v["class_type"] == "PreviewImage")
    assert prev["inputs"]["images"] == ["dmk_ref_pick", 0]
    p = method_a.patch_workflow(w, "init.png", "a hall", "blurry", "x.safetensors", 1, 8, 7.0, 0.4, None, None, "c.png",
                                "cn.safetensors", 1.0, 1.0)
    assert any(v.get("inputs", {}).get("image") == "init.png" for v in p.values())
    return f"{len(w)} nodes"


@test("app: parameters for both modes, track and BPM only")
def _():
    import dmk_app as A
    out = []
    for m in A.MODI:
        with tempfile.TemporaryDirectory() as d:
            p, q, n, t = A.lauf_params("Image", str(E / "hall.png"), None, "", str(E / "signal_decay_20s.m4a"),
                                       A.STANDARD_REZEPT, "beton", "", 2, 1.0, 0.5, True, 512, None, Path(d), True, m)
        assert n == 60 and p["motion"] == "virtuell" and p["weich_modus"] == "anzeige"
        assert "glaettung" in p["audio"]["bindungen"] and "seed" not in p["audio"]
        assert (m == "Rigid frame") == ("quellzoom" not in p["audio"]["bindungen"])
        out.append(m)
    from dmk import audio
    with tempfile.TemporaryDirectory() as d:
        p, q, n, t = A.lauf_params("Image", str(E / "hall.png"), None, "", None, A.STANDARD_REZEPT, "beton", "", 6, 1.0,
                                   0.5, False, 512, None, Path(d), wechsel="Every bar", eigene_welten="a\nb",
                                   beat_quelle="BPM only", bpm=150)
        _, _, _, info = audio.aufloesen(p["audio"], n, 1, 0, None, log=lambda s: None)
        assert info["welt_wechsel"][:2] == [15, 63], info["welt_wechsel"]   # Kick bei 0,5 s, Takt 1,6 s
    return ", ".join(out) + ", BPM 150"


@test("flux hybrid: shared world schedule and dry run of the Flux stage")
def _():
    import dmk_app as A
    from dmk import hybrid
    with tempfile.TemporaryDirectory() as d:
        p, q, n, t = A.lauf_params("Image", str(E / "hall.png"), None, "", None, A.STANDARD_REZEPT, "beton", "", 4, 1.0,
                                   0.5, False, 512, None, Path(d), wechsel="Every bar", eigene_welten="a\nb\nc",
                                   beat_quelle="BPM only", bpm=120, engine="flux")
        pl = hybrid.plan(p, n, "Every beat")
        assert set(pl["wechsel"]) <= set(pl["keys"]) and pl["welt"][pl["wechsel"][0]] == 1
        fr = hybrid.flux_stufe(q, Path(d) / "flux", n, pl, A.FOTO, log=lambda s: None, dry=True)
        assert len(list(Path(fr).glob("*.png"))) == n
        p2 = hybrid.sd_stufe(p)
        assert hybrid.sd_stufe(p, q)["motion"] == "virtuell"   # Bild: virtuelle Kamera bleibt
        assert p2["motion"] == "kamera" and any(x.get("quelle") == "wechsel" for x in p2["audio"]["bindungen"]["alpha"])
    return f"{len(pl['keys'])} keyframes, {n} frames"


@test("loop: dry run on an example image (virtual camera, 4/4 recipe)")
def _():
    from dmk import method_a
    spec = json.loads((ROOT / "configs" / "recipe_bar44.json").read_text(encoding="utf-8"))["audio"]
    with tempfile.TemporaryDirectory() as d:
        meta = method_a.run(out_dir=d, count=24, start=0, fps=30, seq_root=str(E / "cloister.png"), motion="virtuell",
                            controlnet="none", mono=True, audio=spec, dry_run=True, log=lambda s: None,
                            weich_modus="anzeige", schaerfe_halten=0.8)
        assert meta["frames"] == 24
        assert (Path(d) / "kontakt.png").exists() and (Path(d) / "audio_kurven.png").exists()
    return "24 frames"


@test("sync tools: MP4 index at the start (faststart)")
def _():
    from dmk import core
    import cv2
    import numpy as np
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "frames"
        f.mkdir()
        for i in range(6):
            core.imwrite(f / f"{i:05d}.png", np.full((64, 64, 3), i * 40, np.uint8))
        core.frames_to_mp4(f, Path(d) / "v.mp4", 30)
        kopf = (Path(d) / "v.mp4").read_bytes()
        assert kopf.find(b"moov") < kopf.find(b"mdat"), "moov after mdat"
    return "ok"


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    ok = all(r[1] for r in ERG)
    for name, gut, info in ERG:
        print(f"[{'PASS' if gut else 'FAIL'}] {name}: {info}")
    print(f"\n{sum(r[1] for r in ERG)}/{len(ERG)} passed")
    raise SystemExit(0 if ok else 1)
