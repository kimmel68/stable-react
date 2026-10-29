"""Test der Beat-Nodes ohne ComfyUI: gleiche Beats wie die App-Analyse, Grammatik, Schedules, Weltwechsel, Plot."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deforum_mosh"))
import comfyui_dmk as N  # noqa: E402
from dmk import audio as A  # noqa: E402

track = ROOT / "examples" / "signal_decay_20s.m4a"
y = A.dekodieren(track, 44100)
au = {"waveform": torch.from_numpy(np.stack([y, y]))[None], "sample_rate": 44100}
b, info = N.NODE_CLASS_MAPPINGS["DMKAudioBeats"]().run(au, 30.0, 360, 2.0, False, 60, 140)
print(info)
_, info2 = N.NODE_CLASS_MAPPINGS["DMKAudioBeats"]().run(au, 30.0, 360, 2.0, True, 60, 140)
print(info2)
c, s = N.NODE_CLASS_MAPPINGS["DMKBeatGrammar"]().run(b, N.nodes_beat._b.DEFAULT_GRAMMAR)
print(s)
sched, v = N.NODE_CLASS_MAPPINGS["DMKCurveSchedule"]().run(c, "zoom", 50, 4)
print(sched[:80].replace("\n", " "), "...", v)
print(N.NODE_CLASS_MAPPINGS["DMKBarPrompts"]().run(b, "a\nb\nc", 2, 200))
img = N.NODE_CLASS_MAPPINGS["DMKPlotCurves"]().run(b, c)[0]
import cv2  # noqa: E402
cv2.imwrite(str(ROOT / "out" / "_node_plot.png"), (img[0].numpy()[..., ::-1] * 255).astype("uint8"))
mm, inf = A.merkmale(A.dekodieren(track, 22050), 22050, 360, 1 / 30, 2.0, log=lambda s: None)
gleich = inf["beat_frames"] == b["beat_frames"]
print("app beats ", inf["beat_frames"][:10])
print("node beats", b["beat_frames"][:10], "identical:", gleich)
sys.exit(0 if gleich else 1)
