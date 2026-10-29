"""DMK beat grammar nodes: audio → beats on a 4/4 bar (on the actual transient) → per-frame parameter curves →
keyframe schedules and bar-synced prompt (world) changes for any frame-by-frame workflow. Analysis in dmk_beat.py."""
from __future__ import annotations

import torch

from . import dmk_beat as _b


class DMKAudioBeats:
    """Audio → beats on a 4/4 bar, per video frame: features (onset, bass, envelope, ...), beat and counts 1–4 as
    one-frame impulses placed on the actual transient (never before the sound), bar/beat phase ramps."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.001}),
            "frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 = as long as the audio"}),
            "offset_s": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
            "beat_grid": ("BOOLEAN", {"default": False, "tooltip": "fit a fixed-BPM grid (for tightly produced music)"}),
            "bpm_min": ("FLOAT", {"default": 60.0, "min": 30.0, "max": 300.0}),
            "bpm_max": ("FLOAT", {"default": 140.0, "min": 30.0, "max": 300.0}),
        }}

    RETURN_TYPES = ("DMK_BEATS", "STRING")
    RETURN_NAMES = ("beats", "info")
    FUNCTION = "run"
    CATEGORY = "dmk/beat"

    def run(self, audio, fps, frames, offset_s, beat_grid, bpm_min, bpm_max):
        b = _b.analyse(audio, fps, frames, offset_s, beat_grid, bpm_min, bpm_max)
        g = b["grid"]
        info = (f"{b['bpm']:.1f} bpm, {len(b['beat_frames'])} beats and {len(b['bar_frames'])} bars in {b['frames']} frames"
                + (f"; grid {g['bpm']:.3f} bpm, onsets scatter {g['scatter_ms']:.1f} ms around it" if g else ""))
        return (b, info)


class DMKBeatGrammar:
    """Beats → per-frame parameter curves from a grammar (JSON: parameter → binding or list of bindings that add up)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"beats": ("DMK_BEATS",),
                             "grammar": ("STRING", {"multiline": True, "default": _b.DEFAULT_GRAMMAR})}}

    RETURN_TYPES = ("DMK_CURVES", "STRING")
    RETURN_NAMES = ("curves", "summary")
    FUNCTION = "run"
    CATEGORY = "dmk/beat"

    def run(self, beats, grammar):
        c = _b.grammar(beats, grammar)
        return (c, "\n".join(f"{k}: {v.min():.4g} .. {v.max():.4g} (mean {v.mean():.4g})" for k, v in c.items()))


class DMKCurveSchedule:
    """One curve → keyframe schedule "0:(v), 1:(v), ..." (Deforum / FizzNodes / value-schedule style) and the value at
    a given frame (for per-frame loops)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"curves": ("DMK_CURVES",), "parameter": ("STRING", {"default": "denoise"}),
                             "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                             "decimals": ("INT", {"default": 4, "min": 0, "max": 8})}}

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("schedule", "value_at_frame")
    FUNCTION = "run"
    CATEGORY = "dmk/beat"

    def run(self, curves, parameter, frame, decimals):
        if parameter not in curves:
            raise ValueError(f"no curve {parameter!r}; available: {', '.join(curves)}")
        v = curves[parameter]
        keep = [0] + [i for i in range(1, len(v)) if round(float(v[i]), decimals) != round(float(v[i - 1]), decimals)]
        return (",\n".join(f"{i}:({v[i]:.{decimals}f})" for i in keep), float(v[min(frame, len(v) - 1)]))


class DMKBarPrompts:
    """World change on the bar: prompts (one per line) switch on every N-th count 1. Outputs a prompt schedule
    ("frame": "prompt" lines as used by FizzNodes/Deforum-style prompt schedulers) and the prompt at a given frame."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"beats": ("DMK_BEATS",),
                             "prompts": ("STRING", {"multiline": True, "default": "brutalist concrete cathedral\n"
                                                    "copperplate engraving of a basilica\ncathedral of fungal mycelium"}),
                             "every_bars": ([1, 2, 4], {"default": 2}),
                             "frame": ("INT", {"default": 0, "min": 0, "max": 100000})}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("schedule", "prompt_at_frame", "change_frames")
    FUNCTION = "run"
    CATEGORY = "dmk/beat"

    def run(self, beats, prompts, every_bars, frame):
        ps = [p.strip() for p in prompts.splitlines() if p.strip()] or [""]
        jumps, world = _b.bar_prompts(beats, ps, int(every_bars))
        sched = ",\n".join(f'"{f}": "{ps[world[f]]}"' for f in [0] + jumps if f < len(world))
        return (sched, ps[world[min(frame, len(world) - 1)]], ", ".join(map(str, jumps)))


class DMKPlotCurves:
    """Preview: curves (or the main audio features) over time, beat markers coloured by count (1 red … 4 blue)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"beats": ("DMK_BEATS",)}, "optional": {"curves": ("DMK_CURVES",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "dmk/beat"

    def run(self, beats, curves=None):
        img = _b.plot(beats, curves)
        return (torch.from_numpy(img.astype("float32") / 255.0)[None],)


NODES = {"DMKAudioBeats": DMKAudioBeats, "DMKBeatGrammar": DMKBeatGrammar, "DMKCurveSchedule": DMKCurveSchedule,
         "DMKBarPrompts": DMKBarPrompts, "DMKPlotCurves": DMKPlotCurves}
NAMES = {"DMKAudioBeats": "DMK Audio to Beats (4/4)", "DMKBeatGrammar": "DMK Beat Grammar",
         "DMKCurveSchedule": "DMK Curve to Schedule", "DMKBarPrompts": "DMK Prompts on the Bar",
         "DMKPlotCurves": "DMK Plot Beat Curves"}
