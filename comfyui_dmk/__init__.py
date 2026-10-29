"""ComfyUI-dmk: nodes for flow-guided, audio-reactive feedback loops (part of the dmk project).

DMK Reference Attention
    Lets SD 1.5 "see" a reference image (in dmk: the previous output, warped along the motion into the current pose)
    while it draws the current frame. The reference latent is sampled in the same batch; in every self-attention layer
    the target frame attends to its own tokens AND to the reference tokens (reference-only / StreamV2V feature-bank
    principle). Forms that already exist are recognised and continued instead of being re-invented, which reduces
    texture boiling without blurring. The reference itself is held at its latent via a zero noise mask.

    Outputs a patched MODEL and a 2-item LATENT batch [reference, target]. Decode it and keep batch index 1
    (core node "Get Image from Batch").

DMK Audio to Beats (4/4) · DMK Beat Grammar · DMK Curve to Schedule · DMK Prompts on the Bar · DMK Plot Beat Curves
    The dmk beat grammar as nodes (nodes_beat.py, analysis in dmk_beat.py): audio → beats on a 4/4 bar, placed on the
    actual transient → per-frame parameter curves → keyframe schedules and bar-synced world (prompt) changes.

License: GPL-3.0-or-later (runs inside ComfyUI, which is GPL-3.0). The rest of dmk is MIT.
"""
from __future__ import annotations

import torch

_BLOCKS = {"all": ("input", "middle", "output"), "middle+output": ("middle", "output"), "output": ("output",)}


class DMKReferenceAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "latent": ("LATENT",),
            "reference": ("LATENT",),
            "blocks": (list(_BLOCKS), {"default": "all"}),
            "sigma_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01,
                                    "tooltip": "reference is only used while sigma >= this value (0 = all steps)"}),
        }}

    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "apply"
    CATEGORY = "dmk"

    def apply(self, model, latent, reference, blocks="all", sigma_min=0.0):
        erlaubt = _BLOCKS.get(blocks, _BLOCKS["all"])
        m = model.clone()

        def patch(n, ctx, val, extra):
            blk = extra.get("block")
            if blk is not None and blk[0] not in erlaubt:
                return n, ctx, val
            sig = extra.get("sigmas")
            if sigma_min > 0 and sig is not None and float(sig.max()) < sigma_min:
                return n, ctx, val
            if n.shape[0] % 2:            # unexpected batch layout: leave untouched
                return n, ctx, val
            ctx2 = torch.cat([ctx, ctx], dim=1)
            # batch = [ref, target] per cond/uncond chunk; the target additionally attends to the reference tokens.
            # The reference attends to itself twice, which leaves its attention unchanged.
            for o in range(0, n.shape[0], 2):
                ctx2[o + 1, ctx.shape[1]:] = n[o]
            return n, ctx2, ctx2

        m.set_model_attn1_patch(patch)
        ref = reference["samples"][:1]
        tgt = latent["samples"][:1]
        if ref.shape != tgt.shape:
            ref = torch.nn.functional.interpolate(ref, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
        h, w = tgt.shape[-2:]
        mask_t = latent.get("noise_mask")
        mask_t = torch.ones((1, h, w)) if mask_t is None else torch.nn.functional.interpolate(
            mask_t.reshape(-1, 1, *mask_t.shape[-2:]).float()[:1], size=(h, w), mode="bilinear").reshape(1, h, w)
        out = {"samples": torch.cat([ref, tgt]), "noise_mask": torch.cat([torch.zeros((1, h, w)), mask_t])}
        return (m, out)


from .nodes_beat import NAMES as _BEAT_NAMES, NODES as _BEAT_NODES  # noqa: E402

NODE_CLASS_MAPPINGS = {"DMKReferenceAttention": DMKReferenceAttention, **_BEAT_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {"DMKReferenceAttention": "DMK Reference Attention", **_BEAT_NAMES}
