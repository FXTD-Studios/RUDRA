"""Opt-in preservation diagnostics, outside the optimization graph."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def validate_args(args):
    domain = getattr(args, "crushed_hold_domain", "postclamp")
    if domain not in ("postclamp", "preclamp"):
        raise ValueError("unknown crushed-hold domain")
    if domain == "preclamp" or getattr(args, "preservation_diagnostics", False):
        if args.mode != "image":
            raise ValueError("pre-clamp hold and diagnostics require image training")
    if domain == "preclamp" and (not math.isfinite(args.crushed_hold) or args.crushed_hold <= 0):
        raise ValueError("pre-clamp hold requires a positive finite crushed_hold")


@torch.no_grad()
def hold_diagnostics(output, sdr):
    delta = output.preclamp_log_delta
    if delta is None or output.preclamp_log_scale != 16.0:
        raise ValueError("preservation diagnostics require forward capture at log_scale=16")
    top = sdr.clamp(0, 1).amax(1, keepdim=True)
    held = F.max_pool2d((top <= 2 / 255).to(delta.dtype), 3, 1, 1)
    count = held.sum() * 3
    base_log = torch.log1p(output.baseline.clamp_min(0) * 16)
    negative = ((base_log + delta) < 0) & held.bool()
    post = (torch.log1p(output.hdr.clamp_min(0) * 16) - base_log).abs()
    return {"held_channels": count,
            "negative_held_channels": negative.sum(),
            # Zero count is separately logged, not interpreted as coverage.
            "negative_held_fraction": negative.sum() / count.clamp_min(1),
            "hold_preclamp": (delta.abs() * held).sum() / (count + 1e-6),
            "hold_postclamp": (post * held).sum() / (count + 1e-6)}
