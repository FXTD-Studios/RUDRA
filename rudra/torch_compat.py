"""Torch behaviour pinned across versions (31 Oct 2026).

torch 2.15 made ``clamp``'s gradient zero AT its bounds (it was one before).
Losses that start from an exactly black prediction then get no gradient at all,
so training could never lift it. ``clamp_inclusive`` keeps the old rule: the
value is ``clamp``'s, the gradient is one inside the closed interval and zero
outside it, on every torch version.
"""

from __future__ import annotations

import torch


def clamp_inclusive(x: torch.Tensor, lo=None, hi=None) -> torch.Tensor:
    inside = torch.ones_like(x, dtype=torch.bool)
    if lo is not None:
        inside = inside & (x >= lo)
    if hi is not None:
        inside = inside & (x <= hi)
    return torch.where(inside, x, x.detach().clamp(min=lo, max=hi))
