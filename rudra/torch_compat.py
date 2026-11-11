"""Torch behaviour pinned across versions (31 Oct 2026).

torch 2.15 made ``clamp``'s gradient zero AT its bounds (it was one before).
Losses that start from an exactly black prediction then get no gradient at all,
so training could never lift it. ``clamp_inclusive`` keeps the old rule: the
value is ``clamp``'s, the gradient is one inside the closed interval and zero
outside it, on every torch version.
"""

from __future__ import annotations

import os
import pickle

import torch


def load_checkpoint(path, map_location="cpu"):
    """``torch.load`` for a file that may not be ours (security review, 11 Nov 2026).

    A ``.pt`` file is a pickle, and unpickling one with ``weights_only=False``
    runs whatever code the file names: a checkpoint from a download or a
    colleague was code execution. This loads tensors and plain values only
    (dicts, lists, numbers, strings), which is all a RUDRA checkpoint holds:
    every committed one in ``checkpoints/`` references nothing else.

    A file that holds more is refused with a message that says so. For a file
    you made yourself and trust (an old training checkpoint with objects in
    it), set ``RUDRA_ALLOW_PICKLE=1`` and run again.
    """
    if os.environ.get("RUDRA_ALLOW_PICKLE") == "1":
        return torch.load(path, map_location=map_location, weights_only=False)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise RuntimeError(
            f"{path} holds more than tensors and plain values, so loading it would run code "
            "from the file. If you made this file yourself and trust it, set "
            "RUDRA_ALLOW_PICKLE=1 and run again."
        ) from exc


def clamp_inclusive(x: torch.Tensor, lo=None, hi=None) -> torch.Tensor:
    inside = torch.ones_like(x, dtype=torch.bool)
    if lo is not None:
        inside = inside & (x >= lo)
    if hi is not None:
        inside = inside & (x <= hi)
    return torch.where(inside, x, x.detach().clamp(min=lo, max=hi))
