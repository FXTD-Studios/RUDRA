"""Torch behaviour pinned across versions (31 Oct 2026).

torch 2.15 made ``clamp``'s gradient zero AT its bounds (it was one before).
Losses that start from an exactly black prediction then get no gradient at all,
so training could never lift it. ``clamp_inclusive`` keeps the old rule: the
value is ``clamp``'s, the gradient is one inside the closed interval and zero
outside it, on every torch version.
"""

from __future__ import annotations

import pickle

import torch


def load_checkpoint(path, map_location="cpu"):
    """``torch.load`` for a file that may not be ours (security review, 11 Nov 2026).

    A ``.pt`` file is a pickle, and unpickling one with ``weights_only=False``
    runs whatever code the file names: a checkpoint from a download or a
    colleague was code execution. This loads tensors and plain values only
    (dicts, lists, numbers, strings), which is all a RUDRA checkpoint holds:
    every committed one in ``checkpoints/`` references nothing else.

    A file that holds more is refused with a message that says so. There is no
    way round it (13 Nov 2026: the ``RUDRA_ALLOW_PICKLE`` opt-in is gone, so no
    setting makes RUDRA run code from a file). An old training checkpoint with
    objects in it is re-saved once as tensors and plain values by the code that
    made it.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise RuntimeError(
            f"{path} holds more than tensors and plain values, so loading it would run code "
            "from the file, and RUDRA does not. If you made this file, re-save it with only "
            "tensors and plain values (a state_dict) and load that."
        ) from exc


def clamp_inclusive(x: torch.Tensor, lo=None, hi=None) -> torch.Tensor:
    inside = torch.ones_like(x, dtype=torch.bool)
    if lo is not None:
        inside = inside & (x >= lo)
    if hi is not None:
        inside = inside & (x <= hi)
    return torch.where(inside, x, x.detach().clamp(min=lo, max=hi))
