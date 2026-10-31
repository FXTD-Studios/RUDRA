"""Scene-linear ACEScg <-> the Wan VAE's pixel range.

The VAE was trained on display images in [-1, 1]; scene-linear values span ~30 stops. We use
ACEScct (Academy S-2016-001: a log2 curve with a linear toe, so 0 and small values stay
invertible) normalised so that 0 -> -1 and M -> +1. With M = 2**14 (16 384, about 16.5 stops
over 18 % grey) grey sits near -0.47 and the SDR view's white (scene ~16-40) near -0.1 .. +0.05,
leaving the upper half of the range for the highlights the fill invents.

Values are clamped to [0, M]; the composite keeps the master everywhere the fill is not used,
so negatives (out-of-AP1 colours) in the master are never touched by this encoding.
Works on numpy arrays and torch tensors alike.
"""
from __future__ import annotations

import math

M_DEFAULT = 2.0 ** 14

_A, _B = 10.5402377416545, 0.0729055341958355     # ACEScct linear toe
_X_BREAK = 0.0078125                                 # 2**-7
_C_BREAK = 0.155251141552511


def _cct_scalar(x: float) -> float:
    return _A * x + _B if x <= _X_BREAK else (math.log2(x) + 9.72) / 17.52


def _lib(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return torch
    except ImportError:  # pragma: no cover
        pass
    import numpy
    return numpy


def acescct(x):
    """ACEScct of scene-linear x >= 0."""
    L = _lib(x)
    lin = _A * x + _B
    lg = (L.log2(L.clip(x, _X_BREAK, None)) + 9.72) / 17.52
    return L.where(x <= _X_BREAK, lin, lg)


def acescct_inverse(c):
    L = _lib(c)
    lin = (c - _B) / _A
    lg = 2.0 ** (L.clip(c, _C_BREAK, None) * 17.52 - 9.72)
    return L.where(c <= _C_BREAK, lin, lg)


def encode(x, M: float = M_DEFAULT):
    """Scene-linear ACEScg (any shape) -> [-1, 1]."""
    L = _lib(x)
    lo, hi = _cct_scalar(0.0), _cct_scalar(M)
    v = (acescct(L.clip(x, 0.0, M)) - lo) / (hi - lo)
    return v * 2.0 - 1.0


def decode(y, M: float = M_DEFAULT):
    """[-1, 1] -> scene-linear ACEScg in [0, M]."""
    L = _lib(y)
    lo, hi = _cct_scalar(0.0), _cct_scalar(M)
    v = L.clip((y + 1.0) * 0.5, 0.0, 1.0)
    return L.clip(acescct_inverse(v * (hi - lo) + lo), 0.0, M)
