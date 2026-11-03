"""The highlight ceiling, as the native engine applies it (3 Nov 2026).

Near-white tinted display codes (253,255,255; 255,255,253) invert through the
ACES 2.0 SDR view far past pure white (09V: 296x and 969x against 128x), and the
view sends such light back to white, so the values are neither displayable as
the source nor plausible. native/core/src/aces_v2.cpp run_frame caps the
baseline, the restored source and the master at the AP1 luminance of the
inverse of display white, hue kept. This module is the same arithmetic for the
Python pipelines (tools/export_native_v2.py staged_reference mirrors the native
stages exactly; tools/rudra_aces_export.py caps the baseline and the master).
"""
from __future__ import annotations

import numpy as np

LUMA_AP1 = (.2722287168, .6740817658, .0536895174)


def inverse_white_luma() -> float:
    """AP1 luminance of the reference view's inverse of display white (1, 1, 1)."""
    import torch
    from training.aces_reference import inverse_aces2_reference
    w = inverse_aces2_reference(torch.ones(1, 3, 1, 1))[0, :, 0, 0].double().numpy()
    return float(np.dot(w, LUMA_AP1))


def cap_luminance_np(rgb: np.ndarray, luma_max: float):
    """HxWx3 (or Nx3) -> (capped copy, pixels capped). Brighter than luma_max: scaled down to it."""
    rgb = np.asarray(rgb)
    y = rgb.astype(np.float64) @ np.array(LUMA_AP1)
    over = y > luma_max
    out = rgb.copy()
    if over.any():
        out[over] = (rgb[over].astype(np.float64) * (luma_max / y[over])[:, None]).astype(rgb.dtype)
    return out, int(over.sum())


def cap_luminance_torch(x, luma_max: float):
    """BCHW tensor -> (capped tensor, pixels capped)."""
    import torch
    w = torch.tensor(LUMA_AP1, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    y = (x * w).sum(1, keepdim=True)
    over = y > luma_max
    k = torch.where(over, luma_max / y.clamp_min(1e-30), torch.ones_like(y))
    return x * k, int(over.sum())
