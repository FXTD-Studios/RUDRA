"""Full-only routing of the learned ACES 2.0 reconstruction (registered arm, 1 Nov 2026).

The learned model's 0 EV grading failure is entirely in PARTIALLY clipped
highlights (CURRENT_HANDOFF 24-25 Oct: partial .186 vs analytic .086 code^2;
fully clipped already beats the analytic inverse). This arm keeps the learned
result only where the source is FULLY clipped (all three codes >= 254.5/255,
the evaluators' definition) and the exact analytic inverse everywhere else,
so partially clipped pixels score exactly the analytic's error by
construction.

An optional feather ramps the learned weight from 0 at the edge of the fully
clipped area to 1 at `feather_px` pixels inside it, so no seam is drawn at the
boundary; the ramp lies entirely INSIDE the fully clipped area, so every
pixel outside it stays bit-exact analytic. Apply after smooth_clipped and
before constrain_colour, which still guarantees the 1-code round trip.
"""
from __future__ import annotations

import numpy as np

FULL_CODE = 254.5 / 255.0


def fully_clipped(source: np.ndarray) -> np.ndarray:
    """HxW bool: all three source codes at 255 (the evaluators' `full`)."""
    return np.asarray(source).min(axis=-1) >= FULL_CODE


def full_only_weight(source: np.ndarray, feather_px: float = 0.0) -> np.ndarray:
    """HxW float32 in [0, 1]: 1 inside the fully clipped area (ramped over
    `feather_px` pixels from its edge), exactly 0 outside it."""
    full = fully_clipped(source)
    if feather_px <= 0:
        return full.astype(np.float32)
    from scipy.ndimage import distance_transform_edt
    # Distance from each fully clipped pixel to the nearest pixel that is not;
    # 1 for a pixel touching the edge.
    d = distance_transform_edt(full)
    return np.where(full, np.clip(d / float(feather_px), 0.0, 1.0), 0.0).astype(np.float32)


def route_full_only(prediction: np.ndarray, analytic: np.ndarray, source: np.ndarray,
                    feather_px: float = 0.0) -> np.ndarray:
    """The learned prediction inside the fully clipped area, the analytic
    inverse outside it (bit-exact), blended linearly across the feather."""
    prediction = np.asarray(prediction, dtype=np.float32)
    analytic = np.asarray(analytic, dtype=np.float32)
    if prediction.shape != analytic.shape or prediction.shape[:2] != np.asarray(source).shape[:2]:
        raise ValueError('matching prediction, analytic and source required')
    w = full_only_weight(source, feather_px)[..., None]
    out = analytic.copy()
    inside = w[..., 0] > 0
    out[inside] = (analytic * (1.0 - w) + prediction * w)[inside]
    return out
