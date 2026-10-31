"""Dark-region guard: keep the analytic baseline where the SDR carried no shadow detail.

Measured failure (22 Oct 2026, reserved test, Mataha A_0027C021): below ~0.1
nits the network lifts crushed, codec-noisy blacks into saturated coloured
blobs (up to +10 stops over the baseline) and shows 512-px tile seams. Between
0.1 and 100 nits it matched the baseline. The guard blends model -> baseline
by the BASELINE luminance, which is a monotone function of the SDR code, so it
is a pure function of the input and the two predictions:

    w   = smoothstep(log2 Y_base; log2 lo, log2 hi)      0 below lo, 1 above hi
    w   = box-blurred over `feather_px` (no hard edges, no new seams)
    out = base + w * (model - base)

It never changes pixels whose baseline is above `hi`, so highlight expansion is
untouched. Pure numpy; the same function serves offline evaluation and, after
selection, the inference paths.
"""
from __future__ import annotations

import numpy as np

LUMA2020 = np.array([0.2627, 0.6780, 0.0593])


def guard_weight(base_nits: np.ndarray, lo: float, hi: float, feather_px: int = 0) -> np.ndarray:
    if not (0 < lo < hi):
        raise ValueError("need 0 < lo < hi (nits)")
    y = np.maximum(np.asarray(base_nits, np.float64)[..., :3] @ LUMA2020, 1e-9)
    t = np.clip((np.log2(y) - np.log2(lo)) / (np.log2(hi) - np.log2(lo)), 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    if feather_px > 0:
        import cv2
        k = 2 * int(feather_px) + 1
        w = cv2.blur(w.astype(np.float32), (k, k), borderType=cv2.BORDER_REFLECT).astype(np.float64)
    return w


def apply_dark_guard(model_nits: np.ndarray, base_nits: np.ndarray, lo: float, hi: float,
                     feather_px: int = 0) -> np.ndarray:
    model = np.asarray(model_nits, np.float64)
    base = np.asarray(base_nits, np.float64)
    if model.shape != base.shape:
        raise ValueError("model/baseline shape mismatch")
    w = guard_weight(base, lo, hi, feather_px)[..., None]
    return base + w * (model - base)
