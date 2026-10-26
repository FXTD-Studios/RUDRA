"""Experimental bounded-interval diffusion; not enabled in production inference.

Inspired by Ouyang/Tumblin, Removing Quantization Artifacts in Color Images
Using Bounded Interval Regularization (2005). This implementation uses hard
projection and a stable explicit step, not the paper's soft penalty solver.
"""
import numpy as np


def bounded_interval_dequant(source, iterations=32, step=.2):
    """HWC Rec709 source codes [0,1]; deterministic, bounded by +/-0.49/255.

Fixed joint-RGB edge weights allow diffusion across <=1-code transitions.
Clipped/crushed pixels and their adjacent edges stay unchanged. This is only
an 8-bit quantization prior, not recovery of original radiance or codec truth.
"""
    src = np.asarray(source, dtype=np.float32)
    if src.ndim != 3 or src.shape[-1] != 3 or min(src.shape[:2]) < 2:
        raise ValueError('expected HWC RGB image at least 2x2')
    if not np.isfinite(src).all() or np.any((src < 0) | (src > 1)):
        raise ValueError('source must be finite in [0,1]')
    if not isinstance(iterations, int) or iterations < 0 or not np.isfinite(step) or not 0 < step <= .25:
        raise ValueError('iterations must be nonnegative; step must be in (0,.25]')
    q = src * 255
    ordinary = (q.max(axis=2) >= 5) & (q.max(axis=2) <= 254)
    wx = ((np.abs(np.diff(q, axis=1)).max(axis=2) <= 1.01) & ordinary[:, :-1] & ordinary[:, 1:])[..., None]
    wy = ((np.abs(np.diff(q, axis=0)).max(axis=2) <= 1.01) & ordinary[:-1] & ordinary[1:])[..., None]
    lo, hi = np.maximum(q-.49, 0), np.minimum(q+.49, 255)
    restored = q.copy()
    for _ in range(iterations):
        dx = np.diff(restored, axis=1) * wx
        dy = np.diff(restored, axis=0) * wy
        update = np.zeros_like(restored)
        update[:, :-1] += dx; update[:, 1:] -= dx
        update[:-1] += dy; update[1:] -= dy
        restored = np.clip(restored + step * update, lo, hi)
    result = restored / 255
    result[~ordinary] = src[~ordinary]
    return result
