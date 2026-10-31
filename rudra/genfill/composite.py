"""The round-trip-safe composite.

out = master + alpha * (fill - master), per pixel, with alpha <= the feathered mask weight and
alpha chosen so that the master's SDR view keeps every 8-bit code:

    rint(255 * clip(view(out))) == rint(255 * clip(view(master)))   on every pixel

The R contract (1 code to the source on 99.9 % of mask-0 pixels) therefore holds for the filled
master exactly as it held for the master: in a clipped highlight any brighter value still views
as 255, so the fill is free there; in a crushed shadow any value below the code's upper edge
still views as 0; a fill that would change a visible code is pulled back towards the master by
bisection (the largest tested alpha that keeps the codes; alpha = 0 always does).

`view` is any object with forward(acescg (N, 3) float32) -> display RGB, e.g. rudra.aces_view.ACES2View.
"""
from __future__ import annotations

import numpy as np


def _codes(view, rgb: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(view.forward(np.ascontiguousarray(rgb, np.float32)), 0.0, 1.0) * 255.0).astype(np.int16)


def r_safe_blend(master: np.ndarray, fill: np.ndarray, weight: np.ndarray, view,
                 iterations: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """master, fill (H, W, 3) ACEScg; weight (H, W) in [0, 1] -> (out (H, W, 3), alpha (H, W))."""
    if master.shape != fill.shape or master.shape[:2] != weight.shape:
        raise ValueError('master, fill and weight must agree in size')
    out = np.array(master, np.float32, copy=True)
    alpha = np.zeros(weight.shape, np.float32)
    idx = np.nonzero(weight > 0)
    if not len(idx[0]):
        return out, alpha
    m = master[idx].astype(np.float32)
    d = fill[idx].astype(np.float32) - m
    w = np.clip(weight[idx].astype(np.float32), 0.0, 1.0)
    ref = _codes(view, m)

    def ok(a):
        return (_codes(view, m + a[:, None] * d) == ref).all(-1)

    good = ok(w)
    best = np.where(good, w, 0.0).astype(np.float32)
    bad = np.nonzero(~good)[0]
    if len(bad):
        lo = np.zeros(len(bad), np.float32)
        hi = w[bad].copy()
        mb, db, rb = m[bad], d[bad], ref[bad]
        for _ in range(iterations):
            mid = (lo + hi) * 0.5
            g = (_codes(view, mb + mid[:, None] * db) == rb).all(-1)
            lo = np.where(g, mid, lo)
            hi = np.where(g, hi, mid)
        best[bad] = lo
    alpha[idx] = best
    out[idx] = m + best[:, None] * d
    return out, alpha
