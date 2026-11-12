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


def plain_blend(master: np.ndarray, fill: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """out = master + weight * (fill - master): the region re-render (v4, 9 Nov 2026). Inside the
    mask the SDR codes are NOT kept; outside it (weight 0) every pixel is the master's, bit for bit."""
    if master.shape != fill.shape or master.shape[:2] != weight.shape:
        raise ValueError('master, fill and weight must agree in size')
    alpha = np.clip(weight.astype(np.float32), 0.0, 1.0)
    out = np.array(master, np.float32, copy=True)
    idx = np.nonzero(alpha > 0)
    if len(idx[0]):
        out[idx] = master[idx] + alpha[idx][:, None] * (fill[idx].astype(np.float32) - master[idx])
    return out, alpha


LUMA_AP1 = np.array([.2722287168, .6740817658, .0536895174], np.float32)
RERENDER_HUES = ('model', 'source')
RERENDER_FULL_CODE = 253        # the source's max code at which the fill sits at the whole level
RERENDER_HUE_SIGMA = 8.0        # pixels: the source's colour is taken from its neighbourhood, not per pixel
RERENDER_DETAIL_SIGMA = 64.0    # pixels at 1080 lines: above this scale the level is the source's (or the model's, within the ceiling)
# How far over `level` the model's own level may sit. 0 since 11 Nov 2026: the model's level is not steady
# (Porch: the region's median fell from 5.5 to 3.3 x white over the first 8 frames with 2 stops allowed, and
# held 2.8-3.2 with none), and with the crossfade by code the allowance bought nothing on the held-out clips
# (PU21 at -3 EV -28.1 with it, -27.6 without) or on 09V (no pixel moved).
RERENDER_CEILING_STOPS = 0.0
RERENDER_FRINGE_PX = 12.0       # pixels at 1080 lines: this close to a darker neighbour the fill is the interior's
RERENDER_REACH_PX = 6.0         # pixels at 1080 lines: beside something dark a code takes its brightest neighbour's within this
RERENDER_DARK_CODES = 40        # "something dark": a neighbour this many codes under tau_high (fully so at 80)


RERENDER_NEUTRAL = (0.78, 0.88) # the source's min over max code: below the first the region is left to the master, whole from the second


def neutral_weight(codes: np.ndarray) -> np.ndarray:
    """(H, W) in [0, 1]: how near white the source is, from its codes (H, W, 3). The region re-render
    is for an area the source parks near white in every channel: nothing is known there. A saturated
    highlight (a sunset: red at 255, blue at 120) still carries its colour and its gradient, and a fill
    at a level of its own draws a contour across it (11 Nov 2026, the 07V still: the mode whitened the
    sky above a hard edge). There the weight goes to 0 and the master stays."""
    c = codes.astype(np.float32)
    lo, hi = RERENDER_NEUTRAL
    x = np.clip((c.min(-1) / np.maximum(c.max(-1), 1.0) - lo) / (hi - lo), 0.0, 1.0)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


def white_scale(master: np.ndarray, codes: np.ndarray) -> float | None:
    """The master's SDR white on one frame: the median, over mid-tone pixels (every channel between
    codes 50 and 180), of the master's luminance over the source's display light (code ^ 2.4), as
    tools/compare_hdr_renders.py reads its 'scale'. None when the frame has too few mid-tones."""
    mid = np.all((codes > 50) & (codes < 180), -1)
    if int(mid.sum()) < 500:
        return None
    light = np.power(codes[mid].astype(np.float32) / 255.0, 2.4) @ LUMA_AP1
    return float(np.median((master[mid] @ LUMA_AP1) / np.maximum(light, 1e-6)))


def finish_rerender(fill: np.ndarray, master: np.ndarray, source_lin: np.ndarray, codes: np.ndarray, weight: np.ndarray,
                    hue: str = 'model', colour: float = 0.0, level: float = 0.0, smooth: float = 0.0,
                    white: float = 1.0, tau_high: int = 240, state: dict | None = None) -> np.ndarray:
    """The region re-render's fill, finished before the blend (10 Nov 2026, after the v4 gates on 09V:
    the v4 LoRA drew the structure, but too blue, at 1.2 x white where 2.5-4 was asked, its level
    drifting 0.85-1.72 x white along the shot with jumps at the window seams, and brighter where the
    source is darker).

    fill, master, source_lin (the source's display light, code ^ 2.4) (H, W, 3) ACEScg; codes (H, W, 3)
    8-bit; weight (H, W) the blend weight. Every step is off at its default, and then the fill is
    returned as is:
      hue 'source'  the fill keeps its luminance and takes the source's colour (the source's display
                    light blurred RERENDER_HUE_SIGMA px, so near-clip noise does not tint it), mixed
                    with `colour` (0..1) of the model's own
      level > 0     the detail from the model, the level `level` x `white` (the master's SDR white,
                    white_scale): the fill keeps what is finer than RERENDER_DETAIL_SIGMA; above that
                    scale its own level is kept only where brighter and only RERENDER_CEILING_STOPS
                    over (0: not at all; see the constant).
                    That levelled fill is then crossfaded into the MASTER by the source's code, log
                    luminance and colour alike: the master itself at tau_high, the whole fill at
                    RERENDER_FULL_CODE. So the lift starts at nothing where the region starts and a
                    brighter source stays brighter (11 Nov, the Porch shot: a softbox's hot spots,
                    a smooth gradient through tau_high, came out as ovals with an edge when the
                    model's level applied whole from the region's rim).
                    Two edge rules go with it: within RERENDER_FRINGE_PX of a darker neighbour the
                    fill is taken from the interior (the model's fill is soft and dark beside a
                    silhouette), and beside something dark a code takes its brightest neighbour's
                    within RERENDER_REACH_PX (an anti-aliased edge is a mix of figure and sky, not a
                    darker sky; in a gradient there is nothing dark and the codes stand)
      smooth > 0    the fill's log luminance follows the previous frame's (an EMA, this share kept)
                    wherever both frames are filled; `state` (a dict the caller keeps for the shot)
                    carries it. A newly filled pixel starts from its own value."""
    if hue not in RERENDER_HUES:
        raise ValueError(f'hue must be one of {", ".join(RERENDER_HUES)}')
    if not 0.0 <= smooth < 1.0 or not 0.0 <= colour <= 1.0 or level < 0:
        raise ValueError('smooth must be in [0, 1), colour in [0, 1], level >= 0')
    if hue == 'model' and level <= 0 and smooth <= 0:
        return fill
    import cv2
    on = weight > 0
    if not on.any():
        return fill
    y = np.maximum(fill @ LUMA_AP1, 1e-6).astype(np.float32)
    chroma = fill / y[..., None]
    log_y = np.log2(y)
    px = fill.shape[0] / 1080.0
    top_code = codes.max(-1).astype(np.float32)
    if level > 0:                                       # the interior's fill beside a darker neighbour
        r = max(1.0, RERENDER_FRINGE_PX * px)
        d = np.clip(cv2.distanceTransform((top_code >= tau_high - 8).astype(np.uint8), cv2.DIST_L2, 3) / r, 0.0, 1.0)
        conf = (d * d * (3.0 - 2.0 * d) * on).astype(np.float32)
        near_conf = cv2.GaussianBlur(conf, (0, 0), r)
        known = near_conf > 1e-3
        inner_y = cv2.GaussianBlur(log_y * conf, (0, 0), r) / np.maximum(near_conf, 1e-3)
        inner_c = cv2.GaussianBlur(chroma * conf[..., None], (0, 0), r) / np.maximum(near_conf, 1e-3)[..., None]
        log_y = np.where(known, conf * log_y + (1.0 - conf) * inner_y, log_y).astype(np.float32)
        chroma = np.where(known[..., None], conf[..., None] * chroma + (1.0 - conf[..., None]) * inner_c, chroma).astype(np.float32)
    if hue == 'source':
        near = cv2.GaussianBlur(np.ascontiguousarray(source_lin, np.float32), (0, 0), RERENDER_HUE_SIGMA)
        chroma = (1.0 - colour) * near / np.maximum(near @ LUMA_AP1, 1e-6)[..., None] + colour * chroma
    if level > 0:
        sigma = RERENDER_DETAIL_SIGMA * px
        inside = on.astype(np.float32)                  # the model's own neighbourhood: filled pixels only
        low = cv2.GaussianBlur(log_y * inside, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(inside, (0, 0), sigma), 1e-3)
        k = int(round(RERENDER_REACH_PX * px))
        if k > 0:
            el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            e = np.clip((tau_high - RERENDER_DARK_CODES - cv2.erode(top_code, el)) / float(RERENDER_DARK_CODES), 0.0, 1.0)
            top_code = top_code + e * e * (3.0 - 2.0 * e) * (cv2.dilate(top_code, el) - top_code)
        top = cv2.GaussianBlur(top_code, (0, 0), 2.0)
        s = np.clip((top - tau_high) / max(1.0, float(RERENDER_FULL_CODE - tau_high)), 0.0, 1.0)
        s = s * s * (3.0 - 2.0 * s)
        floor = np.float32(np.log2(level * white))
        whole = np.clip(low, floor, floor + RERENDER_CEILING_STOPS) + (log_y - low)
        y_m = np.maximum(master @ LUMA_AP1, 1e-6).astype(np.float32)
        log_y = ((1.0 - s) * np.log2(y_m) + s * whole).astype(np.float32)
        chroma = ((1.0 - s[..., None]) * (master / y_m[..., None]) + s[..., None] * chroma).astype(np.float32)
    if smooth > 0 and state is not None:
        prev = state.get('log_y')
        if prev is not None and prev.shape == log_y.shape:
            both = on & state['on']
            log_y[both] = smooth * prev[both] + (1.0 - smooth) * log_y[both]
        state['log_y'], state['on'] = log_y, on
    out = np.array(fill, np.float32, copy=True)
    out[on] = (np.exp2(log_y)[..., None] * chroma)[on]
    return out


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
