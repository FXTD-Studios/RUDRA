"""Tighter appearance for PARTIALLY clipped highlights (the 0 EV arm, 2 Nov 2026).

The learned model's 0 EV grading failure is all in partially clipped highlight
pixels: the 1-code constraint lets them sit up to a code from the source, and
at 0 EV the analytic inverse sits almost exactly on it (fresh FXTD v2: learned
0 EV 0.224 vs analytic 0.057 code^2; RESULT_FULL_ONLY_ROUTE_2026_11_02.md).

Applied AFTER constrain_colour, on its output, to support pixels that are not
fully clipped: if the displayed codes are more than `tol_codes` from the source,
move the pixel toward the analytic baseline (which the view returns to the
source) until they are within it.

  mode 'halving'  the 25 Oct attempt: alpha 1, 1/2, ... 1/256, 0, the steps
                  constrain_colour uses (0 EV .0694, but the aligned-residual
                  motion gate broke).
  mode 'bisect'   the hypothesis: the largest alpha in [0, 1] that meets the
                  tolerance, by bisection to 2^-iters. Alpha then moves
                  continuously with the input instead of jumping a factor of 2,
                  so neighbouring frames should not flip between steps.

Fully clipped pixels, pixels outside support and pixels already within the
tolerance are returned bit-exact. Registration key: "partial_tighten":
{"mode": "bisect", "tol_codes": 0.5, "iters": 12}.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, laplace, uniform_filter

FULL_CODE = 254.5 / 255.0


def partial_support(source, weight):
    source = np.asarray(source)
    return (np.asarray(weight) > 0) & (source.max(-1) > .5) & (source.min(-1) < FULL_CODE)


def tighten_partial(source, constrained, baseline, weight, view, mode='bisect', tol_codes=0.5, iters=12, band_codes=0.0,
                    target='source', dq_iters=200, dq_max_lap=1.5, strength=1.0):
    """-> (result, diagnostics). `constrained` is constrain_colour's output."""
    source = np.asarray(source, np.float32)
    cur = np.asarray(constrained, np.float32)
    base = np.asarray(baseline, np.float32)
    if cur.shape != source.shape or base.shape != source.shape or np.asarray(weight).shape != source.shape[:2]:
        raise ValueError('matching source, constrained, baseline and weight required')
    if mode == 'lock':
        return lock_observed(source, constrained, baseline, weight, view, iters=max(3, min(int(iters), 6)), band_codes=band_codes,
                             target=target, dq_iters=dq_iters, dq_max_lap=dq_max_lap, strength=strength)
    if target != 'source':
        raise ValueError("target='dequant' is a lock-mode option")
    if mode not in ('halving', 'bisect'):
        raise ValueError("mode must be 'halving', 'bisect' or 'lock'")
    out = cur.copy()
    sup = partial_support(source, weight)
    idx = np.flatnonzero(sup)
    diag = {'mode': mode, 'tol_codes': float(tol_codes), 'support_pixels': int(idx.size), 'changed_pixels': 0,
            'baseline_fallback_pixels': 0, 'remaining_errors': 0}
    if idx.size == 0:
        return out, diag
    src = source.reshape(-1, 3)[idx]
    c = cur.reshape(-1, 3)[idx]
    b = base.reshape(-1, 3)[idx]
    err = lambda x: np.abs(np.asarray(view.forward(x), np.float32) - src).max(-1) * 255.0
    bad = err(c) > tol_codes
    if not bad.any():
        return out, diag
    bi = np.flatnonzero(bad)
    cs, bs, ss = c[bi], b[bi], src[bi]
    errb = lambda x: np.abs(np.asarray(view.forward(x), np.float32) - ss).max(-1) * 255.0
    if mode == 'halving':
        alpha = np.ones(bi.size, np.float32)
        active = np.ones(bi.size, bool)
        res = cs.copy()
        for step in range(9):
            alpha[active] = alpha[active] * .5 if step < 8 else 0.0
            res[active] = bs[active] + alpha[active, None] * (cs[active] - bs[active])
            ok = errb(res) <= tol_codes
            active &= ~ok
            if not active.any():
                break
    else:
        lo = np.zeros(bi.size, np.float32)   # alpha 0: the baseline, feasible by construction
        hi = np.ones(bi.size, np.float32)    # alpha 1: the constrained output, infeasible here
        for _ in range(int(iters)):
            mid = (lo + hi) * .5
            ok = errb(bs + mid[:, None] * (cs - bs)) <= tol_codes
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        alpha = lo
        res = bs + alpha[:, None] * (cs - bs)
    final_err = errb(res)
    diag['changed_pixels'] = int(bi.size)
    diag['baseline_fallback_pixels'] = int((alpha == 0).sum())
    diag['remaining_errors'] = int((final_err > tol_codes + 1e-6).sum())
    flat = out.reshape(-1, 3)
    flat[idx[bi]] = res
    return out, diag


def from_registration(reg):
    """The arm's settings from a registration's "partial_tighten" key, or None (off)."""
    cfg = reg.get('partial_tighten') if isinstance(reg, dict) else None
    if not cfg:
        return None
    out = {'mode': cfg.get('mode', 'bisect'), 'tol_codes': float(cfg.get('tol_codes', 0.5)), 'iters': int(cfg.get('iters', 12))}
    if 'band_codes' in cfg:
        out['band_codes'] = float(cfg['band_codes'])
    if cfg.get('target', 'source') != 'source':
        out['target'] = str(cfg['target'])
        out['dq_iters'] = int(cfg.get('dq_iters', 200))
        out['dq_max_lap'] = float(cfg.get('dq_max_lap', 1.5))
    if 'strength' in cfg:
        out['strength'] = float(cfg['strength'])
    return out


def dequantize_recorded(source, region=None, iters=200, max_lap=1.5, half_bin=0.49, clip_ceiling=4.0):
    """The lock's target without the 8-bit rounding (5 Nov 2026). -> (codes / 255, used mask).

    The exact lock pins a recorded channel to the source's integer code, but the true
    display value lies anywhere in that code's bin (+-0.5), and as content moves by a
    fraction of a pixel the rounding flips frame to frame: that rounding noise is what
    the lock copied into motion (aligned residual 0.0129). Per channel, this finds the
    smoothest field (least squared Laplacian, projected accelerated gradient, as in
    aces_curvature_dequant) that stays inside every recorded code's bin; a clipped
    sample is only bounded below by the clip, so it does not drag the field down.

    Used only where the content is smooth (mean |code Laplacian| over 5x5 <= max_lap:
    rounding alone gives at most 2, texture or noise much more); elsewhere, and in
    every clipped channel, the source code itself. The result never leaves the bin
    (|result - code| <= half_bin), so it is always a value the camera could have shown.
    No reference data and no learned weights. `region`: a bool HxW mask; the solve
    runs on its bounding box (plus a margin) only.
    """
    src = np.asarray(source, np.float64)
    if src.ndim != 3 or src.shape[-1] != 3 or not np.isfinite(src).all():
        raise ValueError('finite HWC RGB source required')
    out = src.copy()
    used = np.zeros(src.shape, bool)
    if region is not None:
        region = np.asarray(region, bool)
        if not region.any():
            return out, used
        ys, xs = np.nonzero(region)
        m = 24
        y0, y1 = max(int(ys.min()) - m, 0), min(int(ys.max()) + m + 1, src.shape[0])
        x0, x1 = max(int(xs.min()) - m, 0), min(int(xs.max()) + m + 1, src.shape[1])
    else:
        y0, y1, x0, x1 = 0, src.shape[0], 0, src.shape[1]
    codes = src[y0:y1, x0:x1] * 255.0
    rec = codes < 254.5
    lo = np.where(rec, codes - half_bin, 254.5)
    hi = np.where(rec, codes + half_bin, 255.0 + clip_ceiling)
    u = codes.copy()
    y = u.copy()
    t = 1.0
    for _ in range(int(iters)):   # ||Laplacian|| <= 8, so a 1/64 step on L^T L is nonexpansive
        g = np.stack([laplace(laplace(y[..., c], mode='reflect'), mode='reflect') for c in range(3)], -1)
        nu = np.clip(y - g / 64.0, lo, hi)
        nt = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = nu + ((t - 1) / nt) * (nu - u)
        u = nu
        t = nt
    smooth = np.zeros_like(rec)
    for c in range(3):
        k = codes[..., c]
        r = rec[..., c]
        lap = np.zeros_like(k)
        ok = np.zeros_like(r)
        lap[1:-1, 1:-1] = k[:-2, 1:-1] + k[2:, 1:-1] + k[1:-1, :-2] + k[1:-1, 2:] - 4 * k[1:-1, 1:-1]
        ok[1:-1, 1:-1] = r[1:-1, 1:-1] & r[:-2, 1:-1] & r[2:, 1:-1] & r[1:-1, :-2] & r[1:-1, 2:]
        num = uniform_filter(np.abs(lap) * ok, 5, mode='nearest')
        den = uniform_filter(ok.astype(np.float64), 5, mode='nearest')
        smooth[..., c] = (den > 0.2) & (num <= max_lap * den)
    use = rec & smooth
    out[y0:y1, x0:x1] = np.where(use, u, codes) / 255.0
    used[y0:y1, x0:x1] = use
    return out, used


def lock_observed(source, constrained, baseline, weight, view, iters=4, eps=2e-3, band_codes=0.0,
                  target='source', dq_iters=200, dq_max_lap=1.5, strength=1.0):
    """The 0 EV fix without new data (5 Nov 2026).

    In a partially clipped pixel (codes such as 255, 250, 230) the clipped channel
    shows 255 for the model and for the reference alike, so it cannot make 0 EV
    error; the error is all in the channels the camera RECORDED (here 250 and 230),
    which the 1-code constraint lets the model drift in. This holds each recorded
    channel's displayed value at the plain inverse's (which matches the source) and
    leaves the clipped channel(s), the highlight reconstruction, to the model.

    The view mixes channels, so per pixel this solves for the smallest change in
    ACEScg that puts the recorded channels' display back: Gauss-Newton on
    view(x)[recorded] = view(baseline)[recorded], the 3x3 Jacobian by finite
    differences, rows weighted by which channels are recorded. Continuous in the
    input (no halving steps), so neighbouring frames move together. A pixel whose
    clipped channel would drop below the clip, or that does not converge, keeps
    its constrained value. Fully clipped pixels and pixels outside support are
    returned bit-exact.

    band_codes (6 Nov 2026): a recorded channel within this many codes of the source is
    left as the model made it; only one outside the band is brought to the band's edge.
    0 is the exact lock. The exact lock passes 0 EV on real footage (fresh 0.241 against
    0.453) but makes motion follow the source's frame-to-frame code jitter (aligned
    residual 0.0129 against 0.0104); a band keeps the model's smooth values inside it.

    target (5 Nov 2026): 'source' locks to the recorded integer codes (as registered on
    5 and 6 Nov, bit for bit); 'dequant' locks to dequantize_recorded's estimate of the
    value before 8-bit rounding, which does not flip as content moves. Either way a pixel
    is only changed if every recorded channel ends within 1 code of the source.
    """
    if target not in ('source', 'dequant'):
        raise ValueError("target must be 'source' or 'dequant'")
    source = np.asarray(source, np.float32)
    cur = np.asarray(constrained, np.float32)
    base = np.asarray(baseline, np.float32)
    out = cur.copy()
    sup = partial_support(source, weight)
    idx = np.flatnonzero(sup)
    band = float(band_codes) / 255.0
    diag = {'mode': 'lock', 'band_codes': float(band_codes), 'support_pixels': int(idx.size), 'changed_pixels': 0, 'kept_pixels': 0,
            'max_recorded_error_codes': 0.0}
    if idx.size == 0:
        return out, diag
    src = source.reshape(-1, 3)[idx].astype(np.float64)
    x = cur.reshape(-1, 3)[idx].astype(np.float64)
    if target == 'dequant':
        dq, used = dequantize_recorded(source, region=sup, iters=dq_iters, max_lap=dq_max_lap)
        diag['target'] = 'dequant'
        diag['dequant_channels'] = int(used.reshape(-1, 3)[idx].sum())
        target = dq.reshape(-1, 3)[idx]
    else:
        target = src   # the recorded codes themselves: what the camera measured (the plain inverse returns them)
    rec = src < FULL_CODE                              # the recorded channels
    clipped = ~rec
    fwd = lambda v: np.asarray(view.forward(v.astype(np.float32)[None]), np.float64)[0]
    first = fwd(x)
    if band > 0:   # inside the band already: leave the pixel as the model made it (bit-exact)
        inside = (~rec | (np.abs(first - target) <= band)).all(1)
        keep_idx, idx, src, x, rec, clipped, target = idx[inside], idx[~inside], src[~inside], x[~inside], rec[~inside], clipped[~inside], target[~inside]
        diag['inside_band_pixels'] = int(inside.sum())
        if idx.size == 0:
            return out, diag
    for _ in range(iters):
        d = fwd(x)
        goal = np.clip(d, target - band, target + band) if band > 0 else target   # the band's nearest edge
        r = np.where(rec, goal - d, 0.0)
        J = np.empty((x.shape[0], 3, 3))
        for c in range(3):
            h = eps * np.maximum(np.abs(x[:, c]), 1e-3)
            xp = x.copy()
            xp[:, c] += h
            J[:, :, c] = (fwd(xp) - d) / h[:, None]
        # The smallest RELATIVE change (in log x, so every channel stays positive and moves
        # in proportion) that brings the recorded channels' display back: the pseudo-inverse
        # of the recorded rows, the clipped row left free. The view mixes channels, so a
        # rebuilt clipped channel that the recorded ones could not have been seen with is
        # brought down to what they allow: the camera saw those values with the true one.
        xp_ = np.maximum(x, 1e-6)
        Jr = J * xp_[:, None, :] * rec[:, :, None]
        du = (np.linalg.pinv(Jr, rcond=1e-10) @ r[..., None])[..., 0]
        x = xp_ * np.exp(np.clip(du, -2.0, 2.0))
    d = fwd(x)
    err = np.maximum(np.abs(np.where(rec, d - target, 0.0)).max(1) * 255.0 - float(band_codes), 0.0)
    near_source = (np.abs(np.where(rec, d - src, 0.0)).max(1) * 255.0) <= 1.0   # the 1-code bound, whatever the target
    ok = np.isfinite(x).all(1) & (err <= 0.5) & near_source & ~(clipped & (d < FULL_CODE)).any(1)
    flat = out.reshape(-1, 3)
    if strength < 1.0:
        # Partial strength (6 Nov 2026): a geometric fraction lambda of the way from the model to
        # the lock, channel by channel (each stays positive). Both gates are squared errors: the
        # 0 EV drift left is (1 - lambda)^2 of the model's, the frame-to-frame jitter added is
        # lambda^2 of the full lock's. Measured ends (w4): model 0.453 / 0.0090, full lock with
        # the source target 0.241 / 0.0129, with the dequantized target 0.239 / 0.0119.
        lam = float(strength)
        raw = cur.reshape(-1, 3)[idx][ok].astype(np.float64)
        flat[idx[ok]] = (np.maximum(raw, 1e-9) ** (1.0 - lam) * np.maximum(x[ok], 1e-9) ** lam).astype(np.float32)
    else:
        flat[idx[ok]] = x[ok].astype(np.float32)
    diag['strength'] = float(strength)
    diag['changed_pixels'] = int(ok.sum())
    diag['kept_pixels'] = int((~ok).sum())
    diag['max_recorded_error_codes'] = float(err[ok].max()) if ok.any() else 0.0
    return out, diag
