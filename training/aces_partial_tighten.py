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

FULL_CODE = 254.5 / 255.0


def partial_support(source, weight):
    source = np.asarray(source)
    return (np.asarray(weight) > 0) & (source.max(-1) > .5) & (source.min(-1) < FULL_CODE)


def tighten_partial(source, constrained, baseline, weight, view, mode='bisect', tol_codes=0.5, iters=12):
    """-> (result, diagnostics). `constrained` is constrain_colour's output."""
    source = np.asarray(source, np.float32)
    cur = np.asarray(constrained, np.float32)
    base = np.asarray(baseline, np.float32)
    if cur.shape != source.shape or base.shape != source.shape or np.asarray(weight).shape != source.shape[:2]:
        raise ValueError('matching source, constrained, baseline and weight required')
    if mode == 'lock':
        return lock_observed(source, constrained, baseline, weight, view, iters=max(3, min(int(iters), 6)))
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
    return {'mode': cfg.get('mode', 'bisect'), 'tol_codes': float(cfg.get('tol_codes', 0.5)), 'iters': int(cfg.get('iters', 12))}


def lock_observed(source, constrained, baseline, weight, view, iters=4, eps=2e-3):
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
    """
    source = np.asarray(source, np.float32)
    cur = np.asarray(constrained, np.float32)
    base = np.asarray(baseline, np.float32)
    out = cur.copy()
    sup = partial_support(source, weight)
    idx = np.flatnonzero(sup)
    diag = {'mode': 'lock', 'support_pixels': int(idx.size), 'changed_pixels': 0, 'kept_pixels': 0,
            'max_recorded_error_codes': 0.0}
    if idx.size == 0:
        return out, diag
    src = source.reshape(-1, 3)[idx].astype(np.float64)
    x = cur.reshape(-1, 3)[idx].astype(np.float64)
    target = src   # the recorded codes themselves: what the camera measured (the plain inverse returns them)
    rec = src < FULL_CODE                              # the recorded channels
    clipped = ~rec
    fwd = lambda v: np.asarray(view.forward(v.astype(np.float32)[None]), np.float64)[0]
    for _ in range(iters):
        d = fwd(x)
        r = np.where(rec, target - d, 0.0)
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
    err = np.abs(np.where(rec, d - target, 0.0)).max(1) * 255.0
    ok = np.isfinite(x).all(1) & (err <= 0.5) & ~(clipped & (d < FULL_CODE)).any(1)
    flat = out.reshape(-1, 3)
    flat[idx[ok]] = x[ok].astype(np.float32)
    diag['changed_pixels'] = int(ok.sum())
    diag['kept_pixels'] = int((~ok).sum())
    diag['max_recorded_error_codes'] = float(err[ok].max()) if ok.any() else 0.0
    return out, diag
