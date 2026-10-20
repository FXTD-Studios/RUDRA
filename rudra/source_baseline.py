"""The Source picker's baselines, as the desktop app computes them (v10, 13 Oct 2026).

The app's baseline is the inverse of the curve the artist picks in
Reconstruct > Source (native/core/src/source_curve.cpp): the ACES inverse times
a per-code log2 correction, one knot per 8-bit code, linear between knots. Since
13 Oct 2026 the default is BT.1886, which on real SDR scores +14 dB over the ACES
inverse every model before v10 was trained on (docs/RESULTS.md, R3).

v10 trains its residual over THIS baseline, so the model learns only what the
picked curve cannot know. This module is the Python twin of build_params():

    knots[code] = log2(inverse_curve(display) / inverse_aces(display)),
    display = srgb_to_linear(code / 255), knots[0] = knots[1].

``BASELINE_CURVES`` is the app's picker, in its own id strings. "unknown" and
"aces" are the ACES inverse itself (all-zero knots).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

# "bt1886_shadows" (14 Oct 2026, opt-in) is appended last so existing indices keep their meaning.
BASELINE_CURVES: tuple[str, ...] = ("unknown", "bt1886", "aces", "hable", "agx", "camera_log", "clip", "bt1886_shadows")
DEFAULT_BASELINE = "bt1886"     # the app's default for new sessions since 13 Oct 2026
KNOTS = 256


def baseline_curve_index(name: str | None) -> int:
    """Picker id -> index into BASELINE_CURVES; None is the app default (bt1886)."""
    name = DEFAULT_BASELINE if name is None else str(name)
    try:
        return BASELINE_CURVES.index(name)
    except ValueError as exc:
        raise ValueError(f"unknown baseline curve {name!r}; the picker has {BASELINE_CURVES}") from exc


# native/core/src/source_curve.cpp, float32 as it runs.
def _f(x):
    return np.asarray(x, dtype=np.float32)


def _forward_hable(x):
    a, b, c, d, e, f = (np.float32(v) for v in (0.15, 0.50, 0.10, 0.20, 0.02, 0.30))

    def h(v):
        return (v * (a * v + c * b) + d * e) / (v * (a * v + b) + d * f) - e / f
    return np.clip(h(x * np.float32(2.0)) / h(np.float32(11.2)), 0.0, 1.0).astype(np.float32)


def _forward_agx(x):
    lo, hi = np.float32(-12.47), np.float32(4.03)
    t = (np.log2(np.maximum(x, np.exp2(lo))) - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)
    s = 1.0 / (1.0 + np.exp(-(t - 0.6) * 10.0))
    s0 = 1.0 / (1.0 + np.exp(np.float32(0.6 * 10.0)))
    s1 = 1.0 / (1.0 + np.exp(np.float32(-0.4 * 10.0)))
    return np.power(np.clip((s - s0) / (s1 - s0), 0.0, 1.0), 2.2).astype(np.float32)


def _forward_camera_log(x):
    lg = np.log2(np.maximum(x, np.float32(1e-6)) / np.float32(0.18)) / np.float32(14.0) + np.float32(0.5)
    lg = np.clip(lg, 0.0, 1.0)
    k = np.float32(7.0)
    s = 1.0 / (1.0 + np.exp(-(lg - 0.5) * k))
    s0 = 1.0 / (1.0 + np.exp(0.5 * k))
    s1 = 1.0 / (1.0 + np.exp(-0.5 * k))
    return np.power(np.clip((s - s0) / (s1 - s0), 0.0, 1.0), 2.4).astype(np.float32)


def _bt709_oetf(x):
    return np.where(x < 0.018, 4.5 * x, 1.099 * np.power(np.maximum(x, 0.0), 0.45) - 0.099).astype(np.float32)


def _table_inverse(forward, display):
    """InverseTable: 8001 log-spaced exposures over 22 stops, bisection + lerp."""
    x = np.exp2(np.linspace(-16.0, 6.0, 8001)).astype(np.float32)
    y = np.maximum.accumulate(forward(x).astype(np.float32))
    v = np.clip(display, 0.0, 0.995).astype(np.float32)
    i = np.searchsorted(y, v, side="left")
    out = np.empty_like(v)
    for j, (vi, ii) in enumerate(zip(v.ravel(), i.ravel())):
        if ii == 0:
            out.flat[j] = 0.0
        elif ii >= len(y):
            out.flat[j] = x[-1]
        else:
            y0, y1 = y[ii - 1], y[ii]
            t = (vi - y0) / (y1 - y0) if y1 > y0 else 0.0
            out.flat[j] = x[ii - 1] + (x[ii] - x[ii - 1]) * t
    return out


def _inverse_aces(display):
    from rudra.sdr2hdr import inverse_aces_approx
    return inverse_aces_approx(torch.as_tensor(display, dtype=torch.float32)).numpy()


# BT.1886 shadows: log2 scene-linear at SDR codes 1..39, the same numbers as
# native/core/src/source_curve.cpp kShadowLog2Scene (tools/shadow_correction_eval.py,
# fit_v3.json; docs/ai/SHADOW_CORRECTION_RESULT.md). Code 40 up is BT.1886.
SHADOW_KNEE = 40
SHADOW_LOG2_SCENE = np.array([0.0,
    -14.685428, -14.249545, -14.009058, -13.741516, -13.537101, -13.242504,
    -13.008029, -12.860731, -12.457915, -12.053934, -12.033934, -12.013934,
    -11.993934, -11.973934, -11.252472, -10.590089, -10.570089, -10.389724,
    -10.164267, -10.013963, -9.968597, -9.929899, -9.870572, -9.778883,
    -9.693991, -9.599686, -9.497595, -9.392905, -9.292866, -9.197775,
    -9.111351, -9.028863, -8.925111, -8.815446, -8.703925, -8.587965,
    -8.450235, -8.357113, -8.253048,
], np.float32)


def _inverse_bt1886(display):
    return np.power(_bt709_oetf(np.clip(display, 0.0, 0.995)), 2.4).astype(np.float32)


def _inverse_bt1886_shadows(display):
    """native inverse_bt1886_shadows: BT.1886 from the knee code's display value up."""
    display = np.asarray(display, np.float32)
    knee_display = np.float32(((SHADOW_KNEE / 255.0 + 0.055) / 1.055) ** 2.4)
    x = np.clip(np.minimum(display, np.float32(0.995)), 0.0, 1.0)
    code = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32) * 255.0
    code = np.minimum(code, SHADOW_KNEE)
    knee = np.log2(_inverse_bt1886(np.asarray([knee_display]))[0])
    table = np.append(SHADOW_LOG2_SCENE, knee)
    i = np.clip(np.floor(code).astype(int), 1, SHADOW_KNEE - 1)
    t = code - i
    lifted = np.exp2(table[i] + (table[i + 1] - table[i]) * t)
    lifted = np.where(code < 1.0, code * np.exp2(table[1]), lifted)
    lifted = np.where(code <= 0.0, 0.0, lifted)
    return np.where(display >= knee_display, _inverse_bt1886(display), lifted).astype(np.float32)


def _inverse(curve: str, display: np.ndarray) -> np.ndarray:
    if curve in ("unknown", "aces"):
        return _inverse_aces(display)
    if curve == "bt1886":
        return _inverse_bt1886(display)
    if curve == "bt1886_shadows":
        return _inverse_bt1886_shadows(display)
    if curve == "clip":
        return np.clip(display, 0.0, 0.995).astype(np.float32)
    forward = {"hable": _forward_hable, "agx": _forward_agx, "camera_log": _forward_camera_log}[curve]
    return _table_inverse(forward, display)


@lru_cache(maxsize=None)
def baseline_knots(curve: str) -> np.ndarray:
    """(257,) float32: [exposure 0, 256 per-code log2 corrections over the ACES inverse]."""
    if curve not in BASELINE_CURVES:
        raise ValueError(f"unknown baseline curve {curve!r}")
    out = np.zeros(1 + KNOTS, np.float32)
    if curve in ("unknown", "aces"):
        return out
    from rudra.sdr2hdr import srgb_to_linear
    display = srgb_to_linear(torch.arange(KNOTS, dtype=torch.float32) / 255.0).numpy()
    want, have = _inverse(curve, display), _inverse_aces(display)
    ok = (want > 0) & (have > 0)
    corr = np.where(ok, np.log2(np.where(ok, want, 1.0) / np.where(ok, have, 1.0)), 0.0).astype(np.float32)
    corr[0] = corr[1]
    out[1:] = corr
    out.setflags(write=False)
    return out


def knots_table(device=None, dtype=torch.float32) -> torch.Tensor:
    """(len(BASELINE_CURVES), 256) per-code log2 corrections, one row per picker entry."""
    return torch.from_numpy(np.stack([baseline_knots(c)[1:] for c in BASELINE_CURVES])).to(device=device, dtype=dtype)


def correction_log2(sdr: torch.Tensor, ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Per-pixel, per-channel log2 correction of the picked curves.

    ``sdr`` (B,3,H,W) codes, ``ids`` (B,) indices into BASELINE_CURVES, ``table``
    from knots_table(). Linear between knots, as composite.frag curveLog2.
    """
    b = sdr.shape[0]
    ids = torch.as_tensor(ids, device=sdr.device).long().reshape(-1)
    if ids.numel() == 1 and b > 1:
        ids = ids.expand(b)
    rows = table.to(sdr.device, sdr.dtype)[ids]                  # (B, 256)
    pos = sdr.clamp(0.0, 1.0) * (KNOTS - 1)
    lo = pos.floor().clamp(max=KNOTS - 2).long()
    frac = pos - lo.to(pos.dtype)
    flat = lo.reshape(b, -1)
    v0 = torch.gather(rows, 1, flat).reshape_as(sdr)
    v1 = torch.gather(rows, 1, flat + 1).reshape_as(sdr)
    return v0 + (v1 - v0) * frac
