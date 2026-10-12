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

BASELINE_CURVES: tuple[str, ...] = ("unknown", "bt1886", "aces", "hable", "agx", "camera_log", "clip")
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


def _inverse(curve: str, display: np.ndarray) -> np.ndarray:
    if curve in ("unknown", "aces"):
        return _inverse_aces(display)
    if curve == "bt1886":
        return np.power(_bt709_oetf(np.clip(display, 0.0, 0.995)), 2.4).astype(np.float32)
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
