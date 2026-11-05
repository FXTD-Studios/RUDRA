"""The reconstruction region (docs/composite.spec.md revision 2, RUDRA 1.1, 5 Nov 2026).

The oracle for native/core composite.cpp ``reconstruction_gate`` and
render/shaders/composite.frag: where the reconstruction may act, by the SDR's
Rec.709 luma. Off, the gate is revision 1's (fixed sigmoid priors at 0.82 and
0.10, slope 24). On, the priors are weighted by compact smoothstep ramps, each
scaled by its own strength; wherever the gate is exactly 0 the composite
returns the baseline itself, so preserved pixels are bit-identical.

Pure numpy, float32 throughout, as the C++ (fp32) is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ReconstructionRegion", "region_ramp_up", "region_ramp_down", "region_weights",
           "reconstruction_gate", "MODES"]

MODES = ("all", "highlights", "shadows", "off")
_F = np.float32


@dataclass(frozen=True)
class ReconstructionRegion:
    enabled: bool = False
    band: bool = False
    highlight_from: float = 0.75
    highlight_softness: float = 0.10
    highlight_strength: float = 1.0
    shadow_below: float = 0.15
    shadow_softness: float = 0.10
    shadow_strength: float = 1.0
    band_low: float = 0.50
    band_high: float = 0.90
    band_softness: float = 0.05


def _smooth(t: np.ndarray) -> np.ndarray:
    return t * t * (_F(3.0) - _F(2.0) * t)


def region_ramp_up(y: np.ndarray, at: float, softness: float) -> np.ndarray:
    """1 at y >= at + softness/2, 0 at y <= at - softness/2 (a hard step at softness 0)."""
    y = np.asarray(y, _F)
    at, soft = _F(at), _F(softness)
    if not soft > 0:
        return (y >= at).astype(_F)
    t = np.clip((y - (at - _F(0.5) * soft)) / soft, _F(0.0), _F(1.0))
    return _smooth(t).astype(_F)


def region_ramp_down(y: np.ndarray, at: float, softness: float) -> np.ndarray:
    """1 at y <= at - softness/2, 0 at y >= at + softness/2."""
    y = np.asarray(y, _F)
    at, soft = _F(at), _F(softness)
    if not soft > 0:
        return (y <= at).astype(_F)
    t = np.clip(((at + _F(0.5) * soft) - y) / soft, _F(0.0), _F(1.0))
    return _smooth(t).astype(_F)


def region_weights(y: np.ndarray, r: ReconstructionRegion) -> tuple[np.ndarray, np.ndarray]:
    """(highlight, shadow) weights at luma y; both 1 when the region is off."""
    y = np.asarray(y, _F)
    if not r.enabled:
        one = np.ones_like(y, dtype=_F)
        return one, one
    if r.band:
        b = region_ramp_up(y, r.band_low, r.band_softness) * region_ramp_down(y, r.band_high, r.band_softness)
        return (b * _F(r.highlight_strength)).astype(_F), (b * _F(r.shadow_strength)).astype(_F)
    hi = region_ramp_up(y, r.highlight_from, r.highlight_softness) * _F(r.highlight_strength)
    lo = region_ramp_down(y, r.shadow_below, r.shadow_softness) * _F(r.shadow_strength)
    return hi.astype(_F), lo.astype(_F)


def reconstruction_gate(y: np.ndarray, shadow_weight: float, mode: str, strength: float,
                        r: ReconstructionRegion) -> np.ndarray:
    """Spec section 2 steps 3 and 4 with revision 2's region, per pixel, float32."""
    if mode not in MODES:
        raise ValueError("mode must be one of " + ", ".join(MODES))
    y = np.asarray(y, _F)
    hp = (_F(1.0) / (_F(1.0) + np.exp(-((y - _F(0.82)) * _F(24.0))))).astype(_F)
    sp = (_F(1.0) / (_F(1.0) + np.exp(-((_F(0.10) - y) * _F(24.0))))).astype(_F) * _F(shadow_weight)
    if r.enabled:
        wh, ws = region_weights(y, r)
        hp, sp = hp * wh, sp * ws
    gate = {"all": np.maximum(hp, sp), "highlights": hp, "shadows": sp,
            "off": np.zeros_like(y, dtype=_F)}[mode]
    return (gate * _F(strength)).astype(_F)
