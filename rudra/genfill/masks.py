"""The fill regions: where the 8-bit source holds no information.

    highlight  max(R, G, B) code >= tau_high  (default 250: the model's context around clipped white)
    shadow     max(R, G, B) code <= tau_low   (default -1 = off since v1.1; 4 was v1)

v1.1 (29 Oct 2026, after the v1 gates failed): the model still SEES those regions, but the fill is
BLENDED only where the source holds no information at all: max code >= blend_high (255, fully
clipped) or, with shadows on, <= blend_low (0, fully crushed), with no outward feather. In codes
250-254 and 1-4 the exact inverse is already within ~0.1 stop and v1's fill made them worse
(claude/TASK_GENFILL_LAYER3_2026_10_27.md).

The user moves tau_high / tau_low in the app (the live mask overlay shows the result). Each
mask is dilated a little (the model needs the context ring), then steadied in time with a
per-pixel exponential moving average (alpha 0.7) and thresholded at one half, so a region
does not blink on and off between frames. `feather` gives the soft weight the composite uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TAU_HIGH_DEFAULT = 250
TAU_LOW_DEFAULT = -1          # v1.1: shadows off by default (v1: 4)
BLEND_HIGH_DEFAULT = 255
BLEND_LOW_DEFAULT = 0


@dataclass
class MaskSettings:
    tau_high: int = TAU_HIGH_DEFAULT      # 8-bit code; >= is a highlight to fill (256 = none)
    tau_low: int = TAU_LOW_DEFAULT        # 8-bit code; <= is a shadow to fill (-1 = none)
    dilate: int = 4                       # pixels, at the frame's own resolution
    ema: float = 0.7                      # weight of the previous frame's mask
    feather: int = 0                      # pixels of soft edge for the composite (v1: 6, outward)
    blend_high: int = BLEND_HIGH_DEFAULT  # the fill is blended only at max code >= this (256 = never)
    blend_low: int = BLEND_LOW_DEFAULT    # ... and, with shadows on, at max code <= this
    max_gain_stops: float = 3.0           # v2: the fill's luminance at most this many stops over the master (0 = no cap)

    def check(self):
        if not (-1 <= self.tau_low < self.tau_high <= 256):
            raise ValueError(f'need -1 <= tau_low < tau_high <= 256 (got {self.tau_low}, {self.tau_high})')
        if not (0 <= self.blend_high <= 256 and -1 <= self.blend_low <= 255):
            raise ValueError(f'need 0 <= blend_high <= 256 and -1 <= blend_low <= 255 (got {self.blend_high}, {self.blend_low})')
        if not (0.0 <= self.max_gain_stops <= 20.0):
            raise ValueError('max_gain_stops must be in [0, 20]')
        if not (0.0 <= self.ema < 1.0):
            raise ValueError('ema must be in [0, 1)')
        return self


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    if r <= 0 or not m.any():
        return m
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def frame_masks(codes: np.ndarray, s: MaskSettings) -> tuple[np.ndarray, np.ndarray]:
    """codes: (H, W, 3) uint8 SDR codes -> (highlight, shadow) bool masks, dilated."""
    mx = codes.max(-1)
    hi = mx >= s.tau_high if s.tau_high <= 255 else np.zeros(mx.shape, bool)
    lo = mx <= s.tau_low if s.tau_low >= 0 else np.zeros(mx.shape, bool)
    hi, lo = _dilate(hi, s.dilate), _dilate(lo, s.dilate)
    return hi, lo & ~hi


class TemporalMask:
    """Per-pixel EMA of the masks across frames (one instance per shot, frames in order)."""

    def __init__(self, s: MaskSettings):
        self.s = s.check()
        self._hi = self._lo = None

    def __call__(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hi, lo = frame_masks(codes, self.s)
        hi, lo = hi.astype(np.float32), lo.astype(np.float32)
        if self._hi is not None and self._hi.shape == hi.shape and self.s.ema > 0:
            a = self.s.ema
            hi = a * self._hi + (1 - a) * hi
            lo = a * self._lo + (1 - a) * lo
        self._hi, self._lo = hi, lo
        h, l = hi >= 0.5, lo >= 0.5
        return h, l & ~h


def blend_core(codes: np.ndarray, s: MaskSettings) -> np.ndarray:
    """codes (H, W, 3) uint8 -> bool: where the fill may be blended (no dilation, no EMA)."""
    mx = codes.max(-1)
    core = mx >= s.blend_high if s.blend_high <= 255 else np.zeros(mx.shape, bool)
    if s.tau_low >= 0 and s.blend_low >= 0:
        core = core | (mx <= s.blend_low)
    return core


def feather(mask: np.ndarray, radius: int) -> np.ndarray:
    """bool (H, W) -> float32 weight in [0, 1]: 1 inside, a smooth fall-off over `radius` px outside."""
    m = mask.astype(np.float32)
    if radius <= 0 or not mask.any():
        return m
    import cv2
    grown = _dilate(mask, radius).astype(np.float32)
    soft = cv2.GaussianBlur(grown, (0, 0), radius / 2.0)
    return np.maximum(m, np.clip(soft, 0, 1)) * grown
