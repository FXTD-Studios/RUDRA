"""1.3.1 (25 Nov 2026): the colour carry reads the SDR through the Source curve, as the anchor does.

Under BT.1886 the anchor set each pixel's level from code^2.4 but ``carry_source_chroma`` took its
colour from the sRGB reading, so analytic masters were 16 % less saturated in the mid-tones and 38 %
in the shadows than the SDR on a gamma 2.4 display (04.mp4, 24 Nov 2026). With the anchor's knots the
carried channel ratios are the Source curve's own; without them nothing changes.
"""
from __future__ import annotations

import numpy as np

from rudra.chroma import (LUMA_REC2020, carry_source_chroma, srgb_to_linear,
                          target_factor_log2)


def bt1886_knots() -> np.ndarray:
    """As native anchor_target_knots gives them for Source BT.1886 (code^2.4 over the sRGB reading)."""
    c = np.arange(256, dtype=np.float64) / 255.0
    k = np.zeros(256, dtype=np.float32)
    k[1:] = np.log2(c[1:] ** 2.4 / srgb_to_linear(c[1:])).astype(np.float32)
    k[0] = k[1]
    return k


def frame(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sdr = rng.integers(10, 240, size=(24, 32, 3)).astype(np.float64) / 255.0   # all below the knee
    hdr = rng.uniform(1.0, 150.0, size=(24, 32, 3))
    return sdr, hdr


def test_without_knots_nothing_changes():
    sdr, hdr = frame()
    np.testing.assert_array_equal(carry_source_chroma(hdr, sdr), carry_source_chroma(hdr, sdr, target_log2=None))


def test_flat_knots_are_the_srgb_reading():
    sdr, hdr = frame()
    np.testing.assert_allclose(carry_source_chroma(hdr, sdr, target_log2=np.zeros(256, np.float32)),
                               carry_source_chroma(hdr, sdr), rtol=0, atol=1e-12)


def test_bt1886_knots_carry_the_bt1886_ratios_and_keep_the_luma():
    sdr, hdr = frame()
    out = carry_source_chroma(hdr, sdr, target_log2=bt1886_knots())
    # On the knots (8-bit codes) the light is code^2.4 to float32 precision.
    want = sdr ** 2.4
    got_ratio = out / out.sum(-1, keepdims=True)
    want_ratio = want / want.sum(-1, keepdims=True)
    np.testing.assert_allclose(got_ratio, want_ratio, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(out @ LUMA_REC2020, hdr @ LUMA_REC2020, rtol=1e-5)   # eps=1e-6 in the carry


def test_bt1886_is_more_saturated_than_the_srgb_reading():
    sdr = np.array([[[0.30, 0.20, 0.15], [0.60, 0.45, 0.35]]])   # a dark and a mid skin tone
    hdr = np.full_like(sdr, 40.0)
    old = carry_source_chroma(hdr, sdr)
    new = carry_source_chroma(hdr, sdr, target_log2=bt1886_knots())
    sat = lambda x: (x.max(-1) - x.min(-1)) / x.max(-1)
    assert (sat(new) > sat(old) + 0.03).all(), (sat(old), sat(new))


def test_knot_interpolation_matches_the_native_rule():
    knots = np.linspace(-1.0, 1.0, 256).astype(np.float32)
    codes = np.array([0.0, 0.5 / 255.0, 0.5, 1.0, 1.5, -0.2], dtype=np.float32)
    got = target_factor_log2(codes, knots)
    want = np.interp(np.clip(codes, 0, 1) * 255.0, np.arange(256), knots)
    np.testing.assert_allclose(got, want, atol=1e-6)
