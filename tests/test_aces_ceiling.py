"""training/aces_ceiling.py: the Python side of the native highlight ceiling."""
import numpy as np
import pytest

from training.aces_ceiling import LUMA_AP1, cap_luminance_np


def test_numpy_cap_scales_only_what_is_over_and_keeps_hue():
    rgb = np.array([[[1.0, 1.0, 1.0], [400.0, 900.0, 1000.0], [-2.0, 0.5, 0.1]]], np.float32)
    out, n = cap_luminance_np(rgb, 128.0)
    assert n == 1
    y = out.astype(np.float64) @ np.array(LUMA_AP1)
    assert y[0, 1] == pytest.approx(128.0, rel=1e-5)
    assert np.allclose(out[0, 1] / out[0, 1, 2], rgb[0, 1] / rgb[0, 1, 2])   # hue kept
    assert np.array_equal(out[0, 0], rgb[0, 0]) and np.array_equal(out[0, 2], rgb[0, 2])
    assert np.array_equal(cap_luminance_np(rgb, 1e9)[0], rgb)


def test_torch_cap_matches_numpy():
    torch = pytest.importorskip("torch")
    from training.aces_ceiling import cap_luminance_torch
    rng = np.random.default_rng(1)
    hwc = rng.uniform(0, 3000, (5, 7, 3)).astype(np.float32)
    a, na = cap_luminance_np(hwc, 500.0)
    t, nt = cap_luminance_torch(torch.from_numpy(hwc).permute(2, 0, 1)[None], 500.0)
    assert na == nt
    assert np.allclose(t[0].permute(1, 2, 0).numpy(), a, rtol=1e-5)


def test_inverse_white_luma_is_the_views_white():
    pytest.importorskip("torch")
    pytest.importorskip("PyOpenColorIO")
    from training.aces_ceiling import inverse_white_luma
    assert inverse_white_luma() == pytest.approx(128.0, rel=0.02)   # ACES 2.0 SDR 100 nits, measured 3 Nov
