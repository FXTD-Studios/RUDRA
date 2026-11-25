"""1.3.1 (25 Nov 2026): Source "BT.1886 shoulder", the Python twin of native inverse_bt1886_shoulder."""
from __future__ import annotations

import numpy as np

from rudra import source_baseline as sb


def _display(codes):
    c = np.asarray(codes, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def test_it_is_appended_so_stored_indices_keep_their_meaning():
    assert sb.BASELINE_CURVES[:8] == ("unknown", "bt1886", "aces", "hable", "agx", "camera_log", "clip", "bt1886_shadows")
    assert sb.baseline_curve_index("bt1886_shoulder") == 8


def test_bt1886_below_the_knee_and_rising_above_it():
    codes = np.arange(1, 256)
    sh = sb._inverse("bt1886_shoulder", _display(codes))
    bt = sb._inverse("bt1886", _display(codes))
    video = sb._bt709_oetf(np.clip(_display(codes), 0.0, 0.995))
    below = video <= sb.SHOULDER_KNEE_CODE
    np.testing.assert_allclose(sh[below], bt[below], rtol=1e-6)
    assert (np.diff(np.log2(sh)) > 0).all()
    assert 1.85 < np.log2(sh[-2] / bt[-2]) <= 2.0


def test_the_knots_carry_it():
    k = sb.baseline_knots("bt1886_shoulder")
    b = sb.baseline_knots("bt1886")
    assert k.shape == (257,)
    assert abs(float(k[1 + 128] - b[1 + 128])) < 1e-6
    assert float(k[1 + 240] - b[1 + 240]) > 0.8
