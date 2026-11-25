"""1.3.1 (25 Nov 2026): flicker inside the fill, measured on every fill for the QC report (gate T of the shot check)."""
from __future__ import annotations

import numpy as np

from rudra.genfill.run import FLICKER_LIMIT_EV, FlickerMeter


def frame(level: float, clipped: bool = True, h: int = 32, w: int = 32):
    codes = np.full((h, w, 3), 120, np.uint8)
    pic = np.full((h, w, 3), 0.5, np.float32)
    if clipped:
        codes[8:24, 8:24] = 255
        pic[8:24, 8:24] = level
    return codes, pic


def test_a_steady_fill_measures_zero_and_passes():
    m = FlickerMeter()
    for _ in range(5):
        m.add(*frame(4.0))
    r = m.report()
    assert r['fill_flicker_p95_ev'] <= 0.002 and r['fill_flicker_pass'] is True and r['fill_flicker_pixels'] == 4 * 256


def test_a_fill_that_pumps_half_a_stop_fails():
    m = FlickerMeter()
    for k in range(6):
        m.add(*frame(4.0 * (2 ** 0.5 if k % 2 else 1.0)))
    r = m.report()
    assert abs(r['fill_flicker_p95_ev'] - 0.5) < 0.002 and r['fill_flicker_pass'] is False
    assert r['fill_flicker_limit_ev'] == FLICKER_LIMIT_EV == 0.10


def test_moving_or_unclipped_pixels_are_not_counted():
    m = FlickerMeter()
    m.add(*frame(4.0, clipped=False))
    m.add(*frame(8.0, clipped=False))
    assert m.report()['fill_flicker_p95_ev'] is None and m.report()['fill_flicker_pass'] is None
    m = FlickerMeter()
    a, b = frame(4.0), frame(8.0)
    b[0][8:24, 8:24] = 200              # the source moved: not a still clipped pixel
    m.add(*a)
    m.add(*b)
    assert m.report()['fill_flicker_pixels'] == 0
