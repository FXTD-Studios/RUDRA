"""training/aces_full_only.py: the full-only routing arm (1 Nov 2026)."""
import numpy as np
import pytest

from training.aces_full_only import FULL_CODE, full_only_weight, fully_clipped, route_full_only


def _frame():
    src = np.full((32, 32, 3), 0.4, np.float32)
    src[8:24, 8:24] = 1.0            # a fully clipped square
    src[2:6, 2:6, 2] = 1.0           # blue-only clip: partial
    src[26:30, 26:30, :2] = 1.0      # red and green: partial
    rng = np.random.default_rng(1)
    analytic = rng.uniform(0.1, 1.0, src.shape).astype(np.float32)
    pred = analytic * 3.0 + 0.5
    return src, analytic, pred.astype(np.float32)


def test_fully_clipped_is_the_evaluators_definition():
    src, _, _ = _frame()
    full = fully_clipped(src)
    assert full[8:24, 8:24].all()
    assert not full[2:6, 2:6].any() and not full[26:30, 26:30].any()
    assert full.sum() == 16 * 16
    edge = np.zeros((1, 1, 3), np.float32) + FULL_CODE
    assert fully_clipped(edge).all() and not fully_clipped(edge - 1e-6).any()


def test_hard_route_is_exact_on_both_sides():
    src, analytic, pred = _frame()
    out = route_full_only(pred, analytic, src)
    full = fully_clipped(src)
    assert np.array_equal(out[~full], analytic[~full])   # partial and unclipped: bit-exact analytic
    assert np.array_equal(out[full], pred[full])


@pytest.mark.parametrize('feather', [2, 4, 8])
def test_feather_stays_inside_the_full_area(feather):
    src, analytic, pred = _frame()
    out = route_full_only(pred, analytic, src, feather_px=feather)
    full = fully_clipped(src)
    assert np.array_equal(out[~full], analytic[~full])
    w = full_only_weight(src, feather)
    assert w[~full].max() == 0.0 and w.max() == 1.0
    assert 0.0 < w[8, 8] < 1.0 or feather == 1                 # the edge pixel is ramped
    assert w[16, 16] == 1.0                                     # the middle is the model
    # monotone into the area along a row
    row = w[16, 8:16]
    assert np.all(np.diff(row) >= 0)


def test_no_full_clip_means_pure_analytic():
    src = np.full((8, 8, 3), 0.9, np.float32)
    src[..., 0] = 1.0
    analytic = np.ones((8, 8, 3), np.float32)
    out = route_full_only(analytic * 5, analytic, src, feather_px=4)
    assert np.array_equal(out, analytic)


def test_shapes_are_checked():
    src, analytic, pred = _frame()
    with pytest.raises(ValueError):
        route_full_only(pred[:4], analytic, src)
