"""training/aces_partial_tighten.py: half-code appearance on partial highlights."""
import numpy as np
import pytest

from training.aces_partial_tighten import from_registration, partial_support, tighten_partial


class ToyView:
    """Pointwise, monotone, invertible: display = x / (1 + x)."""

    def forward(self, x):
        x = np.maximum(np.asarray(x, np.float32), 0)
        return x / (1 + x)

    def inverse(self, y):
        y = np.minimum(np.asarray(y, np.float32), 0.999)
        return y / (1 - y)


def _frame():
    v = ToyView()
    src = np.full((8, 8, 3), 0.3, np.float32)
    src[0:4, 0:4] = [0.95, 0.8, 0.7]        # partial highlights
    src[4:8, 4:8] = 1.0                     # fully clipped
    base = v.inverse(src).astype(np.float32)
    w = np.ones((8, 8), np.float32)
    cur = base.copy()
    # Partial pixels 0.9 code off in display: inside 1 code, outside half a code.
    cur[0:4, 0:4] = v.inverse(src[0:4, 0:4] + 0.9 / 255)
    cur[4:8, 4:8] = base[4:8, 4:8] * 3      # fully clipped: untouched by the arm
    return v, src, base, w, cur


@pytest.mark.parametrize('mode', ['halving', 'bisect'])
def test_partial_pixels_reach_the_tolerance(mode):
    v, src, base, w, cur = _frame()
    out, d = tighten_partial(src, cur, base, w, v, mode=mode, tol_codes=0.5)
    err = np.abs(v.forward(out) - src).max(-1) * 255
    assert err[0:4, 0:4].max() <= 0.5 + 1e-4
    assert d['remaining_errors'] == 0 and d['changed_pixels'] == 16
    assert np.array_equal(out[4:8, 4:8], cur[4:8, 4:8])   # fully clipped: bit-exact
    assert np.array_equal(out[6:8, 0:2], cur[6:8, 0:2])   # outside support: bit-exact


def test_bisect_keeps_more_of_the_prediction_than_halving():
    v, src, base, w, cur = _frame()
    h, _ = tighten_partial(src, cur, base, w, v, mode='halving')
    b, _ = tighten_partial(src, cur, base, w, v, mode='bisect', iters=14)
    dist = lambda x: np.abs(x[0:4, 0:4] - base[0:4, 0:4]).mean()
    assert dist(b) >= dist(h) - 1e-7


def test_bisect_moves_continuously_with_the_input():
    # The temporal hypothesis: a tiny change in the prediction gives a tiny change
    # in the output with bisection; halving can jump by a factor of two.
    v, src, base, w, cur = _frame()
    jumps = {}
    for mode in ('halving', 'bisect'):
        outs = []
        for k in range(40):
            c = cur.copy()
            c[0:4, 0:4] = v.inverse(src[0:4, 0:4] + (0.6 + 0.01 * k) / 255)
            outs.append(tighten_partial(src, c, base, w, v, mode=mode, iters=14)[0][0, 0, 0])
        jumps[mode] = np.abs(np.diff(outs)).max()
    assert jumps['bisect'] < jumps['halving'] / 4


def test_within_tolerance_is_untouched_and_off_by_default():
    v, src, base, w, cur = _frame()
    cur[0:4, 0:4] = v.inverse(src[0:4, 0:4] + 0.3 / 255)
    out, d = tighten_partial(src, cur, base, w, v)
    assert np.array_equal(out, cur) and d['changed_pixels'] == 0
    assert from_registration({}) is None
    assert from_registration({'partial_tighten': {'mode': 'halving'}})['mode'] == 'halving'
    assert partial_support(src, w)[0, 0] and not partial_support(src, w)[5, 5]
