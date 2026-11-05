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


class MixingView:
    """Not separable, like ACES: a channel-mixing matrix, then a per-channel tone curve."""
    M = np.array([[0.90, 0.08, 0.02], [0.02, 0.96, 0.02], [0.01, 0.04, 0.95]], np.float64)

    def forward(self, x):
        y = np.maximum(np.asarray(x, np.float64), 0) @ self.M.T
        return (y / (1 + y)).astype(np.float32)

    def inverse(self, d):
        d = np.minimum(np.asarray(d, np.float64), 0.999)
        return ((d / (1 - d)) @ np.linalg.inv(self.M).T).astype(np.float32)


def _scene_pixel(v):
    """A partially clipped pixel made as a camera makes it: a real scene colour, viewed, quantised."""
    x = np.array([[[600.0, 2.0, 1.0]]], np.float32)         # a warm highlight: red far past the clip
    src = (np.rint(v.forward(x) * 255) / 255).astype(np.float32)
    assert src[0, 0, 0] >= 254.5 / 255 and src[0, 0, 1] < 254.5 / 255 and src[0, 0, 2] < 254.5 / 255
    return x, src


def test_lock_holds_recorded_channels_and_keeps_the_clipped_one():
    v = MixingView()
    x, src = _scene_pixel(v)
    base = x * np.array([0.95, 1.0, 1.0], np.float32)       # the plain inverse: red only up to the clip
    assert v.forward(base)[0, 0, 0] >= 254.5 / 255
    w = np.ones((1, 1), np.float32)
    # The model: red rebuilt to 640 (the most green's recorded 238 allows here is ~667: red
    # leaks into green's display), green and blue drifting by half a percent.
    cur = np.array([[[640.0, 2.0 * 1.006, 1.0 * 0.994]]], np.float32)
    out, d = tighten_partial(src, cur, base, w, v, mode='lock')
    disp = v.forward(out)[0, 0]
    assert np.abs(disp[1:] - src[0, 0, 1:]).max() * 255 < 0.05   # recorded: exactly the camera's codes
    assert disp[0] >= 254.5 / 255                                # clipped: still clipped
    # Green's recorded code pins red (it leaks into green): the rebuild lands near the true 600,
    # nearer than the model's 640 or the baseline's 570.
    assert abs(out[0, 0, 0] - 600) < min(abs(640 - 600), abs(570 - 600))
    assert d['changed_pixels'] == 1 and d['kept_pixels'] == 0
    # outside support, bit-exact
    big = np.concatenate([src, np.full((1, 1, 3), 0.3, np.float32)], 1)
    bb = np.concatenate([base, v.inverse(np.full((1, 1, 3), 0.3, np.float32))], 1)
    cc = np.concatenate([cur, bb[:, 1:] * 1.01], 1)
    o2, _ = tighten_partial(big, cc, bb, np.ones((1, 2), np.float32), v, mode='lock')
    assert np.array_equal(o2[0, 1], cc[0, 1])


def test_lock_is_continuous_in_the_input():
    v = MixingView()
    x, src = _scene_pixel(v)
    base = x * np.array([0.95, 1.0, 1.0], np.float32)
    w = np.ones((1, 1), np.float32)
    outs = [tighten_partial(src, np.array([[[640.0, 2.0 * (1.0 + 0.0002 * k), 1.0]]], np.float32), base, w, v, mode='lock')[0][0, 0]
            for k in range(30)]
    o = np.array(outs, np.float64)
    rel = (np.abs(np.diff(o, axis=0)) / np.abs(o[1:])).max()
    assert rel < 1e-3   # no jumps: a tiny change in the input, a tiny change out (relative)


def test_lock_brings_an_impossible_rebuild_down_to_what_the_recorded_channels_allow():
    v = MixingView()
    x, src = _scene_pixel(v)
    base = x * np.array([0.95, 1.0, 1.0], np.float32)
    cur = np.array([[[1200.0, 2.0, 1.0]]], np.float32)   # red so bright green could not have shown 238
    out, d = tighten_partial(src, cur, base, np.ones((1, 1), np.float32), v, mode='lock')
    disp = v.forward(out)[0, 0]
    assert d['changed_pixels'] == 1
    assert np.abs(disp[1:] - src[0, 0, 1:]).max() * 255 < 0.05   # the recorded channels display as recorded
    assert disp[0] >= 254.5 / 255 and out[0, 0, 0] < 700    # red still clipped, down to what green allows


def test_lock_band_leaves_values_inside_it_and_brings_outside_ones_to_the_edge():
    v = MixingView()
    x, src = _scene_pixel(v)
    base = x * np.array([0.95, 1.0, 1.0], np.float32)
    w = np.ones((1, 1), np.float32)
    d0 = v.forward(x)[0, 0]
    # A model value 0.1 code off in green: inside a 0.25 band, untouched (bit-exact).
    small = x.copy()
    small[0, 0, 1] = small[0, 0, 1] * (1 + 0.1 / 255 / (d0[1] * (1 - d0[1])) * 0.9)
    out, d = tighten_partial(src, small, base, w, v, mode='lock', band_codes=0.25)
    if np.abs(v.forward(small)[0, 0, 1:] - src[0, 0, 1:]).max() * 255 <= 0.25:
        assert np.array_equal(out, small) and d['inside_band_pixels'] == 1
    # A model value 2 codes off: brought to the band's edge, not to the centre.
    far = np.array([[[640.0, 2.0 * 1.04, 1.0]]], np.float32)
    out, d = tighten_partial(src, far, base, w, v, mode='lock', band_codes=0.25)
    e = np.abs(v.forward(out)[0, 0, 1:] - src[0, 0, 1:]).max() * 255
    assert 0.15 <= e <= 0.26, e
    # band 0 is the exact lock, as registered on 5 Nov
    out0, _ = tighten_partial(src, far, base, w, v, mode='lock')
    assert np.abs(v.forward(out0)[0, 0, 1:] - src[0, 0, 1:]).max() * 255 < 0.05
