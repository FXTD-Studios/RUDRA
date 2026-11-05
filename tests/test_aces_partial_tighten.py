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


# --- the dequantized lock target (5 Nov 2026) --------------------------------------------

from training.aces_partial_tighten import dequantize_recorded  # noqa: E402


def _smooth_highlight(shift=(0.0, 0.0), noise=0.0, seed=0, n=72):
    """A smooth warm highlight in display space: red clips in the middle, green and blue are
    recorded. `shift` moves the content by a fraction of a pixel; noise moves with it."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    X, Y = xx + shift[0], yy + shift[1]
    r2 = (X - n / 2) ** 2 + (Y - n / 2) ** 2
    d = np.stack([1.15 * np.exp(-r2 / (2 * 22.0 ** 2)) + 0.05,
                  0.92 * np.exp(-r2 / (2 * 20.0 ** 2)) + 0.04,
                  0.80 * np.exp(-r2 / (2 * 24.0 ** 2)) + 0.06], -1)
    if noise:
        g = np.random.default_rng(seed).normal(0, noise / 255, (n + 8, n + 8, 3))
        ix = np.clip(np.rint(X).astype(int) + 4, 0, n + 7)
        iy = np.clip(np.rint(Y).astype(int) + 4, 0, n + 7)
        d = d + g[iy, ix]
    return np.clip(d, 0, 1)


def _recorded_partial(src):
    return (src.max(-1) >= 254.5 / 255) & (src.min(-1) < 254.5 / 255)


def test_dequant_recovers_smooth_values_inside_the_bins():
    truth = _smooth_highlight()
    src = (np.rint(truth * 255) / 255).astype(np.float32)
    dq, used = dequantize_recorded(src)
    rec = src < 254.5 / 255
    sel = _recorded_partial(src)[..., None] & rec
    assert sel.sum() > 50 and used[sel].mean() > 0.5
    err_codes = np.sqrt(np.mean(((src - truth) * 255)[sel] ** 2))
    err_dq = np.sqrt(np.mean(((dq - truth) * 255)[sel] ** 2))
    assert err_dq < 0.75 * err_codes, (err_dq, err_codes)
    assert (np.abs(dq - src) * 255 <= 0.49 + 1e-6).all()          # never leaves the code's bin
    assert np.array_equal(dq[~rec], src[~rec].astype(np.float64))   # clipped channels untouched


def test_dequant_leaves_noisy_content_at_the_codes():
    truth = _smooth_highlight(noise=3.0, seed=3)
    src = (np.rint(truth * 255) / 255).astype(np.float32)
    dq, used = dequantize_recorded(src)
    assert used.mean() < 0.05
    sel = (src < 254.5 / 255) & ~used
    assert np.array_equal(dq[sel], src[sel].astype(np.float64))


def test_dequant_region_is_bit_exact_outside_its_box():
    src = (np.rint(_smooth_highlight() * 255) / 255).astype(np.float32)
    region = np.zeros(src.shape[:2], bool)
    region[30:34, 30:34] = True
    dq, used = dequantize_recorded(src, region=region)
    box = np.zeros_like(region)
    box[6:58, 6:58] = True                                          # 4x4 + the 24 px margin
    assert np.array_equal(dq[~box], src[~box].astype(np.float64)) and not used[~box].any()


def test_dequant_target_moves_less_under_subpixel_motion():
    """The point of it: as content moves by a fraction of a pixel the integer codes flip,
    the dequantized target follows the content instead (aligned difference vs the truth)."""
    def aligned(r0, r1, dx=.5, dy=.25):
        a = (1 - dy) * ((1 - dx) * r1[:-1, :-1] + dx * r1[:-1, 1:]) + dy * ((1 - dx) * r1[1:, :-1] + dx * r1[1:, 1:])
        return r0[:-1, :-1] - a
    t0, t1 = _smooth_highlight((0.0, 0.0)), _smooth_highlight((0.5, 0.25))
    s0, s1 = [(np.rint(t * 255) / 255).astype(np.float32) for t in (t0, t1)]
    d0, d1 = dequantize_recorded(s0)[0], dequantize_recorded(s1)[0]
    sel = (_recorded_partial(s0)[..., None] & (s0 < 254.5 / 255))[:-1, :-1]
    codes = np.mean((aligned(s0 - t0, s1 - t1) * 255)[sel] ** 2)
    dq = np.mean((aligned(d0 - t0, d1 - t1) * 255)[sel] ** 2)
    assert dq < 0.6 * codes, (dq, codes)


def test_lock_dequant_target_settings_and_bounds():
    v = ToyView()
    truth = _smooth_highlight()
    src = (np.rint(truth * 255) / 255).astype(np.float32)
    base = v.inverse(src).astype(np.float32)
    w = np.ones(src.shape[:2], np.float32)
    cur = (v.inverse(np.clip(truth, 0, 0.999)) * 1.004).astype(np.float32)   # the model, drifting a little
    a, da = tighten_partial(src, cur, base, w, v, mode='lock')
    b, db = tighten_partial(src, cur, base, w, v, mode='lock', target='dequant')
    assert db['target'] == 'dequant' and db['dequant_channels'] > 0 and db['changed_pixels'] > 0
    sup = partial_support(src, w)
    rec = (src < 254.5 / 255) & sup[..., None]
    shown = v.forward(b)
    assert (np.abs(shown - src)[rec] * 255 <= 1.0 + 1e-4).all()         # still within 1 code of the source
    ea = np.sqrt(np.mean(((v.forward(a) - truth) * 255)[rec] ** 2))
    eb = np.sqrt(np.mean(((shown - truth) * 255)[rec] ** 2))
    assert eb < ea, (eb, ea)                                               # nearer what the camera saw
    assert np.array_equal(a[~sup], cur[~sup]) and np.array_equal(b[~sup], cur[~sup])
    reg = {'partial_tighten': {'mode': 'lock', 'iters': 4, 'target': 'dequant', 'dq_max_lap': 1.2}}
    assert from_registration(reg) == {'mode': 'lock', 'tol_codes': 0.5, 'iters': 4, 'target': 'dequant',
                                      'dq_iters': 200, 'dq_max_lap': 1.2}
    assert 'target' not in from_registration({'partial_tighten': {'mode': 'lock', 'band_codes': 0.25}})
    with pytest.raises(ValueError):
        tighten_partial(src, cur, base, w, v, mode='bisect', target='dequant')


def test_lock_source_target_is_unchanged_by_the_new_option():
    v = MixingView()
    x, src = _scene_pixel(v)
    base = x * np.array([0.95, 1.0, 1.0], np.float32)
    far = np.array([[[640.0, 2.0 * 1.04, 1.0]]], np.float32)
    w = np.ones((1, 1), np.float32)
    for band in (0.0, 0.25, 0.40):
        o1, _ = tighten_partial(src, far, base, w, v, mode='lock', band_codes=band)
        o2, _ = tighten_partial(src, far, base, w, v, mode='lock', band_codes=band, target='source')
        assert np.array_equal(o1, o2)
