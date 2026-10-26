"""constrain_colour(forbid_new_negative_y=True): no new negative AP1 luminance on support."""
import numpy as np
from training.aces_colour_backtrack import constrain_colour

LUMA = np.array([.2722287168, .6740817658, .0536895174])


class SaturatingView:
    """Toy view: codes = clip(x, 0, 1). Any value >= 1 displays as 1, so a negative
    green on a clipped-blue pixel is invisible to the 1-code display check."""
    def forward(self, x):
        return np.clip(np.asarray(x, np.float32), 0, 1)


def case():
    source = np.full((2, 2, 3), .2, np.float32); source[0, 0] = (0., 0., 1.)    # one saturated-blue support pixel
    baseline = source.copy(); baseline[0, 0] = (0., .01, 5.)
    pred = baseline.copy(); pred[0, 0] = (0., -5., 30.)                         # Y < 0, displays exactly
    weight = np.zeros((2, 2), np.float32); weight[0, 0] = 1
    return source, pred, baseline, weight


def test_guard_off_keeps_negative_luminance():
    s, p, b, w = case()
    out, d = constrain_colour(s, p, b, w, SaturatingView())
    assert out[0, 0] @ LUMA < 0 and d['new_negative_luminance'] == 1 and not d['forbid_new_negative_y']


def test_guard_backtracks_until_nonnegative():
    s, p, b, w = case()
    out, d = constrain_colour(s, p, b, w, SaturatingView(), forbid_new_negative_y=True)
    assert out[0, 0] @ LUMA >= 0 and d['new_negative_luminance'] == 0
    assert d['remaining_display_errors'] == 0 and d['changed_pixels'] == 1
    assert np.array_equal(out[1:], p[1:]) and np.array_equal(out[0, 1], p[0, 1])   # outside support bit-exact


def test_guard_ignores_pixels_whose_baseline_is_negative():
    s, p, b, w = case(); b[0, 0] = (0., -1., 2.)
    out, _ = constrain_colour(s, p, b, w, SaturatingView(), forbid_new_negative_y=True)
    assert np.array_equal(out, p)
