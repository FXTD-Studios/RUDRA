import numpy as np
from training.aces_clipped_smoothing import smooth_clipped, LUMA_AP1


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    source = rng.uniform(.1, .8, (64, 96, 3)).astype(np.float32)
    source[16:48, 20:70] = 1.0              # fully clipped block
    source[10:14, 10:14, 0] = 1.0           # partial clip only (not full)
    pred = rng.uniform(.5, 4.0, (64, 96, 3)).astype(np.float32)
    return source, pred


def test_outside_full_support_bit_exact():
    source, pred = _scene()
    for mode in ('gaussian', 'hybrid', 'component'):
        out, full = smooth_clipped(pred, source, sigma=16, mode=mode)
        assert np.array_equal(out[~full], pred[~full])
        assert np.isfinite(out).all() and out.dtype == np.float32


def test_hybrid_interior_exactly_flat_rgb():
    source, pred = _scene(1)
    out, full = smooth_clipped(pred, source, sigma=64, mode='hybrid')
    interior = out[19:45, 23:67]            # eroded by >2 px
    assert np.all(interior == interior[0, 0])


def test_gaussian_reduces_interior_gradient_energy():
    source, pred = _scene(2)
    out, _ = smooth_clipped(pred, source, sigma=64)
    g = lambda a: np.abs(np.diff(np.log2(a[20:44, 24:66] @ LUMA_AP1), axis=1)).mean()
    assert g(out) < .1 * g(pred)


def test_no_full_clip_is_noop():
    source, pred = _scene(3)
    source = np.minimum(source, .9)
    out, full = smooth_clipped(pred, source, mode='hybrid')
    assert not full.any() and np.array_equal(out, pred)
