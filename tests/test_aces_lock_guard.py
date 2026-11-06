"""The appearance guard on the lock at partial strength (7 Nov 2026)."""
import numpy as np

from test_aces_partial_tighten import MixingView, _scene_pixel
from training.aces_partial_tighten import tighten_partial

FULL = 254.5 / 255


def _bounds_ok(v, out, src):
    d = np.asarray(v.forward(out), np.float64)
    rec = src < FULL
    recorded = np.abs(np.where(rec, d - src, 0.0)).max(-1) * 255 <= 1.0 + 1e-6
    clipped = ~((~rec) & (d < FULL)).any(-1)
    return recorded & clipped


def test_partial_strength_never_leaves_the_bounds_even_where_the_model_was_outside_them():
    # The shipped constraint cannot always bring a pixel back (the engine counts them:
    # remaining_display_errors). The full lock does; a blend part of the way would not.
    v = MixingView()
    rng = np.random.default_rng(5)
    x, src = _scene_pixel(v)
    n = 300
    S = np.repeat(src.reshape(1, 1, 3), n, 1)
    B = np.repeat((x * np.array([0.95, 1.0, 1.0], np.float32)).reshape(1, 1, 3), n, 1)
    drift = np.concatenate([rng.normal(0, 0.3, (1, n, 1)),
                            rng.uniform(0.02, 0.06, (1, n, 2)) * rng.choice([-1, 1], (1, n, 2))], -1)
    cur = (np.repeat(x.reshape(1, 1, 3), n, 1) * np.exp(drift)).astype(np.float32)
    assert (~_bounds_ok(v, cur, S)).sum() > 200                  # the premise: many start outside
    w = np.ones(cur.shape[:2], np.float32)
    full, _ = tighten_partial(S, cur, B, w, v, mode='lock')
    for lam in (0.5, 0.6):
        unguarded = (np.maximum(cur, 1e-9) ** (1 - lam) * np.maximum(full, 1e-9) ** lam).astype(np.float32)
        assert (~_bounds_ok(v, unguarded, S)).sum() > 100        # without the guard: the 6 Nov failure
        out, d = tighten_partial(S, cur, B, w, v, mode='lock', strength=lam)
        assert _bounds_ok(v, out, S).all()                       # with it: never outside
        assert d['strength_raised_pixels'] > 100


def test_the_guard_leaves_a_blend_that_was_already_inside_untouched():
    v = MixingView()
    x, src = _scene_pixel(v)
    base = x * np.array([0.95, 1.0, 1.0], np.float32)
    cur = np.array([[[640.0, 2.0 * 1.006, 1.0 * 0.994]]], np.float32)
    full, _ = tighten_partial(src, cur, base, np.ones((1, 1), np.float32), v, mode='lock')
    half, d = tighten_partial(src, cur, base, np.ones((1, 1), np.float32), v, mode='lock', strength=0.5)
    assert np.allclose(half, np.sqrt(cur * full), rtol=1e-5)    # inside already: exactly lambda
    assert d['strength_raised_pixels'] == 0
