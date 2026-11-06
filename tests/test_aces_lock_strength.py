"""The lock at partial strength (6 Nov 2026), in its own file beside test_aces_partial_tighten.py."""
import numpy as np

from test_aces_partial_tighten import MixingView, _scene_pixel
from training.aces_partial_tighten import tighten_partial


def test_lock_strength_goes_part_of_the_way():
    v = MixingView()
    x, src = _scene_pixel(v)
    base = x * np.array([0.95, 1.0, 1.0], np.float32)
    w = np.ones((1, 1), np.float32)
    cur = np.array([[[640.0, 2.0 * 1.006, 1.0 * 0.994]]], np.float32)
    full, _ = tighten_partial(src, cur, base, w, v, mode='lock')
    half, d = tighten_partial(src, cur, base, w, v, mode='lock', strength=0.5)
    assert np.allclose(half, np.sqrt(cur * full), rtol=1e-5)   # geometric half-way
    e = lambda o: np.abs(v.forward(o)[0, 0, 1:] - src[0, 0, 1:]).max() * 255
    assert e(full) < e(half) < e(cur)
    same, _ = tighten_partial(src, cur, base, w, v, mode='lock', strength=1.0)
    assert np.array_equal(same, full)                          # strength 1 = the full lock
    assert d['strength'] == 0.5
