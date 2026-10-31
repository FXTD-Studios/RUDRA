"""v3: the brightness estimator and the shape / brightness split (rudra/genfill/brightness.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip('PyOpenColorIO')

from rudra.genfill.brightness import (AP1_Y, FEATURES, BrightnessEstimator, BrightnessSettings, component_features,
                                      components, compose, excess_map, frame_features, log2_lum)


def _scene(h=64, w=96):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.full((h, w, 3), 0.2, np.float32) + 0.3 * (xx / w)[..., None]
    sun = 300 * np.exp(-((xx - 30) ** 2 + (yy - 20) ** 2) / 40.0)
    img += sun[..., None]
    img[45:55, 60:80] = 500.0                                   # a second, flat clipped patch
    return img


def _codes_master(img):
    from rudra.aces_view import ACES2View
    v = ACES2View()
    codes = np.rint(np.clip(v.forward(img.reshape(-1, 3)).reshape(img.shape), 0, 1) * 255).astype(np.uint8)
    master = v.inverse((codes / 255.0).astype(np.float32).reshape(-1, 3)).reshape(img.shape).astype(np.float32)
    return codes, master


def test_components_and_features():
    codes, master = _codes_master(_scene())
    lab, ids = components(codes)
    assert len(ids) == 2
    fr = frame_features(codes, master)
    f = component_features(codes, master, lab == ids[0], fr)
    assert f.shape == (len(FEATURES),) and np.isfinite(f).all()
    assert 0 <= f[FEATURES.index('all_three')] <= 1 and 0 < f[FEATURES.index('compact')] <= 1


def test_estimator_learns_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, len(FEATURES))).astype(np.float32)
    y = np.maximum(0, 1.5 + 0.8 * X[:, 0] - 0.5 * X[:, 4]).astype(np.float32)
    est = BrightnessEstimator().fit(X, y, epochs=150, seed=1)
    p = est.predict(X)
    assert np.mean(np.abs(p - y)) < 0.5 * np.mean(np.abs(y - y.mean()))
    assert (p >= 0).all()
    est.save(tmp_path / 'b.json')
    est2 = BrightnessEstimator.load(tmp_path / 'b.json')
    assert np.allclose(est2.predict(X[:10]), p[:10], atol=1e-6)
    # asymmetric: a high over_weight biases the estimate low
    lo = BrightnessEstimator().fit(X, y + rng.normal(0, 0.7, size=y.shape).astype(np.float32), over_weight=8, epochs=150, seed=1)
    hi = BrightnessEstimator().fit(X, y + rng.normal(0, 0.7, size=y.shape).astype(np.float32), over_weight=0.125, epochs=150, seed=1)
    assert lo.predict(X).mean() < hi.predict(X).mean()


def test_compose_sets_the_mean_brightness_and_keeps_the_shape():
    codes, master = _codes_master(_scene())
    gen = master * np.float32(3.0)
    gen[18:23, 28:33] *= 4.0                                   # the model's own shape: a hot core
    b_map, lab = excess_map(codes, master, None, offset=1.5)   # b = 0 + the artist's 1.5 stops
    out = compose(master, gen, b_map, lab, shape_gain=1.0)
    clipped = np.isfinite(b_map)
    assert np.array_equal(out[~clipped], master[~clipped])     # nothing outside the clipped areas
    lm, lo = log2_lum(master), log2_lum(out)
    for c in np.unique(lab[clipped]):
        s = lab == c
        assert (lo[s] - lm[s]).mean() == pytest.approx(1.5, abs=0.15)   # the brightness is the estimate
    s0 = lab == lab[20, 30]
    assert lo[20, 30] - lm[20, 30] > (lo[s0] - lm[s0]).mean()           # the hot core is still there
    flat = compose(master, gen, b_map, lab, shape_gain=0.0)
    lf = log2_lum(flat)
    assert np.allclose(lf[clipped] - lm[clipped], 1.5, atol=1e-3)


def test_fill_master_v3_uses_the_estimate_and_keeps_the_contract(tmp_path):
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    from rudra.genfill.masks import MaskSettings
    from rudra.genfill.model import FillSettings
    from rudra.genfill.run import fill_master, read_frame
    from rudra.genfill.synthetic import write_master
    view = ACES2View()
    master = tmp_path / 'm'
    write_master(master, n=5)
    out = tmp_path / 'o'
    fill_master(master, out, tiny_fill(), view, FillSettings(steps=2, seed=1), MaskSettings(dilate=2), window=5, overlap=2,
                max_area=48 * 64, brightness=BrightnessSettings(None, offset=1.0, shape_gain=0.0))
    rep = __import__('json').loads((out / 'genfill.json').read_text())
    assert rep['brightness']['offset'] == 1.0
    for f in sorted(master.glob('*.exr')):
        a, b = read_frame(f), read_frame(out / f.name)
        codes = lambda x: np.rint(np.clip(view.forward(x['ap1']), 0, 1) * 255)
        assert (codes(a) == codes(b)).all()
        gen = b['channels']['rudra.generated'] > 0
        if gen.any():
            d = log2_lum(b['ap1'])[gen] - log2_lum(a['ap1'])[gen]
            assert np.median(d) == pytest.approx(1.0, abs=0.2)          # b = 0 + 1 stop, no shape
