import numpy as np
import pytest

from training.temporal_stability import (detect_cuts, evaluate_sequence, pq_encode,
                                         reset_check)


def texture(h=96, w=128, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1.0, (h // 4, w // 4))
    import cv2
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    return (20.0 + 300.0 * np.clip(img, 0, 1)).astype(np.float32)


def panning(n=5, shift=2, seed=0):
    big = texture(96, 128 + n * shift + 8, seed)
    return [big[:, t * shift:t * shift + 128].copy() for t in range(n)]


def test_identical_output_has_zero_excess():
    ref = panning()
    r = evaluate_sequence([f.copy() for f in ref], ref, cuts=[])
    s = r['summary']
    assert s['pairs_scored'] == 4
    assert abs(s['excess']['mean']) < 1e-6 and abs(s['excess']['max']) < 1e-6


def test_flicker_detected_as_positive_excess():
    ref = panning()
    out = [f * (1.0 + (0.15 if t % 2 else 0.0)) for t, f in enumerate(ref)]
    s = evaluate_sequence(out, ref, cuts=[])['summary']
    assert s['pairs_scored'] >= 3
    assert s['excess']['mean'] > 0.005


def test_constant_gain_is_not_flicker():
    ref = panning()
    out = [f * 1.3 for f in ref]
    flick = evaluate_sequence([f * (1.3 if t % 2 else 1.0) for t, f in enumerate(ref)], ref, cuts=[])
    steady = evaluate_sequence(out, ref, cuts=[])
    assert steady['summary']['excess']['mean'] < 0.25 * flick['summary']['excess']['mean']


def test_cut_detected_and_not_scored():
    a, b = panning(3, seed=1), [f * 0.02 + 0.01 for f in panning(3, seed=2)]
    seq = a + b
    assert detect_cuts([pq_encode(f) for f in seq]) == [3]
    r = evaluate_sequence(seq, seq)
    assert r['summary']['cuts'] == [3]
    assert next(p for p in r['pairs'] if p['t'] == 3) == {'t': 3, 'cut': True}


def test_flat_frames_are_skipped_not_scored():
    flat = [np.full((64, 64), 100.0, np.float32) for _ in range(3)]
    s = evaluate_sequence(flat, flat, cuts=[])['summary']
    assert s['pairs_scored'] == 0 and s['pairs_skipped'] == 2
    assert s['excess'] == {'pairs': 0}


def test_highlight_excess_reported_for_bright_regions():
    ref = panning()
    out = [f.copy() for f in ref]
    for t in range(len(out)):
        out[t] = np.where(ref[t] > 203, ref[t] * (1.0 + 0.3 * (t % 2)), ref[t])
    s = evaluate_sequence(out, ref, cuts=[])['summary']
    assert s['highlight_excess']['pairs'] >= 3
    assert s['highlight_excess']['mean'] > s['excess']['mean'] > 0


@pytest.mark.parametrize('bad', ['nan', 'neg', 'shape', 'short'])
def test_invalid_inputs_rejected(bad):
    ref = panning(3)
    out = [f.copy() for f in ref]
    if bad == 'nan':
        out[1][0, 0] = np.nan
    elif bad == 'neg':
        out[1][0, 0] = -1
    elif bad == 'shape':
        out[1] = out[1][:-1]
    else:
        out, ref = out[:1], ref[:1]
    with pytest.raises(ValueError):
        evaluate_sequence(out, ref, cuts=[])


def test_length_mismatch_and_bad_cut_rejected():
    ref = panning(3)
    with pytest.raises(ValueError):
        evaluate_sequence(ref[:2], ref, cuts=[])
    with pytest.raises(ValueError):
        evaluate_sequence(ref, ref, cuts=[5])


def test_reset_check():
    f = texture()
    assert reset_check(f, f)['reset_ok']
    assert not reset_check(f, f * 1.5)['reset_ok']
