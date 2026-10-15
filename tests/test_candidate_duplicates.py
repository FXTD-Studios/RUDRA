import cv2
import numpy as np
import pytest

from pipeline.audit_candidate_duplicates import fingerprint, screen


def rec(image, name, split):
    return dict(asset_id=name, split=split, **fingerprint(image))


def test_resized_brightness_variant_is_flagged_across_splits():
    image = np.random.default_rng(10).uniform(0.1, 0.7, (128, 128, 3)).astype(np.float32)
    image = cv2.GaussianBlur(image, (0, 0), 4)
    # Add spatial structure so it is not a low-information sample.
    image[:64, :64] *= 0.25
    other = cv2.resize(image, (256, 256)) * 0.8
    matches, _ = screen([rec(image, "a", "train"), rec(other, "b", "test")])
    assert len(matches) == 1


def test_within_split_matches_are_not_leakage():
    image = np.random.default_rng(2).random((32, 32, 3)).astype(np.float32)
    assert not screen([rec(image, "a", "train"), rec(image, "b", "train")])[0]


def test_black_frames_are_reported_separately():
    matches, low = screen([rec(np.zeros((32, 32, 3), np.uint8), "black", "test")])
    assert matches == [] and low == ["black"]


def test_invalid_pixels_refused():
    with pytest.raises(ValueError): fingerprint(np.full((32, 32, 3), np.nan))


def test_different_structure_is_not_flagged():
    g = np.random.default_rng(3)
    records = [rec(g.random((32, 32, 3)).astype(np.float32), str(i), "train" if i == 0 else "test") for i in range(2)]
    assert not screen(records)[0]
