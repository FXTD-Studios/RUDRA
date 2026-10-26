import numpy as np
import pytest

from tools.audit_holdout_panorama_similarity import signature, similarity


def test_exposure_and_yaw_copy_is_detected():
    image = np.random.default_rng(4).uniform(.2, 4, (64, 128, 3)).astype(np.float32)
    score, _ = similarity(signature(image), signature(np.roll(image * 8, 32, axis=1)))
    assert score > .99999


def test_distinct_texture_is_not_a_near_copy():
    rng = np.random.default_rng(5)
    a, b = [rng.uniform(.2, 4, (64, 128, 3)).astype(np.float32) for _ in range(2)]
    assert similarity(signature(a), signature(b))[0] < .98


def test_constant_image_cannot_claim_useful_comparison():
    with pytest.raises(ValueError, match='variation'):
        signature(np.ones((64, 128, 3), np.float32))
