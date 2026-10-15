import cv2
import numpy as np

from pipeline.audit_candidate_quality import alignment_proxy, sample_rows


def test_sampling_is_order_independent_and_covers_shot_ends():
    rows = [dict(scene_id="a", frame_index=i, asset_id=str(i)) for i in range(10)]
    assert [r["frame_index"] for r in sample_rows(rows[::-1])] == [0, 5, 9]
    assert sample_rows(rows[:1]) == rows[:1]


def test_alignment_flags_displacement_but_accepts_matched_structure():
    g = np.random.default_rng(12)
    image = cv2.GaussianBlur(g.uniform(.1, .9, (144, 256)).astype(np.float32), (0, 0), 3)
    sdr = np.repeat(image[:, :, None], 3, axis=2)
    hdr = 2 ** (image * 8)
    assert not alignment_proxy(sdr, hdr)["review_flag"]
    assert alignment_proxy(sdr, np.roll(hdr, 12, axis=1))["review_flag"]


def test_uniform_frame_does_not_claim_alignment():
    assert alignment_proxy(np.ones((144, 256, 3), np.float32), np.ones((144, 256), np.float32))["review_flag"]
