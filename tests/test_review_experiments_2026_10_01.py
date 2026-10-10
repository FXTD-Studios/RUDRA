"""Pure-function checks for the 1 Oct 2026 review experiments (no model, no GPU)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from training import bench_hdrtv1k as tv
from training.audit_exposure_control import verdict
from training.cvvdp_display_check import pu21_variants


def test_pq_white_round_trip_is_the_hdrtv1k_convention():
    from rudra.hdr10 import pq_oetf
    code = tv.pq_to_code(pq_oetf(np.array([203.0, 1000.0, 10000.0])))
    assert code[-1] == round(tv.PQ_WHITE_CODE)          # 10,000 nits -> 65282/65283, not 65535
    nits = tv.code_to_nits(code)
    assert np.allclose(nits, [203.0, 1000.0, 10000.0], rtol=2e-3)


def test_identical_frames_score_perfectly():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 60000, size=(64, 80, 3), dtype=np.uint16)
    s = tv.score_pair(img, img)
    assert s["psnr_pq"] == float("inf")
    assert s["ssim_pq"] == pytest.approx(1.0, abs=1e-6)
    assert s["de_itp"] == pytest.approx(0.0, abs=1e-9)


def test_delta_e_itp_grows_with_error_and_is_symmetric():
    ref = np.full((8, 8, 3), 100.0)
    small, large = ref * 1.02, ref * 1.2
    assert 0 < tv.delta_e_itp(small, ref) < tv.delta_e_itp(large, ref)
    assert tv.delta_e_itp(small, ref) == pytest.approx(tv.delta_e_itp(ref, small))


def test_gain_sign_follows_metric_direction():
    better = dict(psnr_pq=40.0, de_itp=2.0)
    worse = dict(psnr_pq=38.0, de_itp=3.0)
    assert tv.paired_gain(better, worse, "psnr_pq") > 0
    assert tv.paired_gain(better, worse, "de_itp") > 0


def test_scene_bootstrap_is_wider_than_frame_bootstrap_for_correlated_frames():
    rng = np.random.default_rng(1)
    scene_effect = np.repeat(rng.normal(0, 1, 20), 10)       # 20 scenes x 10 near-identical frames
    values = scene_effect + rng.normal(0, 0.05, 200)
    scenes = [f"s{i // 10}" for i in range(200)]
    frame = tv.bootstrap_ci(values, None, n=2000)
    scene = tv.bootstrap_ci(values, scenes, n=2000)
    assert scene["units"] == 20 and frame["units"] == 200
    assert (scene["hi"] - scene["lo"]) > 2 * (frame["hi"] - frame["lo"])


def test_collect_matches_stems_and_refuses_ambiguity(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    img = np.zeros((4, 4, 3), np.uint16)
    tv.write_u16_rgb(src / "001_HDR.png", img)
    tv.write_u16_rgb(src / "002.png", img)
    assert tv.collect(src, dst, ["001", "002"]) == 2
    tv.write_u16_rgb(src / "001_other.png", img)
    with pytest.raises(FileNotFoundError):
        tv.collect(src, dst, ["001"])


def test_pu21_in_range_ignores_reference_pixels_above_10000_nits():
    ref = np.full((10, 10, 3), 100.0)
    ref[0, 0] = 50000.0
    test = ref.copy()
    test[0, 0] = 1000.0                                        # a huge error above PU21's range...
    v = pu21_variants(test, ref)
    assert v["pu21_in_range"] == float("inf")                  # ...that in-range scoring cannot see
    assert v["ref_above_10k"] == pytest.approx(0.01)


def _m(shadow_clean, shadow_degraded):
    return dict(deltas=dict(clean=dict(shadow=shadow_clean), degraded=dict(shadow=shadow_degraded)))


def test_h7_verdict_rule():
    control = [_m(-0.03, -0.03)] * 10
    assert verdict(control, [_m(-0.025, -0.03)] * 10)["call"] == "SURVIVES"
    assert verdict(control, [_m(-0.005, -0.005)] * 10)["call"] == "EXPOSURE"
    mixed = [_m(-0.03, -0.03)] * 8 + [_m(0.01, 0.01)] * 2    # 80% improve: neither rule
    assert verdict(control, mixed)["call"] == "PARTIAL"
    assert verdict([_m(0.01, 0.01)] * 10, control)["call"] == "INVALID"
