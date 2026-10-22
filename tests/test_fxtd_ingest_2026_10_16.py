"""pipeline/ingest_fxtd_footage.py: FXTD camera footage -> corpus (16 Oct 2026). Torch-free."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline import ingest_fxtd_footage as ing  # noqa: E402
from pipeline.hdr_io import decode_hdr_u16  # noqa: E402
from rudra.delivery.colorspace import npm  # noqa: E402


def test_awg4_matches_the_native_fixture():
    # native/tests/test_log_encode.cpp, ARRI 2022.
    want = np.array([[0.704858320407232, 0.129760295170463, 0.115837311473976],
                     [0.254524176404027, 0.781477732712002, -0.036001909116029],
                     [0.0, 0.0, 1.089057750759878]])
    assert np.allclose(ing.camera_npm("awg4"), want, atol=1e-9)


def test_awg3_to_rec709_is_arris_published_matrix():
    m = np.linalg.inv(npm("rec709")) @ ing.camera_npm("awg3")
    want = np.array([[1.617523, -0.537287, -0.080237],
                     [-0.070573, 1.334613, -0.264040],
                     [-0.021102, -0.226954, 1.248056]])
    assert np.allclose(m, want, atol=2e-4)


@pytest.mark.parametrize("gamut", ["rwg", "awg3", "awg4"])
def test_white_stays_white_in_rec2020(gamut):
    m = ing.camera_to_rec2020(gamut)
    assert np.allclose(m.sum(1), 1.0, atol=1e-5)


def test_log_curves_put_grey_at_018():
    assert ing.decode_log(np.array([0.391007], np.float32), "logc3")[0] == pytest.approx(0.18, rel=2e-3)
    assert ing.decode_log(np.array([0.278396], np.float32), "logc4")[0] == pytest.approx(0.18, rel=2e-3)


def test_plateau_is_a_clip_and_a_peak_is_not():
    rng = np.random.default_rng(0)
    natural = rng.uniform(0, 3, (64, 64, 3)).astype(np.float32)
    clipped = natural.copy()
    clipped[:20, :20, 0] = 9.27          # 400 bit-equal red pixels
    top, flag = ing.channel_plateaus([ing.frame_peak(natural), ing.frame_peak(clipped)])
    assert flag.tolist() == [True, False, False] and top[0] == np.float32(9.27)
    assert ing.channel_plateaus([ing.frame_peak(natural)])[1].tolist() == [False, False, False]


def test_ceiling_sits_just_under_the_clip_white():
    m = ing.camera_to_rec2020("rwg")
    top = np.array([9.27, 9.27, 9.27], np.float32)
    assert ing.shot_ceiling_nits(top, np.array([False] * 3), m) is None
    got = ing.shot_ceiling_nits(top, np.array([True] * 3), m)
    assert got == pytest.approx(ing.CEILING_MARGIN * 9.27 * 203.0, rel=1e-4)


def test_splits_are_whole_contiguous_shots_and_small_projects_train():
    shots = [f"RL6_FB_SH_{i:02d}" for i in range(1, 41)]
    a, b = ing.assign_splits(shots), ing.assign_splits(list(reversed(shots)))
    assert a == b
    val = [i for i, s in enumerate(sorted(shots)) if a[s] == "val"]
    assert len(val) == round(40 * ing.VAL_FRACTION) and val == list(range(val[0], val[0] + len(val)))
    assert set(ing.assign_splits(shots[:6]).values()) == {"train"}


def test_even_picks():
    assert ing.even_picks(100, 10) == [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    assert ing.even_picks(3, 10) == [0, 1, 2]
    assert ing.even_picks(0, 10) == []


def test_end_to_end_on_a_synthetic_shot(tmp_path, monkeypatch):
    src_root = tmp_path / "plates"
    seq = src_root / "Untitled 7" / "4096x1716"
    seq.mkdir(parents=True)
    for i in range(30):
        (seq / f"Shot_{1000 + i:08d}.exr").write_bytes(b"")
    rng = np.random.default_rng(1)

    def fake_camera_linear(src, item):
        img = rng.uniform(0.001, 2.0, (400, 960, 3)).astype(np.float32)
        img[:50, :50] = 9.27                                  # a blown practical
        return img

    monkeypatch.setattr(ing, "camera_linear", fake_camera_linear)
    monkeypatch.setitem(ing.SOURCES, "manchale", dict(ing.SOURCES["manchale"]))
    out = tmp_path / "corpus"
    assert ing.main(["--out", str(out), "--sources", "manchale", "--root", f"manchale={src_root}",
                     "--frames-per-shot", "4"]) == 0
    # Resume is a no-op.
    assert ing.main(["--out", str(out), "--sources", "manchale", "--root", f"manchale={src_root}",
                     "--frames-per-shot", "4"]) == 0
    report = ing.finalize(out, "G:\\datasets\\corpora\\x")
    assert report["records"] == 4 and report["scenes"] == 1 and not report["scenes_straddling_splits"]
    rows = [json.loads(line) for line in (out / "sdr_hdr_manifest.jsonl").read_text().splitlines()]
    r = rows[0]
    assert r["scene_id"] == "fxtd_manchale_untitled_7" and r["licence_source"] == "fxtd"
    assert r["hdr_path"].startswith("G:\\datasets\\corpora\\x\\hdr\\")
    assert r["tonemap_ev"] == 0.0 and r["sdr_kind"] == "rendered" and r["ceiling_nits"] > 1000
    assert len({row["sdr_curve"] for row in rows}) == 1       # one grade per shot
    cfg = json.loads((out / "_ingest_config.json").read_text())
    hdr = cv2.imread(str(out / "hdr" / f"{r['asset_id']}.png"), cv2.IMREAD_UNCHANGED)
    sdr = cv2.imread(str(out / "sdr" / f"{r['asset_id']}.png"), cv2.IMREAD_UNCHANGED)
    assert hdr.dtype == np.uint16 and sdr.dtype == np.uint8 and hdr.shape == sdr.shape == (400, 960, 3)
    from pipeline.hdr_io import HDRStorage
    lin = decode_hdr_u16(hdr, HDRStorage.from_dict(cfg["storage"]))
    white = 9.27 * ing.camera_to_rec2020("rwg").sum(1)
    assert np.allclose(lin[10, 10], white, rtol=2e-3)       # the clip survives the round trip
    assert r["peak_at_ceiling"] is True


def test_mix_refuses_collisions_and_mixed_exposure():
    from pipeline.build_fxtd_mix_manifest import merge
    base = [{"asset_id": "a", "scene_id": "s", "split": "train", "tonemap_ev": 0.0}]
    fx = [{"asset_id": "b", "scene_id": "fxtd_x", "split": "val", "tonemap_ev": 0.0}]
    out, report = merge(base, fx)
    assert report["rows"] == 2 and out[1]["source_corpus"] == "rudra_fxtd_camera"
    with pytest.raises(SystemExit):
        merge(base, [dict(fx[0], scene_id="s")])
    with pytest.raises(SystemExit):
        merge(base, [dict(fx[0], tonemap_ev=-1.0)])
