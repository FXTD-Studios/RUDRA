import json
import math

import cv2
import numpy as np
import pytest

from training import sihdr_eval as se

M709_2020 = np.array([[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]])


def scene(seed=0, h=64, w=80):
    rng = np.random.default_rng(seed)
    ref = rng.uniform(5, 200, (h, w, 3)).astype(np.float32)
    ref[10:30, 10:40] = 6000.0                   # a highlight the SDR clips
    ref[10:30, 10:40, 1] = 5000.0
    anchor = 255.0                               # input saturates at reference 255
    code = np.clip(np.rint(255 * np.clip(ref / anchor, 0, 1) ** (1 / 2.2)), 0, 255).astype(np.uint8)
    return ref, code, anchor


def to2020(x):
    return x @ M709_2020.T


def test_exposure_is_fitted_and_highlights_are_what_differ():
    ref, code, anchor = scene()
    clipped = np.minimum(ref, anchor)
    r = se.score_image(ref, {"perfect": to2020(ref) * 7.0, "clipped": to2020(clipped) * 0.3}, code, anchor)
    perfect, clip = r["methods"]["perfect"], r["methods"]["clipped"]
    assert perfect["pu21_psnr_db"]["saturated"] > 60
    assert clip["pu21_psnr_db"]["saturated"] < perfect["pu21_psnr_db"]["saturated"] - 20
    # unsaturated pixels are identical up to the fitted exposure for both methods
    assert clip["pu21_psnr_db"]["unsaturated"] > 60
    assert perfect["exposure_scale"] == pytest.approx(100.0 / anchor / 7.0, rel=2e-3)
    assert r["pixels"]["saturated"] >= 20 * 30


def test_ceiling_mask_highlight_exclusion_and_bad_inputs():
    ref, code, anchor = scene()
    ref[0, 0] = se.HALF_MAX
    r = se.score_image(ref, {"x": to2020(ref)}, code, anchor, mask_ceiling=True, highlight_excluded=True)
    assert r["pixels"]["full"] == ref.shape[0] * ref.shape[1] - 1
    assert math.isnan(r["methods"]["x"]["pu21_psnr_db"]["saturated"])
    with pytest.raises(ValueError, match="nonfinite"):
        bad = to2020(ref)
        bad[1, 1, 1] = np.nan
        se.score_image(ref, {"x": bad}, code, anchor)
    with pytest.raises(ValueError, match="anchor"):
        se.score_image(ref, {"x": to2020(ref)}, code, 0.0)


def make_dataset(tmp_path, ids):
    import OpenEXR
    def write_exr(path, rgb, half=True):     # SI-HDR references are compressed half-float EXR
        with OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION}, {"RGB": rgb.astype(np.float16)}) as f:
            f.write(str(path))
    root = tmp_path / "sihdr"
    for i, image in enumerate(ids):
        ref, code, anchor = scene(i)
        (root / "reference").mkdir(parents=True, exist_ok=True)
        write_exr(root / "reference" / f"{image}.exr", ref, half=True)
        for clip in se.CLIP_LEVELS:
            (root / clip).mkdir(exist_ok=True)
            assert cv2.imwrite(str(root / clip / f"{image}.png"), code[..., ::-1])
    groups = {k: v for k, v in {"g1": ids[:2], "g2": ids[2:4], "g3": ids[4:]}.items() if v}
    prep = {"sihdr": {"scene_groups_for_bootstrap": groups,
                      "category_tags_from_contact_sheet_review": {"sun": ids[0:3:2], "skin": ids[1:2]}}}
    anchors = {i: {c: {"ref_maxchannel_threshold": 255.0} for c in se.CLIP_LEVELS} for i in ids}
    (tmp_path / "prep.json").write_text(json.dumps(prep))
    (tmp_path / "anchors.json").write_text(json.dumps(anchors))
    return root


class FakeRunner:
    """Model expands highlights by `gain`; the baseline clips them."""

    def __init__(self, gain):
        self.gain = gain

    def predict(self, code):
        lin = (code.astype(np.float64) / 255.0) ** 2.2
        expanded = np.where(code >= 250, lin * self.gain, lin)
        return {"model": to2020(expanded), "baseline_bt1886": to2020(lin)}


def test_freeze_evaluate_compare_end_to_end(tmp_path):
    ids = ["001", "002", "003", "004", "005", "006"]
    root = make_dataset(tmp_path, ids)
    protocol = tmp_path / "protocol.json"
    se.freeze(tmp_path / "prep.json", tmp_path / "anchors.json", protocol)
    with pytest.raises(ValueError, match="never rewritten"):
        se.freeze(tmp_path / "prep.json", tmp_path / "anchors.json", protocol)
    ckpt_a, ckpt_b = tmp_path / "a.pt", tmp_path / "b.pt"
    ckpt_a.write_bytes(b"a")
    ckpt_b.write_bytes(b"b")
    good = se.evaluate(protocol, root, ckpt_a, tmp_path / "run_a", runner=FakeRunner(20.0))
    weak = se.evaluate(protocol, root, ckpt_b, tmp_path / "run_b", runner=FakeRunner(1.0))
    assert good["complete"] and good["release_authorized"] is False
    assert good["verdict"]["clip_95"]["A2_highlight_expansion"] is True
    assert weak["verdict"]["clip_95"]["A2_highlight_expansion"] is False   # identical to baseline
    assert good["model_minus_baseline"]["clip_95"]["saturated"]["scenes"] == 3
    assert "SIMULATED SDR" in good["evidence_class"]
    cmp = se.compare_runs(protocol, tmp_path / "run_a", tmp_path / "run_b")
    assert cmp["a_minus_b"]["clip_97"]["saturated"]["ci95"][0] > 0
    with pytest.raises(ValueError, match="never overwritten"):
        se.evaluate(protocol, root, ckpt_a, tmp_path / "run_a", runner=FakeRunner(2.0))
    assert not (tmp_path / "run_a.partial").exists()

    class Broken(FakeRunner):
        def predict(self, code):
            raise RuntimeError("GPU lost")
    with pytest.raises(RuntimeError):
        se.evaluate(protocol, root, ckpt_a, tmp_path / "run_c", runner=Broken(1.0))
    assert (tmp_path / "run_c.partial").is_dir() and not (tmp_path / "run_c").exists()
    with pytest.raises(ValueError, match="never overwritten"):
        se.evaluate(protocol, root, ckpt_a, tmp_path / "run_c", runner=FakeRunner(2.0))


def test_protocol_refuses_code_changed_after_freezing(tmp_path, monkeypatch):
    ids = ["001", "002", "003", "004"]
    make_dataset(tmp_path, ids)
    protocol = tmp_path / "protocol.json"
    se.freeze(tmp_path / "prep.json", tmp_path / "anchors.json", protocol)
    data = json.loads(protocol.read_text())
    data["code_sha256"]["training/sihdr_eval.py"] = "0" * 64
    protocol.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed after the protocol was frozen"):
        se.load_protocol(protocol)


def test_discovery_requires_exactly_one_file_per_kind(tmp_path):
    ids = ["001", "002"]
    root = make_dataset(tmp_path, ids)
    assert set(se.discover(root, ids)["001"]) == {"reference", "clip_95", "clip_97"}
    (root / "clip_95" / "copy_001.png").write_bytes((root / "clip_95" / "001.png").read_bytes())
    with pytest.raises(ValueError, match="ambiguous"):
        se.discover(root, ids)


def test_real_runner_produces_finite_bt2020_predictions(tmp_path):
    import torch
    from rudra.sdr2hdr import SDR2HDRNet
    model = SDR2HDRNet(base_channels=8, baseline_source=True, corpus_ev=0.0)
    ckpt = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(), "step": 1,
                "config": {"base_channels": 8, "baseline_source": True, "corpus_ev": 0.0}}, ckpt)
    _, code, _ = scene()
    out = se.Runner(ckpt, "cpu").predict(code)
    assert set(out) == {"model", "baseline_bt1886"}
    assert all(v.shape == code.shape and np.isfinite(v).all() for v in out.values())
