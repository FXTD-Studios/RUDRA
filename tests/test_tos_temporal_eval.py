import json

import numpy as np
import pytest

from training import tos_temporal_eval as te


@pytest.fixture
def small_sections(monkeypatch):
    monkeypatch.setattr(te, "SECTIONS", {"03_2a": (10, 15), "04_5f": (20, 25)})


def make_tos(root):
    import OpenEXR
    def write_exr(path, rgb, half=False):     # ToS plates are compressed EXR
        with OpenEXR.File({"compression": OpenEXR.PIZ_COMPRESSION}, {"RGB": rgb.astype(np.float32)}) as f:
            f.write(str(path))
    rng = np.random.default_rng(3)
    base = rng.uniform(0.02, 1.0, (48, 64, 3)).astype(np.float32)
    base[5:12, 5:30] = 6.0                       # moving-free bright highlight
    for section, (lo, hi) in te.SECTIONS.items():
        for n in range(lo, hi + 1):
            frame = np.roll(base, n - lo, axis=1).copy()
            if section == "04_5f":
                frame[30:36, 40:46] = 11.80     # source-clipped lamp plateau
            p = root / section / f"frame_{n:06d}.exr"
            p.parent.mkdir(parents=True, exist_ok=True)
            write_exr(p, frame, half=False)


def fake_render(lin2020, recipe):
    hdr = np.maximum(lin2020, 0).astype(np.float32)          # 1.0 = 203 nits
    sdr = np.clip(hdr / (1 + hdr), 0, 1) ** (1 / 2.2)
    return (sdr * 65535).astype(np.uint16), hdr, (sdr * 255).astype(np.uint8)


def predictor_from(kit, flicker=0.0):
    rows = [json.loads(l) for l in (kit / "frames.jsonl").read_text().splitlines()]
    from training.exr_any import read_rgb
    refs = {r["sdr_path"]: read_rgb(r["hdr_path"]) * 203.0 for r in rows}
    order = [r["sdr_path"] for r in rows]
    import cv2
    by_pixels = {cv2.imread(p, cv2.IMREAD_UNCHANGED).tobytes(): p for p in order}

    def predict(sdr01, condition, index):
        key = np.rint(sdr01[..., ::-1] * 65535).astype(np.uint16).tobytes()
        ref = refs[by_pixels[key]]
        gain = 1.0 + (flicker if index % 2 else -flicker)
        return ref * gain, np.minimum(ref, 203.0)
    return predict


def test_build_freezes_and_masks_plateau(tmp_path, small_sections):
    make_tos(tmp_path / "tos")
    contract = te.build(tmp_path / "tos", tmp_path / "kit", render=fake_render)
    assert contract["frozen_before_model_scores"] and contract["scene_exposure"] == 1.0
    assert "SYNTHETIC" in contract["evidence_class"]
    _, rows = te.load_kit(tmp_path / "kit")
    assert len(rows) == 12
    import cv2
    mask = cv2.imread(rows[-1]["mask_path"], cv2.IMREAD_GRAYSCALE) > 0
    assert mask[33, 43] and not mask[0, 0]
    with pytest.raises(ValueError, match="immutable"):
        te.build(tmp_path / "tos", tmp_path / "kit", render=fake_render)


def test_missing_frames_refused(tmp_path, small_sections):
    make_tos(tmp_path / "tos")
    (tmp_path / "tos" / "03_2a" / "frame_000012.exr").unlink()
    with pytest.raises(ValueError, match="expected frames"):
        te.build(tmp_path / "tos", tmp_path / "kit", render=fake_render)
    assert not (tmp_path / "kit").exists() and not (tmp_path / "kit.partial").exists()


def test_evaluate_flags_flicker_and_passes_stable_output(tmp_path, small_sections):
    make_tos(tmp_path / "tos")
    kit = tmp_path / "kit"
    te.build(tmp_path / "tos", kit, render=fake_render)
    ckpt = tmp_path / "c.pt"
    ckpt.write_bytes(b"x")
    stable = te.evaluate(kit, ckpt, tmp_path / "stable", predictor=predictor_from(kit))
    v = stable["sections"]["03_2a"]["clean"]["verdict"]
    assert v["scored"] and v["T1"] and v["T3"] and v["T4_stateless_reset"]
    flick = te.evaluate(kit, ckpt, tmp_path / "flicker", predictor=predictor_from(kit, flicker=0.3))
    v = flick["sections"]["03_2a"]["clean"]["verdict"]
    assert v["T1"] is False and v["popping_pairs"]
    assert flick["release_authorized"] is False
    # tampering with frozen media is refused
    first = json.loads((kit / "frames.jsonl").read_text().splitlines()[0])
    with open(first["mask_path"], "ab") as f:
        f.write(b"0")
    with pytest.raises(ValueError, match="kit media changed"):
        te.load_kit(kit)


def test_section_verdict_requires_equal_scored_pairs():
    a = {"summary": {"pairs_scored": 3, "excess": {"mean": 0.0}}, "pairs": []}
    b = {"summary": {"pairs_scored": 2, "excess": {"mean": 0.0}}, "pairs": []}
    assert te.section_verdict(a, b)["scored"] is False
