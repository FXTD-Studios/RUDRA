import json

import cv2
import numpy as np
import pytest
import torch

from training import generative_review as gr


def gen_image(seed=0, h=96, w=128):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 200, (h, w, 3)).astype(np.uint8)
    img[:30, :40] = rng.integers(0, 12, (30, 40, 3))          # noisy crushed blacks
    img[60:80, 90:120] = 255                                   # clipped highlight
    return img


def test_frame_metrics_flags_lifted_coloured_blacks():
    rgb8 = gen_image()
    base = np.full(rgb8.shape, 0.05)
    base[60:80, 90:120] = 203.0
    good = base.copy(); good[60:80, 90:120] = 800.0
    bad = good.copy(); bad[:30, :40, 0] = 5.0; bad[:30, :40, 1] = 4.0   # +6-7 stops, coloured
    g, b = gr.frame_metrics(rgb8, good, base), gr.frame_metrics(rgb8, bad, base)
    assert g["dark_lift_fraction"] == 0 and b["dark_lift_fraction"] > 0.9
    assert b["dark_chroma_shift"] > g["dark_chroma_shift"]
    assert g["dark_visible_lift_fraction"] == 0 and b["dark_visible_lift_fraction"] > 0.9
    assert b["dark_visible_chroma_shift"] > 0.5
    assert g["clipped_median_nits"] == pytest.approx(800.0)


def test_invisible_relative_lift_is_not_visible_lift():
    rgb8 = gen_image()
    base = np.full(rgb8.shape, 0.0001)
    pred = np.full(rgb8.shape, 0.01)                           # +3 stops (1e-3 floor), far below visibility
    m = gr.frame_metrics(rgb8, pred, base)
    assert m["dark_lift_fraction"] > 0.9 and m["dark_visible_lift_fraction"] == 0
    assert m["dark_visible_chroma_shift"] == 0.0


def test_seam_ratio_detects_tile_edges():
    y = np.random.default_rng(1).uniform(10, 11, (300, 1200))
    assert gr.seam_ratio(y) == pytest.approx(1.0, abs=0.2)
    stepped = y.copy(); stepped[:, 480:] *= 2.0                # discontinuity on the first overlap centre line
    assert gr.seam_ratio(stepped) > 1.5
    assert gr.seam_ratio(np.ones((100, 100))) is None


def make_inputs(tmp_path):
    inp = tmp_path / "gen"
    inp.mkdir()
    cv2.imwrite(str(inp / "still.png"), gen_image()[..., ::-1])
    vw = cv2.VideoWriter(str(inp / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (128, 96))
    for i in range(12):
        vw.write(gen_image(i)[..., ::-1])
    vw.release()
    return inp


def test_review_end_to_end_with_real_tiny_model(tmp_path):
    from rudra.sdr2hdr import SDR2HDRNet
    ckpt = tmp_path / "tiny.pt"
    torch.save({"model": SDR2HDRNet(base_channels=8, baseline_source=True, corpus_ev=0.0).state_dict(), "step": 1,
                "config": {"base_channels": 8, "baseline_source": True, "corpus_ev": 0.0}}, ckpt)
    inp = make_inputs(tmp_path)
    rep = gr.review(inp, {"tiny": ckpt}, tmp_path / "out", methods=gr.Methods({"tiny": ckpt}, "cpu"), bursts=2)
    kinds = {i["item"]: i for i in rep["items"]}
    assert set(kinds) == {"still.png", "clip.mp4"}
    assert kinds["clip.mp4"]["frames_sampled"] == 6
    assert "flicker_excess_ev" in kinds["clip.mp4"]["methods"]["tiny"]
    assert (tmp_path / "out" / kinds["still.png"]["sheet"]).is_file()
    assert "| still.png | tiny |" in (tmp_path / "out" / "review_sheet.md").read_text()
    assert rep["release_authorized"] is False
    with pytest.raises(ValueError, match="never overwritten"):
        gr.review(inp, {}, tmp_path / "out", methods=gr.Methods({}, "cpu"))


def test_baseline_only_review_works_without_a_model(tmp_path):
    inp = make_inputs(tmp_path)
    rep = gr.review(inp, {}, tmp_path / "base", methods=gr.Methods({}, "cpu"), bursts=1)
    assert all(set(i["methods"]) == {"baseline_bt1886"} for i in rep["items"])
