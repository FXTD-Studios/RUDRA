"""PeakNet data path (17 Oct 2026). Torch-free; the probe tests need native/tools/clip_fill_probe built
(set RUDRA_CLIP_FILL_PROBE to its path), else they skip."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.peaknet import build_dataset as bd  # noqa: E402

PROBE = os.environ.get("RUDRA_CLIP_FILL_PROBE", "")


def test_error_curve_minimum_is_the_true_dome():
    y0 = np.full(50, 0.01, np.float32)
    p = np.ones(50, np.float32)
    e = bd.error_curve(y0, p, y0 * 2 ** 1.5, np.zeros(50, bool))
    assert bd.A_GRID[int(np.argmin(e))] == pytest.approx(1.5) and e.min() < 1e-4


def test_censored_pixels_are_one_sided():
    y0 = np.full(4, 0.01, np.float32)
    p = np.ones(4, np.float32)
    e = bd.error_curve(y0, p, y0 * 2.0, np.ones(4, bool))
    assert e[0] == pytest.approx(1.0, abs=1e-5)          # below a lower bound: charged
    assert e[np.searchsorted(bd.A_GRID, 3.0)] == 0.0     # above it: free


def test_truth_and_prediction_cap_at_10000_nits():
    e = bd.error_curve(np.full(3, 0.5, np.float32), np.ones(3, np.float32), np.full(3, 50.0, np.float32), np.zeros(3, bool))
    assert e[np.searchsorted(bd.A_GRID, 1.0)] == pytest.approx(0.0, abs=1e-5)


def test_labels_are_8_connected_full_clips():
    codes = np.zeros((6, 6, 3), np.float32)
    codes[1, 1] = codes[2, 2] = 1.0                    # diagonal neighbours: one region
    codes[4, 4, :2] = 1.0                              # partial clip: not a region
    n, labels = bd.full_clip_labels(codes)
    assert n == 2 and labels[1, 1] == labels[2, 2] != 0 and labels[4, 4] == 0


@pytest.mark.skipif(not (PROBE and Path(PROBE).exists()), reason="clip_fill_probe not built")
def test_ecurve_reproduces_the_shipped_fill(tmp_path):
    yy, xx = np.mgrid[0:240, 0:320]
    lin = np.full((240, 320, 3), 0.3, np.float32)
    for cy, cx, peak in ((80, 90, 40.0), (160, 230, 8.0)):
        lin += (peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 14.0 ** 2)))[..., None].astype(np.float32)
    recipe = {"curve": "aces", "contrast": 1.0, "saturation": 1.0, "oetf": "srgb", "render_ev": 1.0,
              "white": 11.2, "codec": "none", "quality": None}
    samples = bd.frame_regions(PROBE, lin, recipe, None, tmp_path, {"frame": "t", "exposure": 0})
    assert samples
    for s in samples:   # E at the default's own dome height is the default's measured error
        assert np.interp(s["a_default"], bd.A_GRID, s["ecurve"]) == pytest.approx(s["e_default"], abs=0.05)
    pred = {s["meta"]["region"]: s["a_oracle"] for s in samples}
    again = bd.frame_regions(PROBE, lin, recipe, None, tmp_path, {"frame": "t", "exposure": 0}, predicted=pred)
    for s in again:     # and the override path delivers what E promised
        assert s["e_native"] == pytest.approx(s["e_oracle"], abs=0.05)
