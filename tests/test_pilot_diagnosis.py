"""Check the diagnostic's metric equivalence and report serialization on CPU."""
import json

import numpy as np
import pytest
import torch

from tools.diagnose_pilot import aggregate, frame_stats
from training.bench_val import pu_psnr


def test_diagnostic_metrics_match_bench_and_training_definitions():
    ref = np.geomspace(0.006, 20000, 120).reshape(3, 5, 8)
    pred = ref * np.linspace(0.7, 1.4, ref.size).reshape(ref.shape)
    result = frame_stats(pred, ref)
    p, r = torch.from_numpy(pred), torch.from_numpy(ref)
    assert result["pu_psnr"] == pytest.approx(pu_psnr(p, r), abs=1e-9)
    error = (torch.log1p(16 * p / 10000) - torch.log1p(16 * r / 10000)).square().mean()
    expected_log = 20 * np.log10(np.log1p(16)) - 10 * np.log10(float(error))
    assert result["train_psnr"] == pytest.approx(expected_log, abs=1e-9)


def test_bright_improvement_can_hide_dark_regression_in_log_metric():
    # Identical reference and pixels for both metrics. The candidate improves
    # bright values while damaging the much larger dark area.
    ref = np.array([1.0] * 90 + [1000.0] * 10)
    baseline = np.array([1.0] * 90 + [900.0] * 10)
    candidate = np.array([5.0] * 90 + [1000.0] * 10)
    before, after = frame_stats(baseline, ref), frame_stats(candidate, ref)
    assert after["train_psnr"] > before["train_psnr"]
    assert after["pu_psnr"] < before["pu_psnr"]


def test_aggregate_serializes_numpy_scores_and_preserves_view_counts():
    rows = [{"views": {"crop": {"pu_gain_db": np.float64(pu),
                                "log_gain_db": np.float64(log)}}}
            for pu, log in [(-2, 3), (1, -1), (0, 4)]]
    result = json.loads(json.dumps(aggregate(rows), allow_nan=False))["crop"]
    assert result["frames"] == 3
    assert result["pu_gain_db"]["mean"] == pytest.approx(-1 / 3)
    assert result["pu_gain_db"]["positive_frames"] == 1
    assert result["log_gain_db"]["positive_frames"] == 2
