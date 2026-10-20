"""Routing and measured validation coverage for the clean scratch pilot."""
import json
import random

import numpy as np
import pytest
import torch

from rudra.source_baseline import baseline_curve_index as idx
from training.sdr2hdr_dataset import effective_baseline_curve as curve


@pytest.mark.parametrize("kind", ["real", "rendered"])
@pytest.mark.parametrize("degraded", [False, True])
@pytest.mark.parametrize("dropout", [0.0, 1.0])
def test_push_routing_overrides_provenance_but_not_degradation(kind, degraded, dropout):
    row = {"sdr_kind": kind, "source_curve": "hable"}
    expected = "bt1886" if degraded or dropout else "aces"
    assert curve(row, degraded, True, True, dropout, random.Random(7)) == idx(expected)
    # Validation never drops the known curve, even when dropout is configured.
    expected_eval = "bt1886" if degraded else "aces"
    assert curve(row, degraded, True, False, dropout) == idx(expected_eval)


@pytest.mark.parametrize("dropout,expected", [(0.0, "aces"), (1.0, "bt1886")])
@pytest.mark.parametrize("degraded", [False, True])
def test_actual_real_dataset_push_selects_matching_baseline(tmp_path, monkeypatch, dropout, expected, degraded):
    from training import sdr2hdr_dataset as ds
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(json.dumps({"asset_id": "f", "scene_id": "shot", "split": "train",
                                   "sdr_kind": "real", "sdr_path": "sdr", "hdr_path": "hdr"}))
    monkeypatch.setattr(ds, "load_rgb", lambda path, hdr: np.full((16, 16, 3),
                                                                 0.02 if hdr else 0.4, np.float32))
    monkeypatch.setattr(ds, "announce_storage", lambda *a: None)
    monkeypatch.setattr(ds.random, "uniform", lambda *a: 1.0)
    monkeypatch.setattr(ds, "degrade_sdr", lambda sdr, strength: sdr * 0.9)
    data = ds.SDRHDRDataset(manifest, crop_size=16, augment=True,
                           exposure_push=1.0, degradation_probability=float(degraded),
                           baseline_dropout=dropout)
    sample = data[0]
    expected_sdr, expected_hdr, _ = ds.exposure_push(torch.full((3, 16, 16), 0.02), 1.0)
    if degraded:
        expected_sdr *= 0.9
        expected = "bt1886"
    torch.testing.assert_close(sample["sdr"], expected_sdr)
    torch.testing.assert_close(sample["hdr"], expected_hdr)
    assert int(sample["baseline_curve"]) == idx(expected)


def test_crop_pu_matches_frame_benchmark_and_ignores_batch_partition():
    from torch.utils.data import DataLoader
    from rudra.sdr2hdr import SDR2HDRNet
    from training.bench_val import pu_psnr
    from training.train_sdr2hdr import evaluate_image
    torch.manual_seed(4)
    model = SDR2HDRNet(base_channels=8, corpus_ev=0, baseline_source=True).eval()
    with torch.no_grad():
        model.head.bias[:3] = 0.3
    rows = [{"sdr": torch.rand(3, 16, 20), "hdr": torch.rand(3, 16, 20) * 0.1,
             "baseline_curve": torch.tensor(idx("bt1886")), "real": torch.tensor(i % 2)}
            for i in range(5)]
    expected = {"all": [], "real": [], "rendered": []}
    with torch.no_grad():
        for row in rows:
            result = model(row["sdr"][None], baseline_curve=row["baseline_curve"][None])
            gain = (pu_psnr(result.hdr[0] * 10000, row["hdr"] * 10000)
                    - pu_psnr(result.analytic_baseline[0] * 10000, row["hdr"] * 10000))
            expected["all"].append(gain)
            expected["real" if row["real"] else "rendered"].append(gain)
    for batch_size in (1, 3):
        metrics = evaluate_image(model, DataLoader(rows, batch_size=batch_size),
                                 torch.device("cpu"), max_batches=0)
        assert not model.training
        for kind, values in expected.items():
            prefix = "" if kind == "all" else f"{kind}_"
            assert metrics[prefix + "pu_frames"] == len(values)
            assert metrics[prefix + "pu_gain_db"] == pytest.approx(np.mean(values), abs=1e-5)


def test_bench_reports_scene_and_ev_coverage_without_changing_aggregate(monkeypatch):
    from types import SimpleNamespace
    from training.bench_val import BenchVal, Frame, pu_psnr
    from training import sdr2hdr_dataset as ds

    class FixedResidual(torch.nn.Module):
        baseline_source = True
        def forward(self, x, **kwargs):
            base = torch.full_like(x, 0.01)
            return SimpleNamespace(hdr=base * 2, analytic_baseline=base)

    def push(hdr, ev):
        # One EV has full clipping; the other has only crushed pixels.
        return torch.full_like(hdr, 1.0 if ev == 1 else 0.0), hdr * 2**ev, ev

    monkeypatch.setattr(ds, "exposure_push", push)
    records = [{"asset_id": str(i), "scene_id": "A" if i < 2 else "B"} for i in range(3)]
    frame = Frame(torch.full((3, 20, 20), 0.5), torch.full((3, 20, 20), 0.04),
                  True, idx("bt1886"), 0)
    bench = BenchVal(records, push_evs=(1.0, 2.0), frames=[frame] * 3)
    model = FixedResidual().eval()
    metrics = bench.evaluate(model, torch.device("cpu"))
    assert not model.training
    real = pu_psnr(frame.hdr * 5000, frame.hdr * 10000) - pu_psnr(frame.hdr * 2500, frame.hdr * 10000)
    assert metrics["bench_real_gain_db"] == pytest.approx(real)
    assert metrics["bench_clip_gain_stops"] == pytest.approx(1.0)
    assert metrics["bench_clip_frames"] == 3
    assert metrics["bench_crushed_frames"] == 0  # Pushed shadows never enter the old real-only pool.
    assert metrics["bench_gain"] == pytest.approx(real + 4.0)
    report = json.loads(json.dumps(bench.last_report, allow_nan=False))
    assert len(report["frames"]) == 9
    first, second = report["per_ev"]
    assert first["clip_frames"] == 3 and first["clip_gain_stops"] == pytest.approx(1)
    assert second["clip_frames"] == 0 and second["clip_gain_stops"] is None
    assert second["crushed_frames"] == 3 and second["crushed_gain_stops"] == pytest.approx(1)
    scenes = [r for r in report["per_scene"] if r["condition"] == "real"]
    assert {r["scene_id"]: r["pu_frames"] for r in scenes} == {"A": 2, "B": 1}
    bench.push_pu = True
    assert bench.evaluate(model, torch.device("cpu")) == metrics
    for row in bench.last_report["per_ev"]:
        expected = (pu_psnr(torch.full_like(frame.hdr, .02) * 10000, frame.hdr.float() * 2**row["ev"] * 10000)
                    - pu_psnr(torch.full_like(frame.hdr, .01) * 10000, frame.hdr.float() * 2**row["ev"] * 10000))
        assert row["pu_frames"] == 3 and row["pu_gain_db"] == pytest.approx(expected)
