"""v9 tooling (12 Oct 2026, claude/TASK_CLIPPED_HIGHLIGHTS.md): the exposure
push that makes clipped highlights common in training, and the gate's clip
rows judged against the clip fill instead of the flat inverse."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_exposure_push_is_the_bench_render_and_the_inverse_holds_below_the_clip():
    torch = pytest.importorskip("torch")
    from training.sdr2hdr_dataset import exposure_push
    from rudra.sdr2hdr import sdr_to_baseline_hdr
    g = torch.Generator().manual_seed(0)
    hdr = torch.rand(3, 32, 32, generator=g) * 0.04          # up to 400 nits
    hdr[:, :8, :8] = 0.3                                      # 3 000 nits: clips at +1 EV
    sdr, pushed, ev = exposure_push(hdr, 1.0)
    assert ev == 1.0
    assert torch.allclose(pushed, hdr * 2.0)
    assert float(sdr.min()) >= 0.0 and float(sdr.max()) <= 1.0
    assert torch.allclose(sdr * 255.0, torch.round(sdr * 255.0), atol=1e-4)   # 8-bit codes
    assert bool((sdr[:, :8, :8] == 1.0).all())                                # the patch clipped
    base = sdr_to_baseline_hdr(sdr[None], 0.0)[0]
    below = sdr.amax(0) < 0.9
    rel = ((base - pushed).abs() / pushed.clamp_min(1e-4))[:, below]
    assert float(rel.median()) < 0.02                         # the inverse is the target below the clip


def test_exposure_push_zero_ev_keeps_the_target():
    torch = pytest.importorskip("torch")
    from training.sdr2hdr_dataset import exposure_push
    hdr = torch.full((3, 4, 4), 0.01)
    _, pushed, _ = exposure_push(hdr, 0.0)
    assert torch.equal(pushed, hdr)


def _regions_csv(path: Path, errs: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "scene", "clipped_pct", "crushed_pct",
                                          "clipped_err_stops", "crushed_err_stops"])
        w.writeheader()
        for i, e in enumerate(errs):
            w.writerow({"frame": f"f{i}", "scene": f"s{i % 3}", "clipped_pct": 1.0, "crushed_pct": 1.0,
                        "clipped_err_stops": e, "crushed_err_stops": 1.0})


def test_the_clip_rows_can_be_judged_against_the_clip_fill(tmp_path):
    from training.cp8_verdicts import score
    n = 30
    for ev in (0, 1, 2):
        res = tmp_path / f"cp8_clip_ev{ev}" / "results"
        _regions_csv(res / "baseline.regions.csv", [0.9] * n)   # the flat inverse: far off
        _regions_csv(res / "clipfill.regions.csv", [0.3] * n)   # the clip fill: much closer
        _regions_csv(res / "v9.regions.csv", [0.5] * n)         # the candidate: between them
    vs_inverse = score(tmp_path, "v9")["gates"]
    vs_fill = score(tmp_path, "v9", "clipfill")["gates"]
    for ev in (0, 1, 2):
        assert vs_inverse[f"G8/clip/ev{ev}"]["status"] == "PASS"
        assert vs_fill[f"G8/clip/ev{ev}"]["status"] == "FAIL"
        assert "clipfill" in vs_fill[f"G8/clip/ev{ev}"]["question"]
