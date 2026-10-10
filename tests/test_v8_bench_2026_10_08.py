"""v8 tooling (8 Oct 2026): the region-focused loss, the region scorer, the
exposure re-render, and the v8 gate, on synthetic data."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")


def test_region_focus_zero_is_the_old_loss_and_focus_reads_destroyed_pixels():
    from rudra.sdr2hdr import SDR2HDRNet, sdr2hdr_loss
    torch.manual_seed(0)
    net = SDR2HDRNet(base_channels=8).eval()
    sdr = torch.rand(1, 3, 32, 32) * 0.8
    sdr[..., :8, :8] = 1.0                 # a clipped patch
    target = torch.rand(1, 3, 32, 32) * 0.05
    with torch.no_grad():
        out = net(sdr)
        old = sdr2hdr_loss(out, sdr, target)
        zero = sdr2hdr_loss(out, sdr, target, region_focus=0.0)
        focus = sdr2hdr_loss(out, sdr, target, region_focus=1.0)
    assert float(old["total"]) == float(zero["total"])
    assert 0.0 < float(focus["region_fraction"]) < 0.25   # the 8x8 patch, dilated by 3 px
    assert float(focus["total"]) == pytest.approx(float(focus["region"]) + 2.0 * float(focus["outside_hard"]), rel=1e-5)


def _exr(path: Path, rgb: np.ndarray) -> None:
    from rudra.delivery.exr import write_exr
    path.parent.mkdir(parents=True, exist_ok=True)
    write_exr(path, rgb.astype(np.float32), half=False)


def test_score_regions_and_the_gate(tmp_path):
    import cv2
    from training import cp8_verdicts, score_regions
    rng = np.random.default_rng(1)
    root = tmp_path / "bench" / "cp8_clip_ev1"
    for i in range(12):
        sdr = np.full((32, 32, 3), 0.5)
        sdr[:16, :16] = 1.0                                  # clipped quarter
        sdr[16:, 16:] = 0.0                                  # crushed quarter
        ref = np.full((32, 32, 3), 1.0)
        ref[:16, :16] = 8.0                                  # 3 stops over white in the clip
        base = ref.copy(); base[:16, :16] = 1.0              # the inverse stops at white: 3 st off
        cand = ref.copy(); cand[:16, :16] = 4.0 * (1 + 0.05 * rng.standard_normal())   # ~1 st off
        name = f"f{i:02d}"
        _exr(root / "ref" / "s" / f"{name}.exr", ref)
        _exr(root / "baseline" / "s" / f"{name}.exr", base)
        _exr(root / "v8" / "s" / f"{name}.exr", cand)
        (root / "sdr" / "s").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / "sdr" / "s" / f"{name}.png"), np.rint(sdr * 255).astype(np.uint8))
    assert score_regions.main([str(root)]) == 0
    rows = list(csv.DictReader(open(root / "results" / "baseline.regions.csv")))
    assert len(rows) == 12
    assert float(rows[0]["clipped_pct"]) == pytest.approx(25.0)
    assert float(rows[0]["clipped_err_stops"]) == pytest.approx(3.0, abs=1e-3)
    s = cp8_verdicts.score(tmp_path / "bench", "v8")
    g = s["gates"]["G8/clip/ev1"]
    assert g["status"] == "PASS" and g["clipped_err_stops"]["mean"] == pytest.approx(2.0, abs=0.1)
    assert s["gates"]["G8/real"]["status"] == "not run" and s["verdict"] == "not run"


def test_exposure_render_moves_the_frame_into_the_clip():
    from training.export_bench_pairs import exposure_sdr
    ramp = np.linspace(0.0, 0.2, 64, dtype=np.float32)          # network units: 0 .. 2,000 nits
    hdr = np.repeat(np.repeat(ramp[None, :, None], 8, 0), 3, 2)
    clipped = [float((exposure_sdr(hdr, ev, "aces")[0].max(0).values >= 254 / 255).float().mean()) for ev in (-1, 0, 1, 2)]
    assert clipped == sorted(clipped) and clipped[-1] > clipped[0]
