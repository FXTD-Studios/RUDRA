"""v10 (13 Oct 2026, docs/MODEL_PLAN_V10.md): the residual on the Source picker's
curve, the crushed hold, the live-action validation split, the bench-like
selection and the gate."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FIXTURE = Path(__file__).with_name("fixtures") / "source_curve_knots_native.json"


def test_the_picker_knots_are_the_apps():
    pytest.importorskip("torch")
    from rudra.source_baseline import baseline_knots
    native = json.loads(FIXTURE.read_text(encoding="utf-8"))["knots"]
    assert set(native) == {"bt1886", "hable", "agx", "camera_log", "clip", "bt1886_shadows"}
    for name, want in native.items():
        got = baseline_knots(name)
        assert got.shape == (257,)
        assert np.abs(got - np.asarray(want, np.float32)).max() < 1e-5, name
    assert not baseline_knots("aces").any() and not baseline_knots("unknown").any()


def test_the_bt1886_baseline_is_the_one_r3_scored():
    torch = pytest.importorskip("torch")
    from rudra.sdr2hdr import SDR2HDRNet
    from training.compare_market import rudra_source_curve
    x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    m = SDR2HDRNet(corpus_ev=0.0, baseline_source=True)
    got = m.picker_baseline(x)[0].permute(1, 2, 0).double().numpy()          # None = the default, BT.1886
    want = rudra_source_curve(x[0].permute(1, 2, 0).double().numpy(), "bt1886") * 203.0 / 10000.0
    assert np.abs(got - want).max() < 1e-5 * max(want.max(), 1e-6)


def test_on_aces_a_v10_model_is_a_v9_model_bit_for_bit():
    torch = pytest.importorskip("torch")
    from rudra.sdr2hdr import SDR2HDRNet
    from rudra.source_baseline import baseline_curve_index
    torch.manual_seed(0)
    old = SDR2HDRNet(corpus_ev=0.0)
    for p in old.head.parameters():                     # a residual that does something
        torch.nn.init.normal_(p, std=0.05)
    new = SDR2HDRNet.from_config({"baseline_source": True, "corpus_ev": 0.0})
    new.load_state_dict(old.state_dict(), strict=True)  # no new parameters: v9 weights load strict
    x = torch.rand(2, 3, 24, 24)
    aces = torch.tensor([baseline_curve_index("aces")] * 2)
    assert torch.equal(old(x).hdr, new(x, baseline_curve=aces).hdr)
    assert not torch.equal(old(x).hdr, new(x).hdr)      # default: BT.1886
    with pytest.raises(ValueError):
        old(x, baseline_curve=torch.tensor([1, 1]))


def test_which_curve_a_sample_rides_on():
    from rudra.source_baseline import baseline_curve_index as idx
    from training.sdr2hdr_dataset import effective_baseline_curve as eff
    real = {"sdr_kind": "real"}
    hable = {"source_curve": "hable"}
    assert eff(real, False, False, True, 0.0) == idx("bt1886")
    assert eff(hable, False, False, False, 0.0) == idx("hable")
    assert eff(hable, True, False, False, 0.0) == idx("bt1886")            # degraded: the default
    assert eff(hable, False, True, False, 0.0) == idx("aces")              # the exposure push renders ACES
    assert eff({"source_curve": "reinhard"}, False, False, False, 0.0) == idx("bt1886")   # not in the picker
    assert eff({}, False, False, False, 0.0) == idx("bt1886")

    class One:
        def random(self):
            return 0.0
    assert eff(hable, False, False, True, 0.5, One()) == idx("bt1886")     # the artist who never picks


def test_the_crushed_hold_charges_only_changes_to_crushed_blacks():
    torch = pytest.importorskip("torch")
    from rudra.sdr2hdr import SDR2HDRNet, sdr2hdr_loss
    m = SDR2HDRNet(corpus_ev=0.0)
    x = torch.full((1, 3, 16, 16), 0.5)
    x[..., :6, :6] = 0.0                                  # a crushed patch
    target = torch.full_like(x, 0.01)
    zero = sdr2hdr_loss(m(x), x, target, crushed_hold=1.0)
    assert float(zero["crushed_change"]) == 0.0           # a fresh head changes nothing
    torch.nn.init.constant_(m.head.bias, 0.0)
    with torch.no_grad():
        m.head.bias[:3] = 2.0                             # lift everything
    out = m(x, recovery_mode="all")
    lifted = sdr2hdr_loss(out, x, target, crushed_hold=1.0)
    assert float(lifted["crushed_change"]) > 0.1
    plain = sdr2hdr_loss(out, x, target)
    assert float(lifted["total"]) > float(plain["total"])


def test_live_action_validation_holds_out_whole_shots():
    from pipeline.build_v10_manifest import resplit
    rows = []
    for title, split in (("Nocturne", "train"), ("Sparks", "train"), ("Sol Levante", "train"),
                         ("Cosmos Laundromat", "val"), ("Meridian", "test")):
        for shot in range(10):
            for f in range(3):
                rows.append({"sdr_kind": "real", "title": title, "split": split,
                             "scene_id": f"{title}_{shot}", "asset_id": f"{title}_{shot}_{f}"})
    rows += [{"split": s, "scene_id": f"r{i}", "asset_id": f"r{i}"} for i, s in enumerate(["train", "val", "test"] * 4)]
    out, report = resplit(rows, val_fraction=0.2)
    val = [r for r in out if r["split"] == "val" and r.get("sdr_kind") == "real"]
    assert val and {r["title"] for r in val} <= {"Nocturne", "Sparks"}
    train_scenes = {r["scene_id"] for r in out if r["split"] == "train"}
    assert not ({r["scene_id"] for r in val} & train_scenes)               # no shot on both sides
    assert all(r["split"] == "train" for r in out if r.get("title") == "Cosmos Laundromat")
    assert [r for r in out if r["split"] == "test"] == [r for r in rows if r["split"] == "test"]
    assert [r for r in out if r.get("sdr_kind") != "real"] == [r for r in rows if r.get("sdr_kind") != "real"]
    assert report["moved"]["live_action_to_val"] == 3 * len(report["held_out_scenes"])


def test_the_bench_validation_of_a_model_that_does_nothing_is_zero():
    torch = pytest.importorskip("torch")
    from rudra.sdr2hdr import SDR2HDRNet
    from rudra.source_baseline import baseline_curve_index
    from training.bench_val import BenchVal, Frame
    g = torch.Generator().manual_seed(2)
    hdr = torch.rand(3, 48, 64, generator=g) * 0.03
    hdr[:, :20, :20] = 0.2                                # 2,000 nits: clips at +1 EV
    sdr = torch.rand(3, 48, 64, generator=g)
    sdr[:, 30:, 40:] = 0.0                                # crushed
    bv = BenchVal(records=[{}, {}], push_evs=(1.0,))
    bv.frames = [Frame(sdr.half(), hdr.half(), True, baseline_curve_index("bt1886"), 0),
                 Frame(sdr.half(), hdr.half(), False, baseline_curve_index("aces"), 0)]
    m = SDR2HDRNet(corpus_ev=0.0, baseline_source=True)
    out = bv.evaluate(m, torch.device("cpu"))
    assert out["bench_real_frames"] == 1 and out["bench_rendered_frames"] == 1
    assert out["bench_clip_frames"] == 1 and out["bench_crushed_frames"] == 1
    for k in ("bench_real_gain_db", "bench_rendered_gain_db", "bench_clip_gain_stops", "bench_crushed_gain_stops", "bench_gain"):
        assert abs(out[k]) < 1e-6, (k, out[k])
    assert m.training                                      # left as it was found


def _csv(path: Path, field: str, values: list[float], extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        names = ["frame", field] + list(extra or {})
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        for i, v in enumerate(values):
            w.writerow({"frame": f"f{i}", field: v, **(extra or {})})


def _whole(path: Path, pu: float, jod: float, n: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "pu_psnr_db", "cvvdp_jod"])
        w.writeheader()
        for i in range(n):
            w.writerow({"frame": f"f{i}", "pu_psnr_db": pu + 0.01 * (i % 3), "cvvdp_jod": jod + 0.001 * (i % 3)})


def _regions(path: Path, clipped: float, crushed: float, n: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "clipped_err_stops", "crushed_err_stops"])
        w.writeheader()
        for i in range(n):
            w.writerow({"frame": f"f{i}", "clipped_err_stops": clipped + 0.01 * (i % 3),
                        "crushed_err_stops": crushed + 0.01 * (i % 3)})


def test_the_gate_judges_against_what_1_0_ships(tmp_path):
    from training.cp10_verdicts import score
    root = tmp_path
    _whole(root / "cp_real/results/v10.csv", 36.0, 8.6)
    _whole(root / "cp_real/results/baseline_bt1886.csv", 35.6, 8.4)   # BT.1886, the reference
    _whole(root / "cp_real/results/baseline.csv", 21.6, 6.4)          # the ACES inverse: not the reference
    _regions(root / "cp_real/results/v10.regions.csv", 0.2, 1.0)
    _regions(root / "cp_real/results/baseline_bt1886.regions.csv", 0.2, 1.0)
    for b in ("aces",):
        _whole(root / f"cp_{b}/results/v10.csv", 40.0, 9.0)
        _whole(root / f"cp_{b}/results/baseline.csv", 40.0, 9.0)
    for b in ("oog", "mix"):
        _whole(root / f"cp_{b}/results/v10.csv", 30.0, 8.0)
        _whole(root / f"cp_{b}/results/baseline_bt1886.csv", 30.0, 8.0)
    for ev in (0, 1, 2):
        r = root / f"cp8_clip_ev{ev}/results"
        _regions(r / "v10.regions.csv", 0.25, 1.0)
        _regions(r / "clipfill.regions.csv", 0.35, 1.0)
        _regions(r / "baseline.regions.csv", 0.6, 1.0)
    s = score(root, "v10")
    assert s["verdict"] == "PASS", {k: g["status"] for k, g in s["gates"].items()}
    # Beating the flat inverse but not the clip fill fails, as v9 did.
    _regions(root / "cp8_clip_ev2/results/v10.regions.csv", 0.45, 1.0)
    s = score(root, "v10")
    assert s["gates"]["G10/clip/ev2"]["status"] == "FAIL" and s["verdict"] == "FAIL"
    # A crushed regression fails on its own.
    _regions(root / "cp8_clip_ev2/results/v10.regions.csv", 0.25, 2.6)
    s = score(root, "v10")
    assert s["gates"]["G10/crushed/ev2"]["status"] == "FAIL"
    # Missing trees read "not run", never pass.
    s = score(tmp_path / "nothing", "v10")
    assert s["verdict"] == "not run"
