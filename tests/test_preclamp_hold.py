"""Opt-in loss, actual forward gates, and bounded scratch integration."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rudra.sdr2hdr import SDR2HDRNet, sdr2hdr_loss
from rudra.source_baseline import baseline_curve_index
from training.preservation import hold_diagnostics, validate_args
from training.routing_ab import arm_args, run_arm, preservation_comparison
from training.routing_replay import ReplayDataset
from test_routing_ab import RandomPairs, recipe_fixture


def model_with_bias(bias, **kwargs):
    torch.manual_seed(12)
    model = SDR2HDRNet(base_channels=8, baseline_source=True, corpus_ev=0, **kwargs).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias[:3].fill_(bias)
    return model


def forward(model, x, **kwargs):
    return model(x, baseline_curve=baseline_curve_index("aces"), **kwargs)


def loss(out, x, domain="postclamp", **kwargs):
    return sdr2hdr_loss(out, x, torch.full_like(x, .00001), crushed_hold=1,
                        crushed_hold_domain=domain, **kwargs)


@pytest.mark.parametrize("mode", ["all", "highlights", "shadows", "off"])
def test_capture_has_identical_outputs_default_loss_and_gradients(mode):
    model = model_with_bias(.02, gate_conditioning=True)
    x = torch.linspace(0, 1, 32 * 32).reshape(1, 1, 32, 32).expand(1, 3, 32, 32)
    kw = dict(recovery_mode=mode, residual_scale=.4, shadow_weight=.3,
              residual_strength=torch.linspace(0, 2, 32 * 32).reshape(32, 32))
    a = forward(model, x, **kw)
    b = forward(model, x, record_preclamp=True, **kw)
    assert a.preclamp_log_delta is None and a.preclamp_log_scale is None
    torch.testing.assert_close(a.hdr, b.hdr, rtol=0, atol=0)
    la, lb = loss(a, x), loss(b, x)
    assert la.keys() == lb.keys()
    for key in la:
        torch.testing.assert_close(la[key], lb[key], rtol=0, atol=0)
    ga = torch.autograd.grad(la["total"], model.head.bias)[0]
    gb = torch.autograd.grad(lb["total"], model.head.bias)[0]
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)


def test_zero_residual_zero_penalty_and_unsaturated_equivalence():
    x = torch.full((1, 3, 32, 32), 1 / 255)
    for bias in (0, .002):
        model = model_with_bias(bias)
        out = forward(model, x, record_preclamp=True)
        before, after = loss(out, x), loss(out, x, "preclamp")
        if bias == 0:
            assert after["crushed_change"].item() == 0
        torch.testing.assert_close(before["crushed_change"], after["crushed_change"], atol=1e-9, rtol=1e-5)
        ga = torch.autograd.grad(before["crushed_change"], model.head.bias, retain_graph=True)[0]
        gb = torch.autograd.grad(after["crushed_change"], model.head.bias)[0]
        # Abs at zero may see round-trip rounding in the historical formula.
        if bias:
            torch.testing.assert_close(ga, gb, atol=1e-7, rtol=1e-5)


def test_saturated_black_gets_finite_restoring_gradient_through_real_head():
    model = model_with_bias(-.1)
    x = torch.full((1, 3, 32, 32), 1 / 255)
    out = forward(model, x, record_preclamp=True)
    assert torch.count_nonzero(out.hdr) == 0
    before, after = loss(out, x), loss(out, x, "preclamp")
    ga = torch.autograd.grad(before["crushed_change"], model.head.bias, retain_graph=True)[0]
    gb = torch.autograd.grad(after["crushed_change"], model.head.bias)[0]
    assert torch.count_nonzero(ga) == 0
    assert torch.isfinite(gb).all() and (gb[:3] < 0).all()
    assert (model.head.bias[:3] - .01 * gb[:3] > model.head.bias[:3]).all()
    diagnostics = hold_diagnostics(out, x)
    assert diagnostics["negative_held_fraction"] == 1
    assert diagnostics["negative_held_channels"] == x.numel()


def test_same_mask_normalization_and_no_change_to_other_terms():
    model = model_with_bias(-.1)
    x = torch.full((1, 3, 32, 32), .5)
    x[:, :, 4, 5] = 2 / 255
    x[:, :, 24, 24] = 3 / 255  # Not crushed.
    out = forward(model, x, record_preclamp=True)
    terms = loss(out, x, "preclamp", region_focus=.5)
    original = loss(out, x, region_focus=.5)
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 3:6, 4:7] = 1
    expected = (out.preclamp_log_delta.abs() * mask).sum() / (27 + 1e-6)
    torch.testing.assert_close(terms["crushed_change"], expected)
    for key in terms.keys() - {"total", "crushed_change"}:
        torch.testing.assert_close(terms[key], original[key], rtol=0, atol=0)
    torch.testing.assert_close(terms["total"] - original["total"],
                               terms["crushed_change"] - original["crushed_change"])
    x.fill_(.5)
    empty = forward(model, x, record_preclamp=True)
    assert loss(empty, x, "preclamp")["crushed_change"].item() == 0
    assert hold_diagnostics(empty, x)["held_channels"].item() == 0


@pytest.mark.parametrize("mode", ["all", "highlights", "shadows", "off"])
def test_capture_respects_actual_artist_and_model_gates(mode):
    model = model_with_bias(.002, gate_conditioning=True)
    x = torch.full((1, 3, 32, 32), 1 / 255)
    out = forward(model, x, recovery_mode=mode, shadow_weight=.25,
                  residual_strength=.6, residual_scale=.2, record_preclamp=True)
    torch.testing.assert_close(loss(out, x)["crushed_change"],
                               loss(out, x, "preclamp")["crushed_change"], atol=1e-9, rtol=1e-5)


@pytest.mark.parametrize("change", ["missing", "scale", "shape", "blend", "domain", "weight"])
def test_invalid_preclamp_requests_fail_explicitly(change):
    model = model_with_bias(-.1)
    x = torch.full((1, 3, 32, 32), .001)
    if change == "blend":
        with pytest.raises(ValueError, match="preserve_outside"):
            forward(model, x, record_preclamp=True, preserve_outside=True)
        return
    out = forward(model, x, record_preclamp=change != "missing")
    if change == "scale": out = replace(out, preclamp_log_scale=8)
    if change == "shape": out = replace(out, preclamp_log_delta=out.preclamp_log_delta[..., :1])
    with pytest.raises(ValueError):
        sdr2hdr_loss(out, x, x, crushed_hold=0 if change == "weight" else 1,
                    crushed_hold_domain="typo" if change == "domain" else "preclamp")


@pytest.mark.parametrize("mode,weight,diagnostics", [("temporal", 1, False), ("image", 0, False),
                                                   ("image", float("nan"), False), ("temporal", 1, True)])
def test_invalid_training_options_rejected_before_preflight(mode, weight, diagnostics):
    with pytest.raises(ValueError):
        validate_args(SimpleNamespace(mode=mode, crushed_hold=weight, crushed_hold_domain="preclamp",
                                      preservation_diagnostics=diagnostics))


def preservation_recipe(tmp_path):
    recipe = recipe_fixture(tmp_path)
    recipe["scope"] = "internal-preservation-ab"
    recipe["trainer_args"] += ["--crushed-hold", "1", "--exposure-push", "0.5"]
    return recipe


@pytest.mark.parametrize("extra", [["--crushed-hold-domain", "preclamp"], ["--region-focus", ".5"],
                                    ["--crushed-hold", "2"], ["--curve-head"],
                                    ["--baseline-dropout", "0"], ["--preservation-diagnostics"]])
def test_preservation_runner_rejects_confounded_args(tmp_path, extra):
    recipe = preservation_recipe(tmp_path)
    recipe["trainer_args"] += extra
    with pytest.raises(ValueError):
        arm_args(recipe, "control")


def test_matched_replay_rejects_a_baseline_change_before_returning_sample(tmp_path):
    control = ReplayDataset(RandomPairs(), "control", 7, tmp_path / "c", matched_baselines=True)
    control[0]; control.close()
    class DifferentBaseline(RandomPairs):
        def __getitem__(self, i):
            sample = super().__getitem__(i)
            sample["baseline_curve"] = torch.tensor(baseline_curve_index("bt1886"))
            return sample
    candidate = ReplayDataset(DifferentBaseline(), "corrected", 7, tmp_path / "t", tmp_path / "c",
                              matched_baselines=True)
    try:
        with pytest.raises(ValueError, match="baseline mismatch"):
            candidate[0]
    finally:
        candidate.close()


@pytest.mark.parametrize("invalid_ev_report", [False, True])
def test_synthetic_preservation_pair_uses_same_baselines_and_only_changes_domain(tmp_path, monkeypatch, invalid_ev_report):
    from training import train_sdr2hdr as trainer, bench_val
    recipe = preservation_recipe(tmp_path)
    Path(recipe["manifest"]).write_text(json.dumps({"sdr_kind": "real"}) + "\n")
    monkeypatch.setattr("pipeline.commercial_preflight.training_preflight",
                        lambda args: {"scope": "internal-pilot", "release_authorized": False})
    class DarkPairs(RandomPairs):
        def __getitem__(self, i):
            sample = super().__getitem__(i)
            sample["sdr"][:, :8, :8] = 1 / 255
            return sample
    monkeypatch.setattr(trainer, "SDRHDRDataset", DarkPairs)
    def evaluate(self, model, device):
        self.last_report = {"per_ev": [{"ev": ev, "clip_frames": 1, "crushed_frames": 1,
                                        "clip_gain_stops": -.1, "crushed_gain_stops": -.2} for ev in (1, 2)]}
        if invalid_ev_report:
            self.last_report["per_ev"][1]["crushed_gain_stops"] = None
        return {"bench_real_frames": 1, "bench_clip_frames": 1, "bench_crushed_frames": 1,
                "bench_real_gain_db": -1., "bench_gain": -1.}
    monkeypatch.setattr(bench_val.BenchVal, "from_manifest", classmethod(lambda cls, *a, **k: cls([{}])))
    monkeypatch.setattr(bench_val.BenchVal, "evaluate", evaluate)
    states, traces, configs = [], [], []
    for arm in ("control", "preclamp"):
        code = run_arm(recipe, arm)
        state = json.loads((tmp_path / f"{arm}_status.json").read_text())
        if invalid_ev_report:
            assert code == 1 and state["state"] == "stopped"
            assert "per-EV preservation scores" in state["reason"]
            assert not (tmp_path / "preclamp").exists()
            return
        assert code == 0
        assert state["state"] == "completed" and state["last_step"] == 2
        assert state["sample_count"] == 8 and state["gradient_observations"] == 2
        assert state["held_channels_seen"] > 0 and state["nonzero_hold_steps"] > 0
        states.append(state)
        traces.append((tmp_path / f"{arm}_samples.jsonl").read_text())
        saved = torch.load(tmp_path / arm / "step_0000002.pt", weights_only=False)
        assert saved["scheduler"]["T_max"] == 1000
        assert saved["config"]["crushed_hold_domain"] == ("preclamp" if arm == "preclamp" else "postclamp")
        configs.append(saved["config"])
        logs = [json.loads(x) for x in (tmp_path / arm / "train.jsonl").read_text().splitlines()]
        steps = [r for r in logs if "gradient_norm_before_clip" in r]
        assert [r["step"] for r in steps] == [1, 2]
        assert all(r["gradient_norm_before_clip"] >= 0 for r in steps)
        assert trainer.SDRHDRDataset is DarkPairs
    assert states[0]["initial_model_sha256"] == states[1]["initial_model_sha256"]
    assert traces[0] == traces[1] and states[1]["changed_baselines"] == 0
    ignore = {"preservation_ab_arm", "crushed_hold_domain", "output_dir"}
    assert {k:v for k,v in configs[0].items() if k not in ignore} == {k:v for k,v in configs[1].items() if k not in ignore}


def test_comparison_reports_shadow_improvement_and_real_tradeoff_without_promotion():
    control = {"metrics": {"bench_real_gain_db": -1}, "per_ev": [
        {"ev": ev, "clip_gain_stops": -.1, "crushed_gain_stops": -2} for ev in (1, 2)]}
    candidate = {"metrics": {"bench_real_gain_db": -2}, "per_ev": [
        {"ev": ev, "clip_gain_stops": -.2, "crushed_gain_stops": -.5} for ev in (1, 2)]}
    result = preservation_comparison({"control": control, "preclamp": candidate})
    assert result["real_pu_delta_db"] == -1
    assert result["both_pushed_crushed_means_improved"] is True
    assert result["production_pass"] is False
    candidate["per_ev"][1]["crushed_gain_stops"] = -2
    assert preservation_comparison({"control": control, "preclamp": candidate})["both_pushed_crushed_means_improved"] is False
    candidate["per_ev"].pop()
    with pytest.raises(ValueError, match="coverage"):
        preservation_comparison({"control": control, "preclamp": candidate})
