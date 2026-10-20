"""The push-baseline A/B (14 Oct 2026): one declared factor, matched draws, no promotion."""
import json
import random
from pathlib import Path

import pytest
import torch

from rudra.source_baseline import baseline_curve_index as idx
from tests.test_preclamp_hold import preservation_recipe
from tests.test_routing_ab import RandomPairs
from training.routing_ab import (PUSH_SCOPE, arm_args, arms, push_baseline_comparison, run_arm)
from training.routing_replay import ReplayDataset
from training.sdr2hdr_dataset import SDRHDRDataset, effective_baseline_curve


def recipe_at(path):
    recipe = preservation_recipe(path)
    recipe["scope"] = PUSH_SCOPE
    recipe["trainer_args"] += ["--deterministic", "--log-every", "1"]
    return recipe


def full_size(recipe):
    recipe["trainer_args"] += ["--crop-size", "256", "--base-channels", "32"]
    return recipe


@pytest.mark.parametrize("record", [{"sdr_kind": "real"}, {"sdr_kind": "rendered", "source_curve": "hable"},
                                    {"sdr_kind": "rendered", "source_curve": "unknown"}])
@pytest.mark.parametrize("degraded", [False, True])
@pytest.mark.parametrize("pushed", [False, True])
def test_render_routing_changes_only_pushes_and_draws_the_same_numbers(record, degraded, pushed):
    out = {}
    for mode in ("picker", "render"):
        random.seed(11)
        out[mode] = (effective_baseline_curve(record, degraded, pushed, True, .5, push_baseline=mode), random.random())
    assert out["picker"][1] == out["render"][1]
    if pushed:
        assert out["render"][0] == idx("aces")
    else:
        assert out["render"][0] == out["picker"][0]


def test_default_routing_is_unchanged_and_typos_fail(tmp_path):
    random.seed(5)
    a = [effective_baseline_curve({"sdr_kind": "real"}, False, True, True, .5) for _ in range(64)]
    random.seed(5)
    b = [effective_baseline_curve({"sdr_kind": "real"}, False, True, True, .5, push_baseline="picker")
         for _ in range(64)]
    assert a == b and set(a) == {idx("bt1886"), idx("aces")}
    with pytest.raises(ValueError, match="push_baseline"):
        effective_baseline_curve({}, False, True, True, .5, push_baseline="aces")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"split": "train", "sdr_path": "x", "hdr_path": "y"}) + "\n")
    with pytest.raises(ValueError, match="push_baseline"):
        SDRHDRDataset(manifest, push_baseline="typo")


def test_arms_differ_only_in_the_declared_factor(tmp_path):
    recipe = full_size(recipe_at(tmp_path))
    assert arms(recipe) == ("control", "render")
    configs = [vars(arm_args(recipe, arm)) for arm in arms(recipe)]
    assert configs[0]["push_baseline"] == "picker" and configs[1]["push_baseline"] == "render"
    for c in configs:
        assert c["crushed_hold_domain"] == "preclamp" and c["region_focus"] == 0 and c["bench_push_pu"]
        assert c["deterministic"] and c["exposure_push"] == .5 and c["baseline_dropout"] == .5
    assert {k for k in configs[0] if configs[0][k] != configs[1][k]} == {
        "output_dir", "preservation_ab_arm", "push_baseline"}


@pytest.mark.parametrize("extra", [["--push-baseline", "render"], ["--region-focus", ".5"],
                                    ["--baseline-dropout", "0"], ["--exposure-push", "0.25"],
                                    ["--crop-size", "128"], ["--log-every", "5"]])
def test_confounds_rejected(tmp_path, extra):
    recipe = full_size(recipe_at(tmp_path))
    recipe["trainer_args"] += extra
    with pytest.raises(ValueError):
        arm_args(recipe, "control")


def test_deterministic_mode_is_required(tmp_path):
    recipe = preservation_recipe(tmp_path)
    recipe["scope"] = PUSH_SCOPE
    recipe["trainer_args"] += ["--log-every", "1", "--crop-size", "256", "--base-channels", "32"]
    with pytest.raises(ValueError, match="deterministic"):
        arm_args(recipe, "render")


@pytest.mark.parametrize("seconds,valid", [(900, True), (901, False)])
def test_cap_and_pending_approval(tmp_path, monkeypatch, seconds, valid):
    from training import routing_ab as ab
    from pipeline.commercial_preflight import sha256
    monkeypatch.setattr(ab, "REPO", tmp_path)
    monkeypatch.setattr(ab, "CODE_FILES", ())
    manifest, evidence = tmp_path / "manifest", tmp_path / "evidence"
    manifest.write_text("synthetic")
    evidence.write_text(json.dumps({"scope": "internal-pilot", "release_authorized": False,
        "reviewed_by": "test", "reviewed_at": "test", "sources": {"t": {"weight_distribution": False}}}))
    recipe = {"scope": PUSH_SCOPE, "release_authorized": False, "arm_steps": 200, "arm_seconds": seconds,
              "storage_stop_bytes": 900000000, "output_root": str(tmp_path / "reports/new"), "code_sha256": {},
              "manifest": str(manifest), "manifest_sha256": sha256(manifest),
              "evidence": str(evidence), "evidence_sha256": sha256(evidence)}
    path, approval = tmp_path / "recipe.json", tmp_path / "approval.json"
    path.write_text(json.dumps(recipe))
    approval.write_text(json.dumps({"approved": False, "approved_by": None, "approved_at": None,
                                    "release_authorized": False, "recipe_sha256": sha256(path)}))
    with pytest.raises(ValueError, match="explicit approval"):
        ab.approved_recipe(path, approval)
    approval.write_text(json.dumps({"approved": True, "approved_by": "test", "approved_at": "test",
                                    "release_authorized": False, "recipe_sha256": sha256(path)}))
    if valid:
        assert ab.approved_recipe(path, approval)["scope"] == PUSH_SCOPE
    else:
        with pytest.raises(ValueError, match="limits"):
            ab.approved_recipe(path, approval)


class Routed(RandomPairs):
    """Even draws stand in for pushes: BT.1886 under picker routing, ACES under render."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.push_baseline = kwargs.get("push_baseline", "picker")

    def __getitem__(self, i):
        sample = super().__getitem__(i)
        sample["sdr"][:, :8, :8] = 1 / 255     # crushed pixels, so the hold has coverage
        pushed = random.random() < .5
        sample["baseline_curve"] = torch.tensor(idx("aces") if pushed and self.push_baseline == "render"
                                                else idx("bt1886"))
        return sample


def test_replay_allows_only_the_declared_direction(tmp_path):
    control = ReplayDataset(Routed(), "control", 7, tmp_path / "c", matched_baselines=True)
    for i in range(16):
        control[i % 4]
    control.close()
    change = (idx("bt1886"), idx("aces"))
    candidate = ReplayDataset(Routed(push_baseline="render"), "corrected", 7, tmp_path / "t", tmp_path / "c",
                              matched_baselines=True, allowed_baseline_change=change)
    for i in range(16):
        candidate[i % 4]
    candidate.finish(16)
    candidate.close()
    assert 0 < candidate.changed_baselines < 16
    reverse = ReplayDataset(Routed(), "corrected", 7, tmp_path / "r", tmp_path / "t",
                            matched_baselines=True, allowed_baseline_change=change)
    try:
        with pytest.raises(ValueError, match="baseline mismatch"):
            for i in range(16):
                reverse[i % 4]
    finally:
        reverse.close()
    with pytest.raises(ValueError, match="matched"):
        ReplayDataset(Routed(), "control", 7, tmp_path / "x", allowed_baseline_change=change)


def test_pair_runs_real_trainer_with_matched_draws(tmp_path, monkeypatch):
    from training import train_sdr2hdr as trainer, bench_val
    recipe = full_size(recipe_at(tmp_path))  # Routed samples stay 16 px; the network is the real one.
    recipe["arm_seconds"] = 900
    Path(recipe["manifest"]).write_text('{"sdr_kind":"real"}\n')
    monkeypatch.setattr("pipeline.commercial_preflight.training_preflight",
                        lambda args: {"scope": "internal-pilot", "release_authorized": False})
    monkeypatch.setattr(trainer, "SDRHDRDataset", Routed)
    monkeypatch.setattr(bench_val.BenchVal, "from_manifest", classmethod(lambda cls, *a, **k: cls([{}])))

    def evaluate(self, model, device):
        assert self.push_pu
        self.last_report = {"per_ev": [{"ev": ev, "clip_frames": 1, "crushed_frames": 1, "pu_frames": 1,
            "pu_gain_db": -2., "clip_gain_stops": -.1, "crushed_gain_stops": -.2} for ev in (1, 2)]}
        return {"bench_real_frames": 1, "bench_clip_frames": 1, "bench_crushed_frames": 1,
                "bench_real_gain_db": -1., "bench_gain": -1.}
    monkeypatch.setattr(bench_val.BenchVal, "evaluate", evaluate)
    # Strict mode is process-wide and refuses a process where CUDA is already up
    # (earlier tests in this pytest run may have done that). The real runner
    # gives each arm a fresh child; here only the request is recorded.
    requested = []
    monkeypatch.setattr(trainer, "configure_determinism", requested.append)
    states, traces, configs = {}, {}, {}
    for arm in arms(recipe):
        assert run_arm(recipe, arm) == 0, json.loads((tmp_path / f"{arm}_status.json").read_text()).get("reason")
        states[arm] = json.loads((tmp_path / f"{arm}_status.json").read_text())
        traces[arm] = [json.loads(line) for line in (tmp_path / f"{arm}_samples.jsonl").read_text().splitlines()]
        configs[arm] = torch.load(tmp_path / arm / "step_0000002.pt", weights_only=False)["config"]
    assert requested == [True, True]            # each arm asked for strict mode
    assert states["control"]["changed_baselines"] == 0 < states["render"]["changed_baselines"]
    assert [{k: v for k, v in r.items() if k != "baseline_curve"} for r in traces["control"]] == \
           [{k: v for k, v in r.items() if k != "baseline_curve"} for r in traces["render"]]
    assert states["control"]["initial_model_sha256"] == states["render"]["initial_model_sha256"]
    assert configs["control"]["push_baseline"] == "picker" and configs["render"]["push_baseline"] == "render"
    ignored = {"output_dir", "preservation_ab_arm", "push_baseline"}
    assert {k: v for k, v in configs["control"].items() if k not in ignored} == \
           {k: v for k, v in configs["render"].items() if k not in ignored}


def _state(real, value, changed):
    return {"changed_baselines": changed, "metrics": {"bench_real_gain_db": real}, "per_ev": [
        {"ev": ev, "clip_frames": 1, "crushed_frames": 1, "pu_frames": 2,
         "clip_gain_stops": value, "crushed_gain_stops": value, "pu_gain_db": value} for ev in (1, 2)]}


def test_verdict_is_preregistered_and_never_promotes():
    pair = {"control": _state(-1.29, -2., 0), "render": _state(-.5, -1., 300)}
    result = push_baseline_comparison(pair)
    assert result["early_hypothesis_supported"] and not result["production_pass"]
    pair["render"]["per_ev"][1]["crushed_gain_stops"] = -2.5
    assert not push_baseline_comparison(pair)["early_hypothesis_supported"]
    pair["render"]["changed_baselines"] = 0
    with pytest.raises(ValueError, match="exercised"):
        push_baseline_comparison(pair)


def test_luminance_breakdown_groups_dark_frames():
    from tools.frame_luminance_breakdown import breakdown, spearman
    manifest = [{"asset_id": f"f{i}", "hdr_mean_nits": n} for i, n in enumerate((0.6, 2.5, 15., 80., 150.))]
    report = {"step": 200, "frames": [
        {"asset_id": f"f{i}", "scene_id": "s", "condition": "real", "ev": None, "pu_gain_db": g,
         "crushed_gain_stops": None} for i, g in enumerate((-29., -22., -1., .2, 3.))]}
    out = breakdown(report, manifest)
    real = out["conditions"][0]
    assert real["bins"][0]["frames"] == 2 and real["bins"][0]["mean"] == -25.5
    assert real["excluding_dark_frames"]["frames"] == 3
    assert real["spearman_gain_vs_log_mean_nits"] == pytest.approx(1.0)
    assert real["worst"][0]["asset_id"] == "f0"
    assert spearman([1, 2], [1, 2]) is None
    with pytest.raises(ValueError, match="hdr_mean_nits"):
        breakdown(report, manifest[:2])


def test_mismatch_table_reproduces_the_diagnosis():
    from tools.frame_luminance_breakdown import baseline_mismatch
    rows = {r["ev"]: r for r in baseline_mismatch(nits=(1, 3, 30), evs=(0.0,))}
    err = rows[0.0]["error_stops"]
    assert err["bt1886"][0] < -5 and err["bt1886"][1] < -4      # dark pushes read far too dark on BT.1886
    assert all(abs(v) < .1 for v in err["aces"])                # their own render's inverse is near exact
    assert abs(err["bt1886"][2]) < .1                           # mid-tones agree


@pytest.mark.parametrize("corruption", [None, "reverse", "config", "count", "replay", "control_changed"])
def test_verifier_accepts_complete_pair_and_rejects_confounds(tmp_path, monkeypatch, corruption):
    from tools import verify_preservation_run as verifier
    from tests.test_preservation_verification import row
    from training.routing_replay import fingerprint

    def journal(path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    journal(manifest, [{"asset_id": "dark", "hdr_mean_nits": 0.6, "sdr_path": str(tmp_path / "s0"),
                        "hdr_path": str(tmp_path / "h0")},
                       {"asset_id": "bright", "hdr_mean_nits": 150., "sdr_path": str(tmp_path / "s1"),
                        "hdr_path": str(tmp_path / "h1")}])
    recipe = {"scope": PUSH_SCOPE, "output_root": str(tmp_path), "arm_steps": 2, "arm_seconds": 900,
              "storage_stop_bytes": 900000000, "manifest": str(manifest), "manifest_sha256": "synthetic",
              "code_sha256": {}}
    recipe_path, approval_path = tmp_path / "recipe.json", tmp_path / "approval.json"
    recipe_path.write_text(json.dumps(recipe)); approval_path.write_text("{}")
    monkeypatch.setattr(verifier, "approved_recipe", lambda *a: recipe)
    initial = {"weight": torch.zeros(1)}
    for arm in ("control", "render"):
        out = tmp_path / arm
        out.mkdir()
        config = {"training_use": "internal-pilot", "resume": None, "init_checkpoint": None,
                  "commercial_provenance": {"initialization": "scratch", "release_authorized": False},
                  "manifest_sha256": "synthetic", "crushed_hold_domain": "preclamp", "crushed_hold": 1.,
                  "preservation_diagnostics": True, "bench_push_pu": True, "region_focus": 0.,
                  "deterministic": True, "push_baseline": "picker" if arm == "control" else "render"}
        if arm == "render" and corruption == "config":
            config["push_baseline"] = "picker"
        for step, name in [(0, "step_0000000.pt"), (2, "step_0000002.pt"), (0, "best.pt")]:
            torch.save({"step": step, "config": config, "scheduler": {"T_max": 1000}, "model": initial}, out / name)
        trace = [{"draw": i, "record_index": i, "baseline_curve": idx("bt1886")} for i in range(8)]
        if arm == "render":
            for i in (0, 2, 4):
                trace[i]["baseline_curve"] = idx("aces")
            if corruption == "replay":
                trace[1]["record_index"] = 99
        if corruption == "reverse":
            trace[6]["baseline_curve"] = idx("aces") if arm == "control" else idx("bt1886")
        changed = 0 if arm == "control" else (2 if corruption == "count" else 3)
        if corruption == "control_changed" and arm == "control":
            changed = 1
        evs = [{"ev": ev, "clip_frames": 1, "crushed_frames": 1, "pu_frames": 2, "clip_gain_stops": -.1,
                "crushed_gain_stops": -.2 if arm == "control" else -.1, "pu_gain_db": -.3} for ev in (1, 2)]
        state = {"state": "completed", "last_step": 2, "sample_count": 8, "changed_baselines": changed,
                 "elapsed_seconds_after_preflight": 1, "initial_model_sha256": fingerprint(initial),
                 "held_channels_seen": 200, "negative_held_channels_seen": 40, "gradient_observations": 2,
                 "per_ev": evs, "metrics": {"bench_real_gain_db": -1. if arm == "control" else -.2}}
        (tmp_path / f"{arm}_status.json").write_text(json.dumps(state))
        (tmp_path / f"{arm}.log").write_text("verified 4/4 media files")
        journal(tmp_path / f"{arm}_samples.jsonl", trace)
        journal(out / "train.jsonl", [row(1), row(2)])
        dark = -29. if arm == "control" else -3.
        journal(out / "bench_val.jsonl", [{"step": 2, "per_ev": evs, "frames": [
            {"asset_id": "dark", "scene_id": "s", "condition": "real", "ev": None, "pu_gain_db": dark},
            {"asset_id": "bright", "scene_id": "s", "condition": "real", "ev": None, "pu_gain_db": 3.}]}])
    if corruption:
        with pytest.raises(ValueError):
            verifier.verify(recipe_path, approval_path)
        assert not (tmp_path / "verified_summary.json").exists()
        return
    verifier.verify(recipe_path, approval_path)
    result = json.loads((tmp_path / "verified_summary.json").read_text())
    assert result["changed_baselines"] == 3 and result["effect"]["early_hypothesis_supported"]
    assert result["effect"]["production_pass"] is False and result["release_authorized"] is False
    lum = result["luminance_breakdown"]
    assert lum["dark_real_frames"] == 1 and lum["dark_real_pu_delta_db"] == pytest.approx(26.)
    assert result["baseline_counts_by_arm"]["render"][str(idx("aces"))] == 3
