import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from rudra.source_baseline import baseline_curve_index as idx
from training.routing_replay import ReplayDataset, augmentation_rng, fingerprint
from training.routing_ab import approved_recipe, arm_args, check_limits, run, run_arm


class RandomPairs:
    def __init__(self, *args, **kwargs):
        self.augment = kwargs.get("augment", True)
        self.records = [{"sdr_kind": "real", "scene_id": "shot", "asset_id": str(i)} for i in range(4)]

    def __len__(self):
        return 4

    def __getitem__(self, i):
        value = random.random() + float(np.random.random())
        return {"sdr": torch.rand(3, 16, 16) * 0.5 + value * 0.05,
                "hdr": torch.full((3, 16, 16), 0.1), "source_curve": torch.tensor(0),
                "baseline_curve": torch.tensor(idx("aces")), "real": torch.tensor(1),
                "asset_id": str(i), "scene_id": "shot", "ceiling": torch.tensor(float("inf"))}


def test_rng_scope_restores_streams_and_repeats_inputs():
    random.seed(3); np.random.seed(3); torch.manual_seed(3)
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    with augmentation_rng(23):
        first = RandomPairs()[0]
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.random.get_rng_state(), before[2])
    with augmentation_rng(23):
        assert fingerprint(first) == fingerprint(RandomPairs()[0])


def test_replay_matches_repeated_draws_and_only_changes_baseline(tmp_path):
    control_path, corrected_path = tmp_path / "c.jsonl", tmp_path / "t.jsonl"
    control = ReplayDataset(RandomPairs(), "control", 7, control_path)
    samples = [control[i] for i in (2, 2, 0)]
    control.finish(3); control.close()
    corrected = ReplayDataset(RandomPairs(), "corrected", 7, corrected_path, control_path)
    try:
        for i, sample in zip((2, 2, 0), samples):
            other = corrected[i]
            torch.testing.assert_close(sample["sdr"], other["sdr"], rtol=0, atol=0)
            assert int(sample["baseline_curve"]) == idx("bt1886")
            assert int(other["baseline_curve"]) == idx("aces")
        corrected.finish(3)
        assert corrected.changed_baselines == 3
        assert not torch.equal(samples[0]["sdr"], samples[1]["sdr"])
    finally:
        corrected.close()


@pytest.mark.parametrize("change", ["seed", "order", "tensor", "short", "long"])
def test_replay_rejects_mismatched_inputs_or_coverage(tmp_path, change):
    control_path = tmp_path / "c.jsonl"
    control = ReplayDataset(RandomPairs(), "control", 7, control_path)
    control[0]; control.finish(1); control.close()
    data = RandomPairs()
    if change == "tensor":
        class Changed(RandomPairs):
            def __getitem__(self, i):
                sample = super().__getitem__(i)
                sample["hdr"] *= 2
                return sample
        data = Changed()
    corrected = ReplayDataset(data, "corrected", 8 if change == "seed" else 7,
                              tmp_path / "t.jsonl", control_path)
    try:
        with pytest.raises(ValueError, match="mismatch|coverage|ended"):
            if change == "short":
                corrected.finish(1)
            else:
                corrected[1 if change == "order" else 0]
                if change == "long":
                    corrected[0]
    finally:
        corrected.close()


def test_pending_approval_rejects_before_parsing_recipe_or_creating_output(tmp_path):
    approval = tmp_path / "approval.json"
    approval.write_text('{"approved":false}')
    with pytest.raises(ValueError, match="explicit approval"):
        run(tmp_path / "does-not-exist.json", approval)
    assert list(tmp_path.iterdir()) == [approval]


def recipe_fixture(tmp_path):
    return {"output_root": str(tmp_path), "manifest": str(tmp_path / "manifest.jsonl"),
            "evidence": "synthetic-test-only", "arm_steps": 2, "arm_seconds": 600,
            "storage_stop_bytes": 900_000_000,
            "trainer_args": ["--mode", "image", "--steps", "1000", "--batch-size", "4",
                             "--workers", "0", "--baseline-source", "--real-corpus-ev", "0",
                             "--eval-every", "2", "--save-every", "2", "--best-metric", "bench_gain",
                             "--device", "cpu", "--crop-size", "16", "--base-channels", "8",
                             "--eval-batches", "1", "--bench-val-frames", "1"]}


@pytest.mark.parametrize("extra", [["--workers", "1"], ["--steps", "200"],
                                    ["--init-checkpoint", "old.pt"], ["--grad-accum", "2"]])
def test_runner_rejects_unbounded_or_confounded_recipe(tmp_path, extra):
    recipe = recipe_fixture(tmp_path)
    recipe["trainer_args"] += extra
    with pytest.raises(ValueError, match="requires scratch"):
        arm_args(recipe, "control")


def test_time_and_shared_storage_limits(tmp_path):
    check_limits(tmp_path, None, 600, 100, now=10000)  # Disk preflight is excluded.
    check_limits(tmp_path, 10, 600, 100, now=609)
    with pytest.raises(TimeoutError):
        check_limits(tmp_path, 10, 600, 100, now=610)
    (tmp_path / "control.log").write_bytes(b"x" * 100)
    with pytest.raises(RuntimeError, match="artifact"):
        check_limits(tmp_path, None, 600, 100)


@pytest.mark.parametrize("change", [None, "code", "manifest", "receipt", "missing_pin", "release"])
@pytest.mark.parametrize("scope", ["internal-routing-ab", "internal-preservation-ab"])
def test_approval_pins_and_internal_scope(tmp_path, monkeypatch, change, scope):
    from training import routing_ab as ab
    from pipeline.commercial_preflight import sha256
    monkeypatch.setattr(ab, "REPO", tmp_path)
    monkeypatch.setattr(ab, "CODE_FILES", ("runner.py",))
    code, manifest, receipt = [tmp_path / name for name in ("runner.py", "manifest.jsonl", "review.json")]
    code.write_text("frozen code")
    manifest.write_text("frozen manifest")
    evidence = {"scope": "internal-pilot", "release_authorized": False,
                "reviewed_by": "synthetic test", "reviewed_at": "test-only",
                "sources": {"fixture": {"weight_distribution": False}}}
    if change == "release":
        evidence["sources"]["fixture"]["weight_distribution"] = True
    receipt.write_text(json.dumps(evidence))
    recipe = {"scope": scope, "release_authorized": False,
              "output_root": str(tmp_path / "reports" / "new"), "arm_steps": 200,
              "arm_seconds": 600, "storage_stop_bytes": 900_000_000,
              "code_sha256": {"runner.py": sha256(code)},
              "manifest": str(manifest), "manifest_sha256": sha256(manifest),
              "evidence": str(receipt), "evidence_sha256": sha256(receipt)}
    if change == "missing_pin":
        recipe["code_sha256"] = {}
    path, approval = tmp_path / "recipe.json", tmp_path / "approval.json"
    path.write_text(json.dumps(recipe))
    approval.write_text(json.dumps({"approved": True, "approved_by": "synthetic test",
                                   "approved_at": "test-only", "release_authorized": False,
                                   "recipe_sha256": sha256(path)}))
    for name, target in (("code", code), ("manifest", manifest), ("receipt", receipt)):
        if change == name:
            target.write_text("changed")
    if change is None:
        assert approved_recipe(path, approval) == recipe
    else:
        with pytest.raises(ValueError, match="changed|pins|internal-only"):
            approved_recipe(path, approval)


def test_parent_kills_worker_on_limit_and_never_starts_second_arm(tmp_path, monkeypatch):
    from training import routing_ab as ab
    recipe = recipe_fixture(tmp_path / "run")
    monkeypatch.setattr(ab, "approved_recipe", lambda *a: recipe)
    events = []

    class HungChild:
        returncode = None
        def __init__(self, *a, **kw): events.append("spawn")
        def poll(self): return self.returncode
        def kill(self): self.returncode = -9; events.append("kill")
        def wait(self): events.append("wait")

    def exceeded(*a, **kw): raise TimeoutError("synthetic budget")
    monkeypatch.setattr(ab.subprocess, "Popen", HungChild)
    monkeypatch.setattr(ab, "check_limits", exceeded)
    assert run(tmp_path / "recipe", tmp_path / "approval") == 1
    assert events == ["spawn", "kill", "wait"]
    assert "synthetic budget" in (tmp_path / "run/stopped.json").read_text()


def test_two_synthetic_arms_share_inputs_weights_and_stop_at_bound(tmp_path, monkeypatch):
    from training import train_sdr2hdr as trainer
    from training import bench_val
    recipe = recipe_fixture(tmp_path)
    Path(recipe["manifest"]).write_text(json.dumps({"sdr_kind": "real"}) + "\n")
    monkeypatch.setattr("pipeline.commercial_preflight.training_preflight",
                        lambda args: {"scope": "internal-pilot", "release_authorized": False})
    monkeypatch.setattr(trainer, "SDRHDRDataset", RandomPairs)

    def evaluate(self, model, device):
        self.last_report = {"per_ev": [{"ev": 1, "clip_frames": 1, "crushed_frames": 1}]}
        return {"bench_real_frames": 1, "bench_clip_frames": 1, "bench_crushed_frames": 1,
                "bench_real_gain_db": -1., "bench_gain": -1.}

    monkeypatch.setattr(bench_val.BenchVal, "from_manifest", classmethod(lambda cls, *a, **k: cls([{}])))
    monkeypatch.setattr(bench_val.BenchVal, "evaluate", evaluate)
    original_save = trainer._atomic_save
    for arm in ("control", "corrected"):
        assert run_arm(recipe, arm) == 0
        state = json.loads((tmp_path / f"{arm}_status.json").read_text())
        assert state["state"] == "completed" and state["last_step"] == 2
        assert state["sample_count"] == 8
        saved = torch.load(tmp_path / arm / "step_0000002.pt", weights_only=False)
        assert saved["scheduler"]["T_max"] == 1000
        assert saved["config"]["routing_ab_arm"] == arm
        assert saved["config"]["training_use"] == "internal-pilot"
        assert trainer.SDRHDRDataset is RandomPairs and trainer._atomic_save is original_save
    control, corrected = [json.loads((tmp_path / f"{arm}_status.json").read_text())
                          for arm in ("control", "corrected")]
    assert control["initial_model_sha256"] == corrected["initial_model_sha256"]
    assert corrected["changed_baselines"] == 8
