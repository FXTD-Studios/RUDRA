import json
from pathlib import Path

import pytest
import torch

from tests.test_preclamp_hold import preservation_recipe
from tests.test_routing_ab import RandomPairs
from training.routing_ab import arm_args, region_comparison, run_arm


def recipe_at(path):
    recipe = preservation_recipe(path)
    recipe["scope"] = "internal-region-focus-ab"
    return recipe


@pytest.mark.parametrize("extra", [["--region-focus", ".5"], ["--crushed-hold", "2"],
                                    ["--bench-push-pu"], ["--bench-val-push", "2"],
                                    ["--crushed-hold-domain", "preclamp"]])
def test_region_recipe_rejects_confounds(tmp_path, extra):
    recipe = recipe_at(tmp_path)
    recipe["trainer_args"] += extra
    with pytest.raises(ValueError):
        arm_args(recipe, "control")


@pytest.mark.parametrize("missing_pu", [False, True])
def test_region_pair_runs_real_trainer_with_identical_inputs(tmp_path, monkeypatch, missing_pu):
    from training import train_sdr2hdr as trainer, bench_val
    recipe = recipe_at(tmp_path)
    Path(recipe["manifest"]).write_text('{"sdr_kind":"real"}\n')
    monkeypatch.setattr("pipeline.commercial_preflight.training_preflight",
                        lambda args: {"scope": "internal-pilot", "release_authorized": False})
    class DarkPairs(RandomPairs):
        def __getitem__(self, i):
            sample = super().__getitem__(i)
            sample["sdr"][:, :8, :8] = 1 / 255
            return sample
    monkeypatch.setattr(trainer, "SDRHDRDataset", DarkPairs)
    monkeypatch.setattr(bench_val.BenchVal, "from_manifest", classmethod(lambda cls, *a, **k: cls([{}])))
    def evaluate(self, model, device):
        assert self.push_pu
        self.last_report = {"per_ev": [{"ev": ev, "clip_frames": 1, "crushed_frames": 1,
            "pu_frames": 0 if missing_pu else 1, "pu_gain_db": None if missing_pu else -2.,
            "clip_gain_stops": -.1, "crushed_gain_stops": -.2} for ev in (1, 2)]}
        return {"bench_real_frames": 1, "bench_clip_frames": 1, "bench_crushed_frames": 1,
                "bench_real_gain_db": -1., "bench_gain": -1.}
    monkeypatch.setattr(bench_val.BenchVal, "evaluate", evaluate)
    configs, states, traces = [], [], []
    for arm in ("control", "region"):
        code = run_arm(recipe, arm)
        state = json.loads((tmp_path / f"{arm}_status.json").read_text())
        if missing_pu:
            assert code == 1 and "pushed full-frame PU21" in state["reason"]
            assert not (tmp_path / "region").exists()
            return
        assert code == 0 and state["last_step"] == 2 and state["sample_count"] == 8
        timing = [json.loads(line) for line in (tmp_path / f"{arm}_timing.jsonl").read_text().splitlines()]
        assert [(r['event'], r['step']) for r in timing] == [('eval', 0), ('eval', 2)]
        assert 0 <= timing[0]['elapsed_seconds_after_preflight'] <= timing[1]['elapsed_seconds_after_preflight']
        saved = torch.load(tmp_path / arm / "step_0000002.pt", weights_only=False)
        assert saved["scheduler"]["T_max"] == 1000
        assert saved["config"]["crushed_hold_domain"] == "preclamp"
        assert saved["config"]["region_focus"] == (0 if arm == "control" else .5)
        logs = [json.loads(line) for line in (tmp_path / arm / "train.jsonl").read_text().splitlines()]
        training = [row for row in logs if "total" in row]
        assert len(training) == 2
        if arm == "region":
            assert all(row["region_fraction"] > 0 and row["region"] > 0 for row in training)
        else:
            assert all(row["region_fraction"] == 0 for row in training)
        configs.append(saved["config"])
        states.append(state)
        traces.append((tmp_path / f"{arm}_samples.jsonl").read_bytes())
    assert traces[0] == traces[1]
    assert states[0]["initial_model_sha256"] == states[1]["initial_model_sha256"]
    ignored = {"output_dir", "preservation_ab_arm", "region_focus"}
    assert {k:v for k,v in configs[0].items() if k not in ignored} == {k:v for k,v in configs[1].items() if k not in ignored}


def test_region_verdict_reports_tradeoff_without_promotion():
    def state(real, value):
        return {"metrics": {"bench_real_gain_db": real}, "per_ev": [
            {"ev": ev, "clip_frames": 1, "crushed_frames": 1, "pu_frames": 2,
             "clip_gain_stops": value, "crushed_gain_stops": value, "pu_gain_db": value} for ev in (1,2)]}
    pair = {"control": state(-3., -2.), "region": state(-2., -1.)}
    result = region_comparison(pair)
    assert result["early_hypothesis_supported"] and not result["production_pass"]
    pair["region"]["per_ev"][0]["pu_gain_db"] = -3
    assert not region_comparison(pair)["early_hypothesis_supported"]
    pair["region"]["per_ev"][0]["pu_frames"] = 1
    with pytest.raises(ValueError, match="coverage"):
        region_comparison(pair)
