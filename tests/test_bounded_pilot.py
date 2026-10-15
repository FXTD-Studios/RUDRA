import json

import pytest

from training.bounded_pilot import Budget, PilotStop, smoke_gate
from pipeline.commercial_preflight import validate
from test_production_safeguards import approved_fixture


def test_smoke_and_continuation_time_limits():
    now = [0.0]
    b = Budget(lambda: now[0])
    now[0] = 9999; b.check()  # Preflight is disk work, outside GPU budget.
    b.start(); now[0] += 1800
    with pytest.raises(PilotStop, match="smoke"): b.check()
    b = Budget(lambda: now[0]); b.start(); now[0] += 100; b.extend()
    now[0] += 3600
    with pytest.raises(PilotStop, match="continuation"): b.check()


def good_metrics():
    return dict(bench_real_frames=48, bench_clip_frames=20, bench_crushed_frames=8,
                bench_real_gain_db=0.1, bench_clip_gain_stops=0, bench_crushed_gain_stops=0)


@pytest.mark.parametrize("key,value", [("bench_real_frames", 0), ("bench_crushed_frames", 0),
                                     ("bench_real_gain_db", 0), ("bench_clip_gain_stops", -0.01),
                                     ("bench_crushed_gain_stops", -0.01), ("bench_real_gain_db", float("nan"))])
def test_smoke_rejects_missing_coverage_or_regression(key, value):
    m = good_metrics(); m[key] = value
    with pytest.raises(PilotStop): smoke_gate(m)


def test_smoke_accepts_finite_improvement():
    smoke_gate(good_metrics())


def test_internal_pilot_does_not_require_or_grant_release(tmp_path):
    manifest, evidence, _, receipt = approved_fixture(tmp_path)
    receipt.update(scope="internal-pilot", release_authorized=False)
    receipt["sources"]["fixture"]["weight_distribution"] = False
    evidence.write_text(json.dumps(receipt))
    result = validate(manifest, evidence)
    assert result["scope"] == "internal-pilot" and result["release_authorized"] is False
    receipt["scope"] = "commercial-candidate"
    evidence.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="approval"): validate(manifest, evidence)


def test_internal_pilot_rejects_release_flag(tmp_path):
    manifest, evidence, _, receipt = approved_fixture(tmp_path)
    receipt.update(scope="internal-pilot", release_authorized=True)
    evidence.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="prohibit release"): validate(manifest, evidence)


def test_runner_stops_at_smoke_gate_and_retains_baseline(tmp_path, monkeypatch):
    import torch
    from training import bounded_pilot, train_sdr2hdr as trainer
    out = tmp_path / "run"
    original = trainer._atomic_save
    def fake_train(args, observer):
        observer("ready", 0, {})
        trainer._atomic_save({"step": 0, "model": {"weight": torch.zeros(1)}}, out / "best.pt")
        observer("eval", 0, good_metrics())
        m = good_metrics(); m["bench_real_gain_db"] = -0.01
        observer("eval", 200, m)
        pytest.fail("must not continue after failed smoke")
    monkeypatch.setattr(trainer, "train", fake_train)
    recipe = {"trainer_args": ["--manifest", "unused", "--commercial-evidence", "unused",
        "--output-dir", str(out), "--steps", "1000", "--workers", "0",
        "--eval-every", "200", "--save-every", "200"]}
    assert bounded_pilot.run(recipe) == 1
    assert (out / "step_0000000.pt").is_file()
    status = json.loads((out / "pilot_status.json").read_text())
    assert status["last_step"] == 200 and status["state"] == "stopped"
    assert not status["continuation_entered"]
    assert trainer._atomic_save is original
