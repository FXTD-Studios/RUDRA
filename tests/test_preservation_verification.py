"""Reject incomplete diagnostic evidence and use pixel-weighted fractions."""
import pytest
import json
import torch

from tools.verify_preservation_run import training_summary


def row(step, held=100, negative=20):
    return {"step": step, "held_channels": held, "negative_held_channels": negative,
            "negative_held_fraction": negative / max(1, held), "gradient_norm_before_clip": 3.,
            "hold_preclamp": .1, "hold_postclamp": .01, "total": .2}


def test_summary_weights_region_coverage_and_excludes_evaluation_rows():
    result = training_summary([row(1, 100, 100), {"step": 1, "eval": {}}, row(2, 900, 0)], 2)
    assert result["overall"]["negative_held_fraction"] == .1
    assert result["overall"]["steps"] == 2
    assert result["windows"][0]["first_step"] == 1
    assert result["windows"][0]["last_step"] == 2


@pytest.mark.parametrize("change", ["missing", "duplicate", "nonfinite", "counts", "fraction"])
def test_corrupt_diagnostic_evidence_rejected(change):
    rows = [row(1), row(2)]
    if change == "missing": rows.pop()
    if change == "duplicate": rows[1]["step"] = 1
    if change == "nonfinite": rows[1]["total"] = float("nan")
    if change == "counts": rows[1]["negative_held_channels"] = 101
    if change == "fraction": rows[1]["negative_held_fraction"] = .5
    with pytest.raises(ValueError):
        training_summary(rows, 2)


@pytest.mark.parametrize("corruption", [None, "hold", "region", "reporting", "replay"])
@pytest.mark.parametrize("repeat", [False, True])
def test_region_verifier_accepts_complete_pair_and_rejects_confounds(tmp_path, monkeypatch, corruption, repeat):
    from tools import verify_preservation_run as verifier
    from training.routing_replay import fingerprint
    def write(path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")
    def journal(path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    journal(manifest, [{"sdr_path": str(tmp_path / "sdr.png"), "hdr_path": str(tmp_path / "hdr.png")}])
    recipe = {"scope": "internal-region-focus-ab", "output_root": str(tmp_path), "arm_steps": 2,
              "arm_seconds": 600, "storage_stop_bytes": 900000000, "manifest": str(manifest),
              "manifest_sha256": "synthetic", "code_sha256": {}}
    if repeat:
        recipe["scope"] = "internal-repeatability"
    candidate = "repeat" if repeat else "region"
    recipe_path, approval_path = tmp_path / "recipe.json", tmp_path / "approval.json"
    write(recipe_path, recipe); write(approval_path, {"synthetic": True})
    monkeypatch.setattr(verifier, "approved_recipe", lambda *a: recipe)
    initial = {"weight": torch.zeros(1)}
    evs = [{"ev": ev, "clip_frames": 1, "crushed_frames": 1, "pu_frames": 1,
            "clip_gain_stops": -.1, "crushed_gain_stops": -.2, "pu_gain_db": -.3} for ev in (1, 2)]
    for arm in ("control", candidate):
        out = tmp_path / arm
        out.mkdir()
        config = {"training_use": "internal-pilot", "resume": None, "init_checkpoint": None,
                  "commercial_provenance": {"initialization": "scratch", "release_authorized": False},
                  "manifest_sha256": "synthetic", "crushed_hold_domain": "preclamp", "crushed_hold": 1.,
                  "preservation_diagnostics": True, "bench_push_pu": True,
                  "region_focus": 0. if arm == "control" or repeat else .5}
        if repeat:
            config.update(deterministic=True, runtime={"deterministic_algorithms": True})
        if arm == candidate:
            if corruption == "hold": config["crushed_hold_domain"] = "postclamp"
            if corruption == "region": config["region_focus"] = .25
            if corruption == "reporting": config["bench_push_pu"] = False
        for step, name in [(0,"step_0000000.pt"),(2,"step_0000002.pt"),(0,"best.pt")]:
            torch.save({"step": step, "config": config, "scheduler": {"T_max": 1000}, "model": initial},out/name)
        state = {"state": "completed", "last_step": 2, "sample_count": 8, "changed_baselines": 0,
                 "elapsed_seconds_after_preflight": 1, "initial_model_sha256": fingerprint(initial),
                 "held_channels_seen": 200, "negative_held_channels_seen": 40, "gradient_observations": 2,
                 "per_ev": evs, "metrics": {"bench_real_gain_db": -1.}}
        write(tmp_path / f"{arm}_status.json",state)
        (tmp_path / f"{arm}.log").write_text("verified 2/2 media files")
        trace = [{"draw": i, "record_index": i, "baseline_curve": 1} for i in range(8)]
        if arm == candidate and corruption == "replay":trace[0]["record_index"] = 99
        journal(tmp_path / f"{arm}_samples.jsonl",trace)
        journal(out / "train.jsonl",[row(1),row(2)])
        journal(out / "bench_val.jsonl",[{"step": 2, "per_ev": evs, "frames": [
            {"asset_id": "a", "scene_id": "scene", "condition": "real", "ev": None,"pu_gain_db": -1.}]}])
    if corruption:
        with pytest.raises(ValueError):verifier.verify(recipe_path,approval_path)
        assert not (tmp_path / "verified_summary.json").exists()
    else:
        verifier.verify(recipe_path,approval_path)
        result = json.loads((tmp_path / "verified_summary.json").read_text())
        assert result["matched_draws"] == 8 and result["effect"]["production_pass"] is False
        if repeat:
            assert result["effect"]["repeatability_pass"]
