import json

import pytest
import torch

from training import freeze_candidate as fc


def setup(tmp_path, proposed="cand"):
    ckpt = tmp_path / "c.pt"
    torch.save({"model": {"w": torch.ones(3)}, "step": 7, "config": {"base_channels": 8}}, ckpt)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"candidates": {"cand": {"sha256": fc.sha256_file(ckpt), "clean": {}, "hard": {}}},
                                   "selection": {"proposed_for_next_stage": proposed}}))
    return ckpt, summary


def test_freeze_and_verify(tmp_path):
    ckpt, summary = setup(tmp_path)
    rec = fc.freeze(ckpt, summary, "cand", tmp_path / "f.json")
    assert rec["release_authorized"] is False and rec["parameters"] == 3 and rec["step"] == 7
    assert fc.verify(tmp_path / "f.json")["ok"]
    with pytest.raises(ValueError, match="never rewritten"):
        fc.freeze(ckpt, summary, "cand", tmp_path / "f.json")
    torch.save({"model": {"w": torch.zeros(3)}, "step": 7}, ckpt)
    assert fc.verify(tmp_path / "f.json")["problems"] == ["checkpoint bytes changed"]


def test_only_the_rules_proposal_is_frozen(tmp_path):
    ckpt, summary = setup(tmp_path, proposed=None)
    with pytest.raises(ValueError, match="not the frozen rule"):
        fc.freeze(ckpt, summary, "cand", tmp_path / "f.json")


def test_override_needs_reason_and_gate_pass(tmp_path):
    ckpt, summary = setup(tmp_path, proposed="other")
    with pytest.raises(ValueError, match="owner-decision"):
        fc.freeze(ckpt, summary, "cand", tmp_path / "f.json")
    with pytest.raises(ValueError, match="did not pass"):
        fc.freeze(ckpt, summary, "cand", tmp_path / "f.json", owner_decision="x" * 30)
    data = json.loads(summary.read_text())
    data["selection"]["ranking"] = ["other", "cand"]
    summary.write_text(json.dumps(data))
    rec = fc.freeze(ckpt, summary, "cand", tmp_path / "f.json", owner_decision="paired hard CI favours cand over other")
    assert rec["selection"]["owner_override"].startswith("paired")
