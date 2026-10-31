import json

import numpy as np
import pytest
import torch

from training import resumable_training as rt
from training.dark_experiment import ShadowCodecAugment, near_black_log_loss, pu_min_score
from tests.test_resumable_training import argv, corpus, load, same_tensors, same_optimizer, one_thread  # noqa: F401

EXP = ["--shadow-codec-p", "0.5", "--near-black-weight", "0.05", "--selection", "pu_min"]


def test_augment_touches_only_shadows_and_is_reproducible():
    x = torch.rand(2, 3, 32, 32) * 0.05            # deep shadows
    x[:, :, :, 16:] = 0.8                          # bright half
    a, b = ShadowCodecAugment(1.0, seed=3), ShadowCodecAugment(1.0, seed=3)
    ya, yb = a(x), b(x)
    assert torch.equal(ya, yb)
    assert torch.allclose(ya[..., 20:], x[..., 20:], atol=1e-6)      # highlights untouched
    assert (ya[..., :12] - x[..., :12]).abs().mean() > 1e-3          # shadows degraded
    s = a.state(); a2 = ShadowCodecAugment(1.0, seed=99); a2.restore(s)
    assert torch.equal(a(x), a2(x))
    assert torch.equal(ShadowCodecAugment(0.0)(x), x)


def test_near_black_loss_penalises_lifted_coloured_blacks_only():
    target = torch.full((1, 3, 8, 8), 1e-6)
    sdr = torch.full((1, 3, 8, 8), 2 / 255)
    assert float(near_black_log_loss(target.clone(), target, sdr)) == pytest.approx(0.0, abs=1e-6)
    blob = target.clone(); blob[:, 0] *= 1000; blob[:, 1] *= 800       # +10 stops, coloured
    assert float(near_black_log_loss(blob, target, sdr)) > 5
    bright_sdr = torch.full((1, 3, 8, 8), 0.5)
    assert float(near_black_log_loss(blob, target, bright_sdr)) == 0.0  # outside near-black: no effect


def test_pu_min_score():
    assert pu_min_score({"clean_pu_gain_db": 1.0, "hard_pu_gain_db": 0.2}) == -0.2


def test_near_black_loss_recovers_real_detail_instead_of_forcing_black():
    target = torch.full((1, 3, 8, 8), 0.01 / 10000)
    sdr = torch.full_like(target, 2 / 255)
    pred = torch.zeros_like(target, requires_grad=True)
    near_black_log_loss(pred, target, sdr).backward()
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad < 0).all()  # gradient descent must increase a crushed prediction
    pred = torch.full_like(target, 0.5 / 10000, requires_grad=True)
    near_black_log_loss(pred, target, sdr).backward()
    assert (pred.grad > 0).all()  # visible lift must decrease toward the true target


def test_experiment_run_resumes_bitwise_and_records_config(tmp_path):
    m = corpus(tmp_path / "c")
    ref, out = tmp_path / "ref", tmp_path / "run"
    mine, args = rt.parse(argv(m, ref, *EXP))
    rt.train(args, experiment=rt.experiment_of(mine))
    mine, args = rt.parse(argv(m, out, *EXP))
    first = rt.train(args, experiment=rt.experiment_of(mine), control=rt.Control(out / "STOP", None, stop_at_step=5))
    assert first["state"] == "stopped_resumable"
    mine, args = rt.parse(argv(m, out, *EXP))
    rt.train(args, resume=True, experiment=rt.experiment_of(mine))
    a, b = load(ref / "step_0000009.pt"), load(out / "step_0000009.pt")
    same_tensors(a["model"], b["model"]); same_optimizer(a["optimizer"], b["optimizer"])
    assert a["config"]["experiment"] == {"shadow_codec_p": 0.5, "near_black_weight": 0.05,
                                         "near_black_code": 16, "selection": "pu_min"}
    logs = [json.loads(l) for l in (ref / "train.jsonl").read_text().splitlines()]
    assert any("near_black" in r for r in logs)
    # resuming with different experiment settings is refused
    mine, args = rt.parse(argv(m, out, "--shadow-codec-p", "0.2"))
    with pytest.raises(rt.ResumeError, match="configuration differs"):
        rt.train(args, resume=True, experiment=rt.experiment_of(mine))
