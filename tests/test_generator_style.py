import numpy as np
import pytest
import torch

from training.generator_style import DEFAULT_OPS, GeneratorStyleAugment


def _batch(seed=0, b=3, h=48, w=64):
    g = torch.Generator().manual_seed(seed)
    ramp = torch.linspace(0, 1, w).view(1, 1, 1, w).expand(b, 3, h, w)
    return (0.8 * ramp + 0.2 * torch.rand(b, 3, h, w, generator=g)).clamp(0, 1)


def test_p_zero_is_identity():
    x = _batch()
    assert torch.equal(GeneratorStyleAugment(0.0)(x), x)


def test_output_range_and_shape():
    y = GeneratorStyleAugment(1.0, seed=3)(_batch())
    assert y.shape == (3, 3, 48, 64) and float(y.min()) >= 0 and float(y.max()) <= 1


def test_input_not_modified():
    x = _batch(); keep = x.clone()
    GeneratorStyleAugment(1.0, seed=1)(x)
    assert torch.equal(x, keep)


def test_resume_state_round_trip():
    a = GeneratorStyleAugment(0.7, seed=11)
    a(_batch(1)); saved = a.state()
    first = a(_batch(2))
    b = GeneratorStyleAugment(0.7, seed=999); b.restore(saved)
    assert torch.equal(b(_batch(2)), first)


def test_clip_only_caps_highlights_and_keeps_midtones():
    aug = GeneratorStyleAugment(1.0, ops={k: (1.0 if k == "highlight_clip" else 0.0) for k in DEFAULT_OPS}, seed=0)
    img = np.full((8, 8, 3), 0.5, np.float32); img[:4] = 1.0
    out, applied = aug._one(img)
    assert applied == ["highlight_clip"]
    assert out[:4].max() <= 0.98 and out[:4].min() >= 0.90
    assert np.allclose(out[4:], 0.5)


def test_rejects_unknown_op_and_bad_p():
    with pytest.raises(ValueError):
        GeneratorStyleAugment(0.5, ops={"exposure": 0.5})
    with pytest.raises(ValueError):
        GeneratorStyleAugment(1.5)


def test_rng_advance_independent_of_which_ops_fire():
    # Same seed, different op probabilities -> the generator ends in the same state.
    a = GeneratorStyleAugment(1.0, seed=5)
    b = GeneratorStyleAugment(1.0, ops={k: 0.0 for k in DEFAULT_OPS}, seed=5)
    a(_batch()); b(_batch())
    assert a.state() == b.state()


def test_trainer_records_generator_style_flag():
    pytest.skip('run04 (pre-registered, not started): the trainer does not take --generator-style-p yet')
    from training import resumable_training as rt
    mine, _ = rt.parse(["--generator-style-p", "0.3", "--manifest", "m.jsonl", "--output-dir", "o"])
    assert rt.experiment_of(mine)["generator_style_p"] == 0.3
    assert "training/generator_style.py" in rt.CODE_FILES
