"""Task 1.3: the source-curve input to SDR2HDRNet's CurveHead.

The contract: "unknown" reproduces today's blind model exactly, a blind
curve_head checkpoint warm-starts the source-aware model, and a model without
the input refuses a known curve instead of ignoring it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rudra.sdr2hdr import (  # noqa: E402
    KNOWN_SOURCE_CURVES, SOURCE_CURVES, SDR2HDRNet, source_curve_index,
)


def _sdr(b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, 64, 96, generator=g)


def _randomise(model: torch.nn.Module, seed=1):
    """Non-zero weights everywhere, so 'identical' cannot pass by all-zeros."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def _pair():
    blind = SDR2HDRNet(base_channels=8, corpus_ev=0.0, curve_head=True)
    _randomise(blind)
    aware = SDR2HDRNet(base_channels=8, corpus_ev=0.0, curve_head=True, source_curve=True)
    aware.load_state_dict(blind.state_dict(), strict=True)  # warm start pads the new columns
    return blind.eval(), aware.eval()


def test_vocabulary_is_fixed():
    assert SOURCE_CURVES[0] == "unknown"
    assert SOURCE_CURVES == ("unknown", "aces", "hable", "reinhard", "agx", "camera_log", "clip")
    assert KNOWN_SOURCE_CURVES == 6
    assert source_curve_index(None) == 0 and source_curve_index("hable") == 2
    with pytest.raises(ValueError):
        source_curve_index("filmic")


def test_unknown_is_bit_identical_to_the_blind_model():
    blind, aware = _pair()
    x = _sdr()
    with torch.no_grad():
        ref = blind(x).hdr
        assert torch.equal(aware(x).hdr, ref)
        assert torch.equal(aware(x, source_curve=torch.zeros(2, dtype=torch.long)).hdr, ref)
        assert torch.equal(aware.predict_curve(x), blind.predict_curve(x))


def test_known_curve_starts_at_blind_and_can_move():
    blind, aware = _pair()
    x = _sdr()
    hable = torch.full((2,), source_curve_index("hable"), dtype=torch.long)
    with torch.no_grad():
        # zero-initialised source columns: a known curve starts where blind is
        assert torch.equal(aware(x, source_curve=hable).hdr, blind(x).hdr)
        aware.curve.mlp[0].weight[:, -KNOWN_SOURCE_CURVES:].normal_(0, 0.5)
        moved = aware(x, source_curve=hable).hdr
        still_blind = aware(x, source_curve=torch.zeros(2, dtype=torch.long)).hdr
    assert not torch.equal(moved, still_blind)
    assert torch.equal(still_blind, blind(x).hdr)


def test_per_sample_ids_and_broadcast():
    _, aware = _pair()
    with torch.no_grad():
        aware.curve.mlp[0].weight[:, -KNOWN_SOURCE_CURVES:].normal_(0, 0.5)
        x = _sdr()
        mixed = aware.predict_curve(x, torch.tensor([0, 4]))
        # batched vs single-sample convs differ in the last bits (~7e-9, the
        # blind model does the same), so this one compares with a tolerance
        assert torch.allclose(mixed[0], aware.predict_curve(x[:1])[0], atol=1e-6, rtol=0)
        assert not torch.allclose(mixed[1], aware.predict_curve(x[1:2])[0], atol=1e-6, rtol=0)
        one = aware.predict_curve(x, torch.tensor([3]))
        assert torch.equal(one, aware.predict_curve(x, torch.tensor([3, 3])))
    with pytest.raises(ValueError):
        aware.predict_curve(_sdr(), torch.tensor([0, 7]))
    with pytest.raises(ValueError):
        aware.predict_curve(_sdr(b=3), torch.tensor([1, 2]))


def test_models_without_the_input_refuse_a_known_curve():
    blind, _ = _pair()
    x = _sdr()
    with torch.no_grad():
        blind(x, source_curve=torch.zeros(2, dtype=torch.long))  # unknown is fine
        with pytest.raises(ValueError):
            blind(x, source_curve=torch.tensor([2, 2]))
        plain = SDR2HDRNet(base_channels=8)
        with pytest.raises(ValueError):
            plain(x, source_curve=torch.tensor([1, 0]))
    with pytest.raises(ValueError):
        SDR2HDRNet(base_channels=8, source_curve=True)  # needs the CurveHead


def test_config_round_trip_and_old_checkpoints():
    m = SDR2HDRNet.from_config({"base_channels": 8, "curve_head": True, "source_curve": True})
    assert m.source_curve and m.curve.source_inputs == KNOWN_SOURCE_CURVES
    # a pre-1.3 config (no key) builds exactly the old architecture
    old = SDR2HDRNet.from_config({"base_channels": 8, "curve_head": True})
    assert not old.source_curve and old.curve.source_inputs == 0
    assert set(old.state_dict()) == set(m.state_dict())
