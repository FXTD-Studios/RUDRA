"""Diagnostic validity: independent native forwards and analytic gradients."""
from dataclasses import replace

import pytest
import torch

from rudra.sdr2hdr import SDR2HDRNet
from rudra.source_baseline import baseline_curve_index
from tools.diagnose_preservation import (aggregate, dark_stats, diagnostic_variants,
                                        gradient_probe, reconstruct, score)


def pilot():
    torch.manual_seed(713)
    model = SDR2HDRNet(base_channels=8, baseline_source=True, corpus_ev=0).eval().requires_grad_(False)
    with torch.no_grad():
        model.head.weight.normal_(0, .002)
        model.head.bias[:3].copy_(torch.tensor([-.01, .03, -.02]))
    return model


@pytest.mark.parametrize("curve", ["bt1886", "aces"])
def test_variants_match_independent_model_forwards(curve):
    model = pilot()
    x = torch.linspace(0, 1, 32 * 32).reshape(1, 1, 32, 32).expand(1, 3, 32, 32)
    curve = baseline_curve_index(curve)
    with torch.no_grad():
        output = model(x, baseline_curve=curve)
        variants, _ = diagnostic_variants(output, x, model)
        for name, kwargs in (("original", {}), ("existing_highlights_mode", {"recovery_mode": "highlights"}),
                             ("existing_preserve_outside", {"preserve_outside": True})):
            torch.testing.assert_close(variants[name], model(x, baseline_curve=curve, **kwargs).hdr, rtol=0, atol=0)


def test_post_clamp_hold_cannot_repair_negative_residual_but_pre_clamp_can():
    model = pilot()
    x = torch.full((1, 3, 32, 32), 1 / 255)
    with torch.no_grad():
        out = model(x, baseline_curve=baseline_curve_index("aces"))
    out = replace(out, log_residual=torch.full_like(out.log_residual, -.1))
    result = gradient_probe(out, x, torch.full_like(x, .00001), model)
    assert result["negative_held_channels"] == x.numel()
    assert result["region_focus_exempts_held_negative_channels"] == x.numel()
    for term in ("post_clamp_hold", "pu21_mse", "log_l1"):
        assert result["terms"][term]["negative_held_gradient_l1"] == 0
    assert result["terms"]["diagnostic_pre_clamp_hold"]["negative_held_nonzero_gradients"] == x.numel()


def test_pre_and_post_clamp_hold_agree_when_no_saturation():
    model = pilot()
    x = torch.full((1, 3, 32, 32), 1 / 255)
    with torch.no_grad():
        out = model(x, baseline_curve=baseline_curve_index("aces"))
    out = replace(out, log_residual=torch.full_like(out.log_residual, .01))
    result = gradient_probe(out, x, torch.full_like(x, .00001), model)
    a, b = [result["terms"][k] for k in ("post_clamp_hold", "diagnostic_pre_clamp_hold")]
    assert result["negative_channels"] == 0
    assert a["value"] == pytest.approx(b["value"], rel=1e-5)
    assert a["gradient_l1"] == pytest.approx(b["gradient_l1"], rel=1e-5)


def test_empty_regions_are_missing_not_zero_gain():
    x = torch.full((1, 3, 32, 32), .5)
    ref, base, pred = x * .1, x * .09, x * .11
    values = score(pred, base, ref, x)
    assert values["clip_gain_stops"] is None and values["crushed_gain_stops"] is None
    result = aggregate([{"arm": "test", "condition": "real", "ev": None,
                         "variants": {"original": values}, "dark": {"pixels": 0}}])[0]
    assert result["variants"]["original"]["crushed_gain_stops"] == {"mean": None, "frames": 0}


def test_dark_counts_distinguish_channel_saturation_and_rgb_black():
    sdr = torch.zeros(1, 3, 2, 2)
    pred = torch.zeros_like(sdr)
    pred[0, 0, 0, 0] = .001
    pre = torch.full_like(sdr, -.1)
    pre[0, 0, 0, 0] = .1
    result = dark_stats(pred, sdr, torch.full_like(sdr, .0001), sdr, pre)
    assert result["pixels"] == 4
    assert result["negative_channels"] == 11
    assert result["zero_rgb_pixels"] == 3
    assert result["black_but_reference_above_floor_pixels"] == 3


def test_unsupported_model_rejected():
    model = SDR2HDRNet(base_channels=8, gate_conditioning=True).eval()
    with pytest.raises(ValueError, match="frozen pilot"):
        reconstruct(None, torch.zeros(1, 3, 32, 32), model)
