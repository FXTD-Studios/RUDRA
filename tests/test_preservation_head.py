import numpy as np
import torch
from pipeline.run05_corpus import canonical
from rudra.preservation_head import bounded_source_head
from rudra.sdr2hdr import SDR2HDRNet
from training.contract_checks import roundtrip


def test_large_predictions_cannot_regrade_ordinary_pixels():
    rng = np.random.default_rng(11)
    src = rng.uniform(.05, .90, (32, 48, 3)).astype(np.float32)
    canon = canonical(src)
    x = torch.from_numpy(canon).permute(2, 0, 1)[None]
    model = SDR2HDRNet(base_channels=8, baseline_source=True, corpus_ev=0)
    raw = torch.full_like(x, 100)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        y = bounded_source_head(x, raw, torch.full_like(x, 4), model.picker_baseline)
    result = roundtrip(y[0].permute(1, 2, 0).detach().numpy(), canon)
    assert result['r_pass'], result


def test_bounded_correction_has_useful_finite_gradient():
    x = torch.full((1, 3, 8, 8), .4)
    raw = torch.zeros_like(x, requires_grad=True)
    model = SDR2HDRNet(base_channels=8, baseline_source=True, corpus_ev=0)
    y = bounded_source_head(x, raw, model.picker_baseline(x), model.picker_baseline)
    y.sum().backward()
    assert torch.isfinite(raw.grad).all()
    assert (raw.grad.abs() > 0).all()


def test_exact_black_has_finite_transfer_and_bounded_head_gradients():
    from rudra.sdr2hdr import linear_to_srgb
    v = torch.tensor([0., .001, .1], requires_grad=True)
    linear_to_srgb(v).sum().backward()
    assert torch.isfinite(v.grad).all()
    assert v.grad[0] == v.grad[1]
    x = torch.zeros((1, 3, 4, 4))
    raw = torch.zeros_like(x, requires_grad=True)
    hdr = torch.full_like(x, .001, requires_grad=True)
    model = SDR2HDRNet(base_channels=8, baseline_source=True, corpus_ev=0)
    y = bounded_source_head(x, raw, hdr, model.picker_baseline, 'observed_clip_v1')
    y.sum().backward()
    assert torch.isfinite(raw.grad).all()
    assert raw.grad.count_nonzero() == 0
    assert torch.all(hdr.grad == 1)


def test_fully_clipped_region_keeps_hdr_prediction():
    x = torch.ones((1, 3, 8, 8))
    hdr = torch.full_like(x, .8, requires_grad=True)
    model = SDR2HDRNet(base_channels=8, baseline_source=True, corpus_ev=0)
    y = bounded_source_head(x, torch.zeros_like(x), hdr, model.picker_baseline)
    assert torch.allclose(y, hdr)
    y.sum().backward()
    assert torch.all(hdr.grad == 1)


def test_integrated_head_learns_and_checkpoint_reloads(tmp_path):
    config = dict(base_channels=8, baseline_source=True, bounded_source=True, corpus_ev=0)
    model = SDR2HDRNet.from_config(config)
    x = torch.full((1, 3, 16, 20), .4)
    output = model(x)
    output.hdr.sum().backward()
    assert model.dequant.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.dequant.bias.fill_(10)
        expected = model(x).hdr
    path = tmp_path / 'model.pt'
    torch.save(dict(config=config, model=model.state_dict()), path)
    saved = torch.load(path, weights_only=True)
    restored = SDR2HDRNet.from_config(saved['config'])
    restored.load_state_dict(saved['model'], strict=True)
    assert torch.equal(restored(x).hdr, expected)
    canon = x[0].permute(1, 2, 0).numpy()
    result = roundtrip(expected[0].permute(1, 2, 0).numpy(), canon)
    assert result['r_pass'], result


def test_legacy_config_has_no_extra_checkpoint_parameters():
    model = SDR2HDRNet.from_config(dict(base_channels=8, baseline_source=True))
    assert model.dequant is None
    assert not any(key.startswith('dequant.') for key in model.state_dict())


def test_observed_clip_preserves_nearwhite_colour_ramps_and_keeps_clipped_range():
    src = np.broadcast_to(np.linspace(240, 254, 64, dtype=np.float32)[None, :, None] / 255, (8, 64, 3)).copy()
    src[..., 0] *= .8
    x = torch.from_numpy(canonical(src)).permute(2, 0, 1)[None]
    model = SDR2HDRNet(base_channels=8, baseline_source=True, corpus_ev=0)
    extended = torch.ones_like(x)
    y = bounded_source_head(x, torch.full_like(x, 10), extended, model.picker_baseline, 'observed_clip_v1')
    result = roundtrip(y.detach()[0].permute(1, 2, 0).numpy(), canonical(src), policy='observed_clip_v1')
    assert result['r_pixels'] == 8 * 64
    assert result['r_pass'], result
    white = torch.ones_like(x)
    assert torch.allclose(bounded_source_head(white, white, extended, model.picker_baseline, 'observed_clip_v1'), extended)


def test_policy_is_checkpoint_config_and_legacy_defaults_remain():
    config = dict(base_channels=8, baseline_source=True, bounded_source=True, reconstruction_policy='observed_clip_v1')
    model = SDR2HDRNet.from_config(config)
    restored = SDR2HDRNet.from_config(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.reconstruction_policy == 'observed_clip_v1'
    assert SDR2HDRNet.from_config(dict(base_channels=8)).reconstruction_policy == 'legacy_nearwhite'


def test_feathered_tiles_preserve_source_with_spatial_corrections():
    from rudra.inference import predict_image
    rng = np.random.default_rng(29)
    src = rng.uniform(.03, .93, (40, 52, 3)).astype(np.float32)
    canon = canonical(src)
    x = torch.from_numpy(canon).permute(2, 0, 1)[None]
    torch.manual_seed(29)
    model = SDR2HDRNet(base_channels=8, baseline_source=True, bounded_source=True, corpus_ev=0)
    with torch.no_grad():
        model.dequant.weight.normal_(0, 10)
        model.head.weight.normal_(0, 1)
    y = predict_image(model.eval(), x, preserve_outside=True, tile_size=24, overlap=8, bf16=False)
    result = roundtrip(y[0].permute(1, 2, 0).numpy(), canon)
    assert result['r_pass'], result
