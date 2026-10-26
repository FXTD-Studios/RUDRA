import numpy as np
import torch

from pipeline.run05_corpus import canonical, declared_inverse
from training.contract_checks import canonical_to_source, view_forward, roundtrip
from training.dequantization import DequantizationLoss
from rudra.preservation_head import bounded_source_head
from rudra.sdr2hdr import SDR2HDRNet


def tensor(a):
    return torch.from_numpy(np.asarray(a, np.float32)).permute(2, 0, 1)[None]


def test_forward_view_matches_independent_numpy_checker():
    src = np.random.default_rng(7).uniform(0, 1, (16, 32, 3)).astype(np.float32)
    radiance = declared_inverse(canonical(src), 'bt1886')
    objective = DequantizationLoss()
    actual = objective.source_code(objective.forward_view(tensor(radiance)))[0].permute(1, 2, 0).numpy()
    expected = canonical_to_source(view_forward(radiance))
    np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_optimizing_bounded_corrections_recovers_subcode_ramp():
    torch.set_num_threads(2)
    src = np.broadcast_to(np.linspace(.18, .42, 256, dtype=np.float32)[None, :, None], (8, 256, 3)).copy()
    quantized = np.rint(src * 255) / 255
    # Model can legally use only +/-0.49 code; target respects this contract.
    src = np.clip(src, quantized - .49 / 255, quantized + .49 / 255)
    x = tensor(canonical(quantized))
    target = tensor(declared_inverse(canonical(src), 'bt1886'))
    model = SDR2HDRNet(base_channels=8, baseline_source=True, corpus_ev=0)
    raw = torch.zeros_like(x, requires_grad=True)
    objective = DequantizationLoss()
    optimizer = torch.optim.Adam([raw], lr=.1)
    history = []
    for _ in range(60):
        optimizer.zero_grad()
        output = bounded_source_head(x, raw, target, model.picker_baseline)
        losses = objective(output, target, x)
        losses['total'].backward()
        assert torch.isfinite(raw.grad).all()
        optimizer.step()
        history.append(float(losses['total'].detach()))
    assert history[-1] < history[0] * .03
    result = roundtrip(output.detach()[0].permute(1, 2, 0).numpy(), x[0].permute(1, 2, 0).numpy())
    assert result['r_pass']


def test_does_not_supervise_destroyed_range_or_nan_on_empty_mask():
    objective = DequantizationLoss()
    for code in (0., 1.):
        x = torch.full((1, 3, 4, 4), code)
        output = torch.full_like(x, .01, requires_grad=True)
        loss = objective(output, torch.ones_like(x), x)['total']
        loss.backward()
        assert loss.item() == 0
        assert torch.isfinite(output.grad).all()
        assert output.grad.count_nonzero() == 0


def test_noise_is_penalized_not_rewarded_as_debanding():
    torch.manual_seed(4)
    objective = DequantizationLoss()
    src = np.full((16, 32, 3), .35, np.float32)
    x = tensor(canonical(src))
    target = tensor(declared_inverse(canonical(src), 'bt1886'))
    noisy = tensor(declared_inverse(canonical(src + np.random.default_rng(5).uniform(-.001, .001, src.shape).astype(np.float32)), 'bt1886'))
    clean = objective(target, target, x)
    perturbed = objective(noisy, target, x)
    assert clean['total'] == 0
    assert perturbed['code'] > 0
    assert perturbed['gradient'] > 0


def test_black_prediction_in_ordinary_region_has_finite_recovery_gradient():
    objective = DequantizationLoss()
    x = torch.full((1, 3, 4, 4), .4)
    output = torch.zeros_like(x, requires_grad=True)
    loss = objective(output, torch.full_like(x, .002), x)['total']
    loss.backward()
    assert torch.isfinite(output.grad).all()
    assert output.grad.abs().sum() > 0
