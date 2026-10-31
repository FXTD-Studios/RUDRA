import numpy as np
import torch
from tools.audit_aces_learned_context import regional
from training.aces_detail_loss import highlight_gradient_loss


def test_region_gradient_matches_training_objective():
    rng = np.random.default_rng(42)
    pred = rng.normal(size=(13, 17, 3)).astype(np.float32)
    target = rng.normal(size=pred.shape).astype(np.float32)
    mask = rng.random(pred.shape[:2]) > .3
    result = regional(pred, target, mask)
    tensor = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None]
    expected = highlight_gradient_loss(tensor(pred), tensor(target), torch.from_numpy(mask)[None, None])
    np.testing.assert_allclose(result['gradient_asinh_mse'], float(expected), rtol=2e-6)
    assert result['pixels'] == int(mask.sum())


def test_region_empty_identity_and_no_cross_boundary_edges():
    target = np.zeros((4, 5, 3), np.float32)
    mask = np.zeros((4, 5), bool)
    assert regional(target, target, mask) is None
    mask[1, 1:3] = True
    pred = np.ones_like(target) * 99
    pred[mask] = 0
    result = regional(pred, target, mask)
    assert result['edges'] == 1
    assert result['photo_asinh_mse'] == result['gradient_asinh_mse'] == 0


def test_constant_region_standard_deviation_uses_stable_accumulation():
    pred = np.full((256, 512, 3), 123.456, np.float32)
    result = regional(pred, pred, np.ones((256, 512), bool))
    assert max(result['pred_asinh_rgb_std']) < 1e-12
