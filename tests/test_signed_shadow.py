import torch
from training.signed_shadow import SignedShadowLoss
from training.reconstruction_gradient import ReconstructionGradientLoss


def test_flat_false_colour_is_not_invisible_to_anchored_loss():
    source = torch.zeros(1, 3, 8, 9)
    target = torch.zeros_like(source)
    pred = torch.zeros_like(source)
    pred[:, 2] = .001
    pred.requires_grad_()
    # Spatial derivatives alone cannot identify a constant colour offset.
    spatial = ReconstructionGradientLoss()(pred, target, source)['total']
    assert spatial.item() < 1e-9
    loss = SignedShadowLoss()(pred, target, source)
    grad, = torch.autograd.grad(loss, pred)
    assert loss.item() > 0 and torch.isfinite(grad).all()
    assert grad[:, 2].mean() > 0
    corrected = pred.detach() - grad * (1e-5 / grad.abs().max())
    assert SignedShadowLoss()(corrected, target, source) < loss


def test_target_identity_and_no_dark_support():
    objective = SignedShadowLoss()
    target = torch.rand(1, 3, 7, 9) * 1e-5
    pred = target.clone().requires_grad_()
    loss = objective(pred, target, torch.zeros_like(target))
    assert loss.item() == 0
    assert torch.isfinite(torch.autograd.grad(loss, pred)[0]).all()
    assert objective(pred + .001, target, torch.ones_like(target)).item() == 0
