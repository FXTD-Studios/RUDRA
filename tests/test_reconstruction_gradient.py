import torch
from training.reconstruction_gradient import ReconstructionGradientLoss


def test_reference_identity_zero_and_missing_support_zero():
    fn=ReconstructionGradientLoss()
    source=torch.full((1,3,12,16),.4)
    output=torch.full_like(source,.002,requires_grad=True)
    target=torch.full_like(source,.003)
    absent=fn(output,target,source)
    assert absent['support_fraction']==0 and absent['total']==0
    source[:,:,:,8:]=1
    exact=fn(target,target,source)
    assert exact['total']==0 and exact['support_fraction']>0


def test_boundary_error_has_finite_corrective_gradient():
    fn=ReconstructionGradientLoss()
    source=torch.full((1,3,12,16),.4)
    source[:,:,:,8:]=1
    target=torch.full_like(source,.001)
    target[:,:,:,8:]=.002
    output=target.clone()
    output[:,:,:,8:]=.004
    output.requires_grad_()
    loss=fn(output,target,source)['total']
    gradient=torch.autograd.grad(loss,output)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    rate=1e-7 / gradient.abs().max()
    assert fn(output-rate*gradient,target,source)['total']<loss
