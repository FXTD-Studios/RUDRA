import torch
from training.aces_detail_loss import highlight_gradient_loss


def test_missing_texture_penalized_and_signed_gradient_finite():
    target=torch.tensor([-2.,-1.,0.,1.,2.]).reshape(1,1,1,5).expand(1,3,3,5)
    prediction=torch.zeros_like(target,requires_grad=True)
    mask=torch.ones((1,1,3,5),dtype=torch.bool)
    loss=highlight_gradient_loss(prediction,target,mask)
    assert loss>0
    loss.backward();assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum()>0
    assert highlight_gradient_loss(target,target,mask)==0


def test_excluded_boundary_and_empty_support_cannot_supply_edges():
    target=torch.zeros((1,3,2,4));prediction=target.clone();prediction[:,:,:,2:]=10
    mask=torch.zeros((1,1,2,4),dtype=torch.bool);mask[:,:,:,:2]=True
    assert highlight_gradient_loss(prediction,target,mask)==0
    assert highlight_gradient_loss(prediction,target,mask&False)==0
