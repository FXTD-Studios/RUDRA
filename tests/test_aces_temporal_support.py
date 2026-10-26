import torch
from tools.audit_aces_temporal_support import supports,active_loss
from training.aces_context_model import aligned_pair_residual_loss
from training.aces_active_temporal import active_temporal_loss


def test_transition_union_and_frozen_only_gradient_scaling():
    torch.manual_seed(5)
    high=torch.ones(2,1,24,32,dtype=torch.bool);full=torch.zeros_like(high)
    full[0,:,7:16,5:12]=True;full[1,:,7:16,10:18]=True
    masks=supports(high,full)
    assert masks['entry'].any() and masks['exit'].any() and masks['stable_full'].any()
    assert torch.equal(masks['active'],masks['stable_full']|masks['entry']|masks['exit'])
    parameter=torch.randn(2,3,24,32,requires_grad=True);pred=torch.where(full,parameter,torch.ones_like(parameter))
    target=torch.zeros_like(pred);a=aligned_pair_residual_loss(pred,target,high);b=active_loss(pred,target,masks)
    ga=torch.autograd.grad(a,parameter,retain_graph=True)[0];gb=torch.autograd.grad(b,parameter)[0]
    assert torch.allclose(ga,gb*(masks['active'].sum()/masks['valid'].sum()),atol=1e-8)
    assert ga[~full.expand_as(ga)].count_nonzero()==0
    actual,count,total=active_temporal_loss(pred,target,high,full)
    assert torch.allclose(actual,b) and count==int(masks['active'].sum()) and total==int(masks['valid'].sum())


def test_no_full_support_has_zero_active_loss_and_gradient():
    high=torch.ones(2,1,16,20,dtype=torch.bool);masks=supports(high,torch.zeros_like(high))
    x=torch.ones(2,3,16,20,requires_grad=True);loss=active_loss(x,torch.zeros_like(x),masks)
    assert loss==0 and not masks['active'].any()
    loss.backward();assert x.grad.count_nonzero()==0
    actual,count,_=active_temporal_loss(x,torch.zeros_like(x),high,torch.zeros_like(high))
    assert actual==0 and count==0
