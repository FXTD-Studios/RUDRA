import torch
from training.aces_context_model import ACESContextModel
from training.aces_full_context_model import ACESFullContextModel,fully_clipped_support
from training.aces_model import model_from_checkpoint


def test_initial_equivalence_and_outside_full_invariance_after_optimizer_step():
    torch.manual_seed(8);parent=ACESContextModel(8).eval()
    with torch.no_grad():parent.highlight_context.head.weight.normal_(0,.01)
    model=ACESFullContextModel.from_parent(parent).eval()
    source=torch.rand(1,3,33,41)*.8+.1;source[:,:,:5]=0;source[:,:,12:24,10:25]=1;source[:,0,5:10]=1
    with torch.no_grad():before=parent(source);initial=model(source)
    assert torch.equal(before.acescg,initial.acescg)
    assert torch.equal(before.fields,initial.fields)
    full=fully_clipped_support(source,before.reconstruction_weight).expand_as(source)
    optimizer=torch.optim.Adam(model.full_context.parameters(),lr=.001)
    result=model(source);loss=(result.acescg[full]-(before.acescg[full]+1)).square().mean()
    optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():after=model(source)
    assert torch.equal(after.acescg[~full],before.acescg[~full])
    assert not torch.equal(after.acescg[full],before.acescg[full])
    for k,v in parent.state_dict().items():assert torch.equal(v,model.state_dict()[k])
    assert all(p.requires_grad==n.startswith('full_context.') for n,p in model.named_parameters())
    restored=model_from_checkpoint({'model':model.state_dict(),'contract':model.contract()}).eval()
    with torch.no_grad():assert torch.equal(restored(source).acescg,after.acescg)


def test_full_support_requires_all_channels_and_boundary_is_declared():
    source=torch.tensor([[[[1.,1.,254.49/255,254.5/255]],[[1.,.9,1.,1.]],[[1.,1.,1.,1.]]]])
    assert fully_clipped_support(source,torch.ones(1,1,1,4)).flatten().tolist()==[True,False,False,True]
    assert not fully_clipped_support(source,torch.zeros(1,1,1,4)).any()


def test_boundary_diagnostic_detects_cross_region_seam():
    from tools.diagnose_aces_full_branch import boundary_error
    truth=torch.zeros(2,3,16,20);high=torch.ones(2,1,16,20,dtype=torch.bool);full=high.clone();full[:,:,:,:10]=False
    same=torch.sinh(torch.ones_like(truth)*.2)
    assert boundary_error(same,truth,high,full)==0
    same[:,:,:,10:]+=1
    assert boundary_error(same,truth,high,full)>0


def test_full_only_temporal_mask_can_miss_newly_clipped_region():
    from training.aces_context_model import aligned_pair_residual_loss
    truth=torch.zeros(2,3,24,32)
    full=torch.zeros(2,1,24,32,dtype=torch.bool);full[:,:,6:18,5:10]=True
    full[1,:,6:18,19:25]=True
    prediction=truth.clone();prediction[1,:,7:17,20:24]=1;prediction.requires_grad_(True)
    assert aligned_pair_residual_loss(prediction,truth,full)==0
    high=torch.ones_like(full)
    loss=aligned_pair_residual_loss(prediction,truth,high)
    assert loss>0
    loss.backward()
    assert prediction.grad[1,:,7:17,20:24].abs().sum()>0
