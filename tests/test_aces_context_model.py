import torch
from training.aces_model import ACESModel,reconstruction_weight,model_from_checkpoint
from training.aces_context_model import ACESContextModel,aligned_pair_residual_loss


def test_highlight_training_cannot_change_ordinary_or_shadow_output():
    torch.manual_seed(17);parent=ACESModel(8,'pixel_rms','local_codes','analytic').eval()
    model=ACESContextModel.from_parent(parent).eval()
    source=torch.rand(1,3,32,40)*.8+.1;source[:,:,:4]=0;source[:,:,-8:]=1
    with torch.no_grad():
        original=parent(source).acescg
        assert torch.equal(original,model(source).acescg)
        model.highlight_context.head.bias[3:].add_(.2)
        changed=model(source).acescg
    high=(reconstruction_weight(source)>0)&(source.amax(1,keepdim=True)>.5)
    assert torch.equal(original[~high.expand_as(original)],changed[~high.expand_as(changed)])
    assert not torch.equal(original[high.expand_as(original)],changed[high.expand_as(changed)])
    assert all(p.requires_grad==name.startswith('highlight_context.') for name,p in model.named_parameters())


def test_aligned_error_loss_ignores_truth_motion_and_detects_flicker():
    target=torch.rand(2,3,12,16);mask=torch.ones((2,1,12,16),dtype=torch.bool)
    assert aligned_pair_residual_loss(target,target,mask)==0
    residual=torch.full_like(target,.2)
    consistent=torch.sinh(torch.asinh(target)+residual)
    assert aligned_pair_residual_loss(consistent,target,mask)<1e-12
    flicker=consistent.clone();flicker[1]+=1;flicker.requires_grad_(True)
    loss=aligned_pair_residual_loss(flicker,target,mask);assert loss>.01
    loss.backward();assert torch.isfinite(flicker.grad).all()
    assert aligned_pair_residual_loss(flicker,target,mask&False)==0


def test_known_translation_alignment_direction_and_empty_pair_count():
    y,x=torch.meshgrid(torch.arange(16),torch.arange(20),indexing='ij')
    a=.01*x+.02*y;b=.01*(x-.5)+.02*(y-.25)
    residual=torch.stack([a,b])[:,None].expand(2,3,16,20)
    prediction=torch.sinh(residual);truth=torch.zeros_like(prediction)
    mask=torch.ones((2,1,16,20),dtype=torch.bool)
    loss,count=aligned_pair_residual_loss(prediction,truth,mask,return_count=True)
    assert loss<1e-12 and count>0
    loss,count=aligned_pair_residual_loss(prediction,truth,mask&False,return_count=True)
    assert loss==0 and count==0


def test_checkpoint_factory_restores_independent_branch_exactly():
    model=ACESContextModel(8).eval()
    with torch.no_grad():model.highlight_context.head.bias[3:].add_(.17)
    restored=model_from_checkpoint({'model':model.state_dict(),'contract':model.contract()}).eval()
    source=torch.ones(1,3,16,20)
    with torch.no_grad():assert torch.equal(model(source).acescg,restored(source).acescg)
    assert restored.contract()==model.contract()
