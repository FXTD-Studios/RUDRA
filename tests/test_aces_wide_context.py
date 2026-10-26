import pytest
import torch
from training.aces_full_context_model import ACESFullContextModel,fully_clipped_support
from training.aces_wide_context_model import ACESWideContextModel
from training.aces_model import model_from_checkpoint


@pytest.mark.parametrize('mode',['local5','coarse64_global_mean','coarse4_spatial'])
def test_zero_identity_update_outside_support_and_roundtrip(mode):
    torch.manual_seed(49);parent=ACESFullContextModel(8).eval();model=ACESWideContextModel.from_parent(parent,mode).eval()
    source=torch.rand(1,3,65,81)*.8;source[:,:,20:50,25:60]=1
    with torch.no_grad():before=parent(source);initial=model(source)
    assert torch.equal(before.acescg,initial.acescg)
    full=fully_clipped_support(source,before.reconstruction_weight).expand_as(source)
    opt=torch.optim.Adam(model.wide_context.parameters(),lr=.001)
    out=model(source);loss=(out.acescg[full]-before.acescg[full]-1).square().mean();loss.backward();opt.step()
    with torch.no_grad():after=model(source)
    assert not torch.equal(after.acescg[full],before.acescg[full])
    assert torch.equal(after.acescg[~full],before.acescg[~full])
    for k,v in parent.state_dict().items():assert torch.equal(v,model.state_dict()[k])
    assert all(p.requires_grad==n.startswith('wide_context.') for n,p in model.named_parameters())
    restored=model_from_checkpoint({'model':model.state_dict(),'contract':model.contract()}).eval()
    with torch.no_grad():assert torch.equal(restored(source).acescg,after.acescg)


@pytest.mark.parametrize('mode',['coarse64_global_mean','coarse4_spatial'])
def test_coarse_crop_uses_full_frame_conditioning_exactly(mode):
    model=ACESWideContextModel(8,mode).eval()
    with torch.no_grad():model.wide_context.head.weight.normal_(0,.02)
    x=torch.rand(1,3,321,515);base=torch.rand_like(x)
    condition=model.conditioning(x,base);full=model.context_fields(condition,(321,515));crop=model.context_fields(condition,(321,515),(64,68,128,192))
    assert torch.equal(crop,full[:,:,64:192,68:260])


def test_global_path_reaches_distant_pixels_local_control_does_not():
    for mode in ('local5','coarse64_global_mean'):
        model=ACESWideContextModel(8,mode).eval()
        with torch.no_grad():
            for p in model.wide_context.parameters():p.fill_(.01)
        x=torch.ones(1,3,256,256,requires_grad=True);base=torch.ones_like(x)
        fields=model.context_fields(model.conditioning(x,base),(256,256))
        fields[0,0,128,128].backward();outside=x.grad[:,:,0:32,0:32].abs().sum()
        assert bool(outside>0)==(mode=='coarse64_global_mean')


def test_spatial_bottleneck_remote_influence_varies_by_position():
    model=ACESWideContextModel(8,'coarse4_spatial').eval()
    with torch.no_grad():
        for p in model.wide_context.parameters():p.fill_(.01)
        source=torch.ones(1,3,256,256);base=torch.ones_like(source)
        before=model.context_fields(model.conditioning(source,base),(256,256))
        source[:,:,:32,:32]=.1
        after=model.context_fields(model.conditioning(source,base),(256,256))
    delta=(after-before)[0,:,128,128]
    assert delta.abs().max()>0
    assert not torch.equal(delta,(after-before)[0,:,224,224])
