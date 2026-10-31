import torch
import pytest
from training.aces_shadow import compose_shadow,luminance
from training.aces_model import ACESModel,model_from_checkpoint


def test_shadow_zero_fields_exact_baseline_signed_colour_and_nonnegative_luma():
    torch.manual_seed(8)
    baseline=torch.full((1,3,16,16),.001)
    baseline[:,0]=-.0001
    zero=torch.zeros_like(baseline);weight=torch.rand(1,1,16,16)
    assert torch.equal(compose_shadow(baseline,zero,weight),baseline)
    fields=torch.randn_like(baseline).requires_grad_()
    result=compose_shadow(baseline,fields,weight)
    assert luminance(result).min()>=-1e-8
    assert (result<0).any()
    result.square().mean().backward()
    assert torch.isfinite(fields.grad).all() and fields.grad.abs().max()>0


def test_black_can_recover_and_luminance_has_no_ceiling():
    base=torch.zeros(1,3,2,2);fields=torch.zeros_like(base)
    fields[:,0]=100
    result=compose_shadow(base,fields,torch.ones(1,1,2,2))
    assert luminance(result).min()>1


@pytest.mark.parametrize('shadow',['sqrt_luma','sqrt_luma_context'])
def test_v5_preserves_ordinary_and_highlight_outputs_and_checkpoint_roundtrip(shadow):
    torch.set_num_threads(2);torch.manual_seed(4)
    old=ACESModel(4,'pixel_rms','local_codes').eval()
    new=ACESModel(4,'pixel_rms','local_codes',shadow).eval()
    new.predictor.load_state_dict(old.predictor.state_dict());new.restorer.load_state_dict(old.restorer.state_dict())
    source=torch.full((1,3,24,24),.5);source[:,:,:8]=1;source[:,:,-8:]=0
    with torch.no_grad():
        new.shadow_predictor.layers[-1].bias.fill_(.2)
        a=old(source);b=new(source)
    assert torch.equal(a.acescg[:,:,:16],b.acescg[:,:,:16])
    assert not torch.equal(a.acescg[:,:,-8:],b.acescg[:,:,-8:])
    loaded=model_from_checkpoint({'model':new.state_dict(),'contract':new.contract()})
    assert loaded.contract()==new.contract()
