import pytest
import torch
from training.aces_consistency import translate_constant,shadow_error_consistency
from training.aces_consistency import quantize_source8


def test_quantized_translation_stays_on_code_grid_and_preserves_endpoints():
    values=torch.linspace(0,1,1000).reshape(1,1,20,50)
    result=quantize_source8(values)
    assert result.min()==0 and result.max()==1
    torch.testing.assert_close(result*255,torch.round(result*255),rtol=0,atol=2e-5)
    assert (result-values).abs().max()<=.5/255+1e-7
    assert torch.equal(quantize_source8(result),result)
    with pytest.raises(ValueError):quantize_source8(values+1)
    with pytest.raises(ValueError):quantize_source8(values.requires_grad_())


def test_constant_translation_and_ramp_direction():
    constant=torch.ones(1,3,24,24)
    shifts=torch.tensor([[.5,-.5]])
    torch.testing.assert_close(translate_constant(constant,shifts),constant)
    ramp=torch.arange(24,dtype=torch.float32)[None,None,None,:].expand(1,3,24,24).clone()
    moved=translate_constant(ramp,shifts)
    torch.testing.assert_close(moved[:,:,:,2:-2],ramp[:,:,:,2:-2]+.5,rtol=1e-5,atol=1e-5)
    with pytest.raises(ValueError):translate_constant(constant.requires_grad_(),shifts)


def test_consistency_preserves_real_target_change_and_masks_empty_support():
    target=torch.full((1,3,24,24),.001);shift_target=target*2
    weight=torch.ones(1,1,24,24)
    assert shadow_error_consistency(target,target,shift_target,shift_target,weight,weight)==0
    pred=target.clone().requires_grad_()
    loss=shadow_error_consistency(pred,target,pred,shift_target,weight,weight)
    assert loss>0;loss.backward();assert torch.isfinite(pred.grad).all()
    assert shadow_error_consistency(pred,target,pred,shift_target,weight,weight*0)==0
