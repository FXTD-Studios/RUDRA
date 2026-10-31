import numpy as np
import pytest
import torch
pytest.importorskip('PyOpenColorIO')
from training.aces_model import ACESModel,compose
from training.aces_reference import reference_view


def test_zero_initialization_is_exact_analytic_inverse_including_saturated_green():
    torch.set_num_threads(2)
    source=torch.full((1,3,17,21),.4)
    source[:,:,0,0]=torch.tensor([0.,1.,0.])
    model=ACESModel(4).eval()
    with torch.no_grad(): output=model(source)
    torch.testing.assert_close(output.acescg,output.baseline,rtol=0,atol=0)
    restored=reference_view().forward(output.acescg.permute(0,2,3,1).numpy())
    assert np.max(np.abs(restored-source.permute(0,2,3,1).numpy()))*255 < 1
    assert model.contract()['output']=='ACEScg linear'


def test_ordinary_source_bound_and_residual_independence():
    source=torch.full((1,3,7,9),.5)
    fields=torch.full((1,6,7,9),100.)
    first=compose(source,fields)
    fields[:,3:]=-100
    second=compose(source,fields)
    assert (first.restored_source-source).abs().max() <= .49001/255
    assert first.reconstruction_weight.max()==0
    torch.testing.assert_close(first.acescg,second.acescg,rtol=0,atol=0)


def test_reconstruction_can_recover_black_and_exceed_white_without_output_clamp():
    source=torch.zeros(1,3,3,3)
    fields=torch.zeros(1,6,3,3);fields[:,3:]=1
    black=compose(source,fields)
    assert (black.acescg>0).all()
    white=compose(torch.ones_like(source),fields)
    assert (white.acescg>white.baseline).all()


def test_actual_model_step_receives_finite_corrective_gradients():
    torch.manual_seed(20261023);torch.set_num_threads(2)
    model=ACESModel(4)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    source=torch.full((1,3,12,12),.4)
    output=model(source)
    target=(output.baseline.detach()*1.002)
    loss=(torch.asinh(output.acescg)-torch.asinh(target)).square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.predictor.head.weight.grad.abs().max()>0
    optimizer.step()
    after=(torch.asinh(model(source).acescg)-torch.asinh(target)).square().mean()
    assert after.detach() < loss.detach()


def test_signed_reconstruction_has_no_multiplicative_floor_or_stop_ceiling():
    source=torch.ones(1,3,2,2)
    fields=torch.zeros(1,6,2,2,requires_grad=True)
    with torch.no_grad():
        fields[:,3]=-2
        fields[:,4]=32
    output=compose(source,fields)
    assert (output.acescg[:,0]<0).all()
    assert (output.acescg[:,1]>16*output.baseline[:,1]).all()
    output.acescg.sum().backward()
    assert torch.isfinite(fields.grad).all()
    assert (fields.grad[:,3:] > 0).all()
    assert ACESModel(4).contract()['residual']=='signed_linear_shared_rgb_scale'


def test_pixel_normalization_fields_ignore_distant_context_with_same_local_pixels():
    torch.manual_seed(12);torch.set_num_threads(2)
    model=ACESModel(4,normalization='pixel_rms').eval()
    torch.nn.init.normal_(model.predictor.head.weight,std=.05)
    source=torch.rand(1,3,384,384);baseline=source.clone()
    changed=source.clone();changed[:,:,:64]=20;changed[:,:,-64:]=-20
    with torch.no_grad():
        a=model.predictor(source,baseline)
        b=model.predictor(changed,baseline)
    torch.testing.assert_close(a[:,:,160:224,160:224],b[:,:,160:224,160:224],rtol=0,atol=0)
    assert model.contract()['normalization']=='pixel_rms'
