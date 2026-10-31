import torch
from types import SimpleNamespace
from training.aces_losses import shadow_relative_loss,shadow_reference_scale
from training.aces_losses import restoration_code_loss
from training.aces_reference import inverse_aces2_reference


def test_shadow_objective_has_signed_colour_gradients_and_ignores_highlights():
    prediction=torch.tensor([-.001,.002,.001]).reshape(1,3,1,1).requires_grad_()
    out=SimpleNamespace(acescg=prediction,reconstruction_weight=torch.ones(1,1,1,1))
    target=torch.zeros_like(prediction)
    loss=shadow_relative_loss(out,target,torch.zeros_like(prediction))
    loss.backward()
    assert shadow_reference_scale()>0 and loss>0
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0,0,0,0]<0 and prediction.grad[0,1,0,0]>0
    assert shadow_relative_loss(out,target,torch.ones_like(prediction))==0


def test_restoration_loss_uses_code_units_and_excludes_reconstruction_pixels():
    source=torch.full((1,3,1,2),.4)
    target=inverse_aces2_reference(source)
    restored=(source+.25/255).requires_grad_()
    out=SimpleNamespace(restored_source=restored,reconstruction_weight=torch.tensor([[[[0.,1.]]]]))
    loss=restoration_code_loss(out,target)
    torch.testing.assert_close(loss,torch.tensor(.25**2),atol=2e-4,rtol=0)
    loss.backward()
    assert (restored.grad[:,:,:,0]>0).all()
    assert (restored.grad[:,:,:,1]==0).all()
    out.reconstruction_weight.fill_(1)
    assert restoration_code_loss(out,target)==0
