import numpy as np
import pytest
import torch
pytest.importorskip('PyOpenColorIO')
from training.aces_reference import inverse_aces2_reference, reference_view


def test_exact_forward_and_directional_backward():
    source = torch.tensor([[[[.2,.4]],[[.3,.5]],[[.4,.6]]]],requires_grad=True)
    result = inverse_aces2_reference(source)
    exact = reference_view().inverse(source.detach().permute(0,2,3,1).numpy())
    np.testing.assert_array_equal(result.detach().permute(0,2,3,1).numpy(),exact)
    weight = torch.tensor([.2,.5,.3]).view(1,3,1,1)
    gradient, = torch.autograd.grad((result*weight).sum(),source)
    direction = torch.tensor([.1,-.2,.05]).view(1,3,1,1).expand_as(source)
    epsilon=.002
    with torch.no_grad():
        direct=((inverse_aces2_reference(source+epsilon*direction)-inverse_aces2_reference(source-epsilon*direction))*weight).sum()/(2*epsilon)
    torch.testing.assert_close((gradient*direction).sum(),direct,rtol=.03,atol=.0001)


def test_endpoints_have_finite_first_derivatives():
    source=torch.tensor([[[[0.,1.]],[[0.,1.]],[[0.,1.]]]],requires_grad=True)
    grad,=torch.autograd.grad(inverse_aces2_reference(source).sum(),source)
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize('step',[0.,-1.,float('nan'),.1])
def test_invalid_step_rejected(step):
    with pytest.raises(ValueError):
        inverse_aces2_reference(torch.zeros(1,3,2,2),step)
