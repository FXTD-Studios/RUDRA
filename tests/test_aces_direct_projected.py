import torch
import pytest
from training.aces_direct_projected import direct_projected_step


def test_direction_and_requested_length():
    p=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64));before=p.detach().clone()
    result=direct_projected_step([p],before,torch.tensor([3.,4.],dtype=p.dtype),.1)
    assert torch.allclose(p,torch.tensor([-.06,-.08],dtype=p.dtype))
    assert result['applied_l2']<=.1 and not result['zero_direction']


def test_zero_direction_no_movement():
    p=torch.nn.Parameter(torch.ones(2));before=p.detach().clone()
    result=direct_projected_step([p],before,torch.zeros(2),1e-4)
    assert torch.equal(p,before) and result['zero_direction']


def test_float32_movement_bound():
    p=torch.nn.Parameter(torch.linspace(.1,1,100));before=p.detach().clone()
    result=direct_projected_step([p],before,torch.ones(100),1e-4)
    assert float((p.detach().double()-before.double()).norm())<=1e-4
    assert result['applied_l2']>0


def test_invalid_direction():
    p=torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError):direct_projected_step([p],p.detach().clone(),torch.tensor([float('nan')]),1e-4)
