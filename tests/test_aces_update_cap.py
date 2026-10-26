import pytest
import torch
from training.aces_update_cap import cap_update


def test_cap_direction_and_noop():
    p=torch.nn.Parameter(torch.tensor([3.,4.],dtype=torch.float64));before=torch.zeros(2,dtype=p.dtype)
    info=cap_update([p],before,1.)
    assert torch.allclose(p,torch.tensor([.6,.8],dtype=p.dtype)) and info['applied_l2']<=1.
    unchanged=p.detach().clone();info=cap_update([p],before,2.)
    assert torch.equal(p,unchanged) and info['scale']==1.


def test_float32_actual_cap_and_optimizer_state_untouched():
    p=torch.nn.Parameter(torch.ones(100));opt=torch.optim.Adam([p],lr=.01)
    before=p.detach().clone();p.grad=torch.ones_like(p);opt.step()
    moments={k:v.clone() for k,v in opt.state[p].items()}
    info=cap_update([p],before,1e-4)
    assert float((p.detach().double()-before.double()).norm())<=1e-4 and info['scale']<1
    assert all(torch.equal(v,opt.state[p][k]) for k,v in moments.items())


@pytest.mark.parametrize('limit',[0,-1,float('nan'),float('inf')])
def test_invalid_cap(limit):
    with pytest.raises(ValueError):cap_update([torch.nn.Parameter(torch.ones(1))],torch.zeros(1),limit)
