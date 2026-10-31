import copy
import torch
from tools.pilot_aces_protected_trajectory import restore,equal_state


def test_rejection_restores_persistent_moments_weights_and_next_update():
    p=torch.nn.Parameter(torch.tensor([1.,2.]));opt=torch.optim.Adam([p],lr=.01)
    p.grad=torch.tensor([.2,.3]);opt.step();weights=p.detach().clone();saved=copy.deepcopy(opt.state_dict())
    p.grad=torch.tensor([-9.,7.]);opt.step();restore([p],opt,weights,saved)
    assert torch.equal(p,weights) and equal_state(opt.state_dict(),saved) and p.grad is None
    q=torch.nn.Parameter(weights.clone());control=torch.optim.Adam([q],lr=.01);control.load_state_dict(copy.deepcopy(saved))
    for param,o in ((p,opt),(q,control)):param.grad=torch.tensor([.7,-.2]);o.step()
    assert torch.equal(p,q) and equal_state(opt.state_dict(),control.state_dict())


def test_first_rejection_restores_empty_optimizer_state():
    p=torch.nn.Parameter(torch.ones(2));opt=torch.optim.Adam([p]);saved=copy.deepcopy(opt.state_dict());weights=p.detach().clone()
    p.grad=torch.ones_like(p);opt.step();restore([p],opt,weights,saved)
    assert not opt.state and torch.equal(p,weights)
