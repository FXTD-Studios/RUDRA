import copy
import torch
from training.aces_protected_update import guarded_update
from tools.pilot_aces_protected_trajectory import equal_state


def test_replay_accepts_photo_improvement_without_temporal_requirement():
    p=torch.nn.Parameter(torch.tensor([2.]));opt=torch.optim.Adam([p],lr=.1)
    def terms():return [p.square().sum(),(p-1).square().sum()]
    r=guarded_update([p],opt,terms(),terms,'projected_search',[1,.5])
    assert r['accepted'] and p.item()<2 and opt.state[p]['step']==1


def test_conflicting_replay_rejection_restores_existing_moments():
    p=torch.nn.Parameter(torch.tensor([2.]));opt=torch.optim.Adam([p],lr=.1)
    p.grad=torch.ones_like(p);opt.step();saved=copy.deepcopy(opt.state_dict());w=p.detach().clone()
    def terms():return [p.square().sum(),(p-4).square().sum()]
    r=guarded_update([p],opt,terms(),terms,'projected_search',[1,.5])
    assert not r['accepted'] and r['rollback_verified']
    assert torch.equal(p,w) and equal_state(opt.state_dict(),saved)
