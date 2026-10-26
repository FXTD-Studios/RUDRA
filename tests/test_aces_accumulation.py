import copy
import torch
from training.aces_accumulation import accumulate_step


def test_two_windows_match_large_batch_adam_weights_and_moments():
    torch.manual_seed(41)
    a=torch.nn.Linear(3,2).double();b=copy.deepcopy(a)
    oa=torch.optim.Adam(a.parameters(),lr=.01);ob=torch.optim.Adam(b.parameters(),lr=.01)
    x=torch.randn(12,3,dtype=torch.float64);y=torch.randn(12,2,dtype=torch.float64)
    first=copy.deepcopy(a.state_dict())
    for i in range(12):
        applied=accumulate_step((a(x[i:i+1])-y[i:i+1]).square().mean(),oa,list(a.parameters()),i+1,6)
        assert applied==((i+1)%6==0)
        if i<5:assert all(torch.equal(v,a.state_dict()[k]) for k,v in first.items())
        if applied:
            ob.zero_grad();(b(x[i-5:i+1])-y[i-5:i+1]).square().mean().backward();ob.step()
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa,pb,rtol=1e-12,atol=1e-12)
        for key in ('exp_avg','exp_avg_sq','step'):torch.testing.assert_close(oa.state[pa][key],ob.state[pb][key],rtol=1e-12,atol=1e-12)


def test_window_one_matches_ordinary_step():
    a=torch.nn.Parameter(torch.tensor([2.]));o=torch.optim.Adam([a],lr=.1)
    assert accumulate_step(a.square().sum(),o,[a],1,1)
    assert int(o.state[a]['step'])==1
