import numpy as np
import pytest
import torch
from training.aces_pcgrad import project_conflicting


def test_nonconflicting_sum_and_zero():
    g=torch.tensor([[1.,0.],[0.,2.],[0.,0.]])
    result,meta=project_conflicting(g,np.random.default_rng(5))
    assert torch.equal(result,g.sum(0)) and meta['projections']==0


def test_two_task_projection_uses_original_other():
    g=torch.tensor([[1.,0.],[-1.,1.]],dtype=torch.float64);before=g.clone()
    result,meta=project_conflicting(g,np.random.default_rng(5))
    # Projected first=(.5,.5); second=(0,1), both against original other.
    assert torch.allclose(result,torch.tensor([.5,1.5],dtype=g.dtype))
    assert meta['projections']==2 and torch.equal(g,before)


def test_exact_opposition_can_stall():
    result,_=project_conflicting(torch.tensor([[2.,0.],[-1.,0.]]),np.random.default_rng(5))
    assert torch.equal(result,torch.zeros(2))


def test_reproducible_order_and_scalar_reference():
    g=np.random.default_rng(7).normal(size=(5,8));expected=g.copy();rng=np.random.default_rng(9)
    for i in range(5):
        for j in rng.permutation([j for j in range(5) if i!=j]):
            dot=expected[i]@g[j]
            if dot<0:expected[i]-=dot/(g[j]@g[j])*g[j]
    a,ma=project_conflicting(torch.from_numpy(g),np.random.default_rng(9))
    b,mb=project_conflicting(torch.from_numpy(g),np.random.default_rng(9))
    assert np.allclose(a.numpy(),expected.sum(0)) and torch.equal(a,b) and ma==mb


def test_nonfinite_rejected():
    with pytest.raises(ValueError):project_conflicting(torch.tensor([[float('nan')]]),np.random.default_rng(0))
