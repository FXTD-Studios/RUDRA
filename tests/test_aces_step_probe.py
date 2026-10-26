import pytest
import torch
from training.aces_step_probe import measure_step


def test_quadratic_actual_counterfactual_and_restore():
    p=torch.nn.Parameter(torch.tensor([1.5],dtype=torch.float64))
    r=measure_step([p],torch.tensor([2.],dtype=p.dtype),torch.tensor([1.],dtype=p.dtype),lambda:{'loss':float(p.square().sum())})
    assert r['baseline']['loss']==4 and r['values']['actual_adam']['loss']==2.25
    assert r['values']['projected_matched']['loss']==2.25
    assert r['values']['projected_small']['loss']==pytest.approx(1.9999**2)
    assert torch.equal(p,torch.tensor([1.5],dtype=p.dtype))


def test_exception_restores_parameters_and_rng():
    p=torch.nn.Parameter(torch.tensor([1.5]));initial=p.detach().clone();rng=torch.get_rng_state().clone();calls=[]
    def evaluate():
        torch.rand(1);calls.append(1)
        if len(calls)==2:raise RuntimeError('failure')
        return {'loss':float(p.square().sum())}
    with pytest.raises(RuntimeError):measure_step([p],torch.tensor([2.]),torch.tensor([1.]),evaluate)
    assert torch.equal(p,initial) and torch.equal(torch.get_rng_state(),rng)
