import numpy as np
import pytest
from tools.audit_partial_objective_conflict import gradient_summary
from tools.audit_partial_objective_conflict import symmetric_parameter_probe
import torch


def test_opposed_objectives_report_adverse_small_term():
    result=gradient_summary({'a':np.array([2.,0.]),'b':np.array([-1.,0.])},{'a':1.,'b':1.})
    assert result['pair_cosines']['a|b']==-1
    assert result['adverse_terms']==['b']
    assert result['unit_descent_directional_derivative']=={'a':-2.,'b':1.}


def test_zero_and_orthogonal_do_not_create_conflict():
    result=gradient_summary({'a':np.array([1.,0.]),'b':np.array([0.,1.]),'zero':np.zeros(2)},dict(a=1,b=1,zero=1))
    assert result['adverse_terms']==[]
    assert result['norms']['zero']==0


def test_nonfinite_rejected():
    with pytest.raises(ValueError):gradient_summary({'a':np.array([np.nan])},{'a':1.})


def test_symmetric_quadratic_derivative_and_restore():
    p=torch.nn.Parameter(torch.tensor([2.,3.],dtype=torch.float64));before=p.detach().clone()
    values=symmetric_parameter_probe([p],np.array([1.,0.]),.001,lambda:{'loss':float(p.square().sum())})
    assert (values[1]['loss']-values[-1]['loss'])/.002==pytest.approx(4.)
    assert torch.equal(p,before)


def test_restore_after_evaluator_error():
    p=torch.nn.Parameter(torch.tensor([2.]));before=p.detach().clone()
    def fail():raise RuntimeError('probe failure')
    with pytest.raises(RuntimeError):symmetric_parameter_probe([p],np.array([1.]),.01,fail)
    assert torch.equal(p,before)
