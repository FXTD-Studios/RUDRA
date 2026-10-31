import numpy as np
from tools.pilot_aces_protected_step import project_direction,acceptable


def test_projection_preserves_feasible_and_removes_conflict():
    g=np.array([[1.,0.],[0.,1.],[1.,1.]])
    assert np.allclose(project_direction([2.,3.],g),[2.,3.])
    assert np.allclose(project_direction([-2.,3.],g),[0.,3.])


def test_opposed_and_degenerate_gradients_stall_safely():
    g=np.array([[1.,0.],[-1.,0.],[0.,0.]])
    assert np.allclose(project_direction([1.,2.],g),[0.,2.])
    assert np.allclose(project_direction([1.,0.],g),[0.,0.])


def test_acceptance_requires_temporal_gain_and_protects_other_losses():
    b=np.array([1.,2.,3.])
    assert acceptable(b,np.array([1.,1.9,2.9]),1e-6,1e-9)
    assert not acceptable(b,b,1e-6,1e-9)
    assert not acceptable(b,np.array([1.1,1.9,2.9]),1e-6,1e-9)
