import numpy as np
import torch
from training.aces_panel_direction import project_panel,log_detail_loss
from tools.pilot_aces_protected_step import project_direction
from tools.audit_aces_reference_detail import gradient_scores


def test_dual_matches_enumeration_and_handles_opposing_constraints():
    rng=np.random.default_rng(84)
    for _ in range(12):
        g=rng.normal(size=(3,7));d=rng.normal(size=7)
        actual,_=project_panel(d,g)
        np.testing.assert_allclose(actual,project_direction(d,g),atol=1e-8)
    actual,_=project_panel(np.array([2.,-3.]),np.array([[1.,0.],[-1.,0.],[0.,1.]]))
    np.testing.assert_allclose(actual,[0,0],atol=1e-10)


def test_many_constraints_have_feasible_projection():
    rng=np.random.default_rng(3);g=rng.normal(size=(24,60));d=rng.normal(size=60)
    projected,diagnostic=project_panel(d,g)
    assert np.min(g@projected)>-1e-8
    assert diagnostic['norm_ratio']<=1+1e-10
    np.testing.assert_allclose(np.dot(projected,d-projected),0,atol=1e-8)


def test_log_detail_matches_reference_and_gradient_is_finite():
    rng=np.random.default_rng(9);p=rng.uniform(.1,3,(5,7,3)).astype('float32');t=rng.uniform(.1,3,(5,7,3)).astype('float32')
    mask=np.ones((5,7),bool);p[0,0]=-1
    x=torch.tensor(p.transpose(2,0,1)[None],requires_grad=True);y=torch.tensor(t.transpose(2,0,1)[None])
    loss=log_detail_loss(x,y,torch.tensor(mask[None,None]))
    np.testing.assert_allclose(float(loss.detach()),gradient_scores(p,t,mask)['rmse_ev'],rtol=2e-6)
    loss.backward();assert torch.isfinite(x.grad).all();assert (x.grad[:,:,0,0]==0).all()
    x=torch.ones(1,3,4,4,requires_grad=True);loss=log_detail_loss(x,x.detach(),torch.ones(1,1,4,4,dtype=torch.bool));loss.backward()
    assert float(loss.detach())==0 and torch.isfinite(x.grad).all()
