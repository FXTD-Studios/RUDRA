import numpy as np
from tools.audit_aces_reference_detail import gradient_scores


def test_metric_detects_lost_detail_and_is_exposure_invariant():
    y,x=np.mgrid[:20,:20]
    reference=np.repeat(np.exp(.1*x+.05*y)[...,None],3,axis=2)
    mask=np.ones((20,20),bool)
    exact=gradient_scores(reference*17,reference,mask)
    flat=gradient_scores(np.ones_like(reference),reference,mask)
    assert exact['rmse_ev']<1e-12
    assert abs(exact['gradient_gain']-1)<1e-12
    assert flat['gradient_gain']==0
    assert flat['rmse_ev']>.1
