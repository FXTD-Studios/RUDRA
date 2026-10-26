import numpy as np
from tools.audit_aces_highlight_strata import strata,metrics


def test_partition_is_disjoint_and_excludes_image_border():
    s=np.zeros((40,50,3),np.float32);s[5:35,5:45]=1;s[10,10,0]=.9
    high=s.max(-1)>.5;m=strata(s,high);total=sum(x.astype(int) for x in m.values())
    expected=high.copy();expected[:4]=False;expected[-4:]=False;expected[:,:4]=False;expected[:,-4:]=False
    np.testing.assert_array_equal(total,expected.astype(int))
    assert all(v.any() for v in m.values())
    assert m['partial'][10,10] and m['full_boundary'][10,11] and m['full_deep'][20,25]


def test_metrics_distinguish_constant_bias_from_detail_error():
    t=np.zeros((20,24,3),np.float32);mask=np.ones((20,24),bool)
    p=np.sinh(np.full_like(t,.5));r=metrics(p,t,mask)
    assert abs(r['asinh_mse']-.25)<1e-6 and abs(r['signed_asinh_bias']-.5)<1e-6
    assert r['gradient_mse']==0
    p[:,12:]+=1;assert metrics(p,t,mask)['gradient_mse']>0
    r=metrics(p,t,mask&False);assert r['asinh_mse'] is None and r['gradient_mse'] is None
