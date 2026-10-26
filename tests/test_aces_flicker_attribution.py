import numpy as np
from tools.audit_aces_flicker_attribution import attribute


def test_mask_only_and_field_only_changes_are_distinguished():
    p=np.ones((2,3));zero=np.zeros_like(p)
    r=attribute(p,p,p,p,{(0,0):zero,(0,1):zero,(1,0):p,(1,1):p})
    assert abs(r['terms']['mask']['mean_abs_ev']-1)<1e-12
    assert r['terms']['field']['mean_abs_ev']==0 and r['closure_max_abs_ev']<1e-12
    r=attribute(p,p,p,p,{(0,0):zero,(1,0):zero,(0,1):p,(1,1):p})
    assert r['terms']['mask']['mean_abs_ev']==0 and abs(r['terms']['field']['mean_abs_ev']-1)<1e-12


def test_shapley_closes_with_three_changes_and_excludes_nonpositive_corners():
    p=np.ones((2,3));r=attribute(p,p*1.4,p,p*1.1,{(0,0):p*.1,(0,1):p*.3,(1,0):p*.2,(1,1):p*.6})
    assert r['closure_max_abs_ev']<1e-12 and r['valid_cells']==6
    r=attribute(p,p,p,p,{(0,0):-p,(0,1):p,(1,0):p,(1,1):p})
    assert r['valid_cells']==0 and r['terms'] is None and r['total_mean_abs_ev'] is None
