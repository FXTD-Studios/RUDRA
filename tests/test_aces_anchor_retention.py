import copy
from tools.pilot_aces_anchor_retention import retained


def row(value=1.):
    return {'photo':value,'detail':value,'temporal':None,'log_detail':value,'log_edges':[4]}


def test_panel_cannot_hide_member_regression_with_another_improvement():
    initial={'a':row(),'b':row()}
    after={'a':row(.1),'b':row(1.01)}
    assert not retained(initial,after,initial,1e-6,1e-9)
    after['b']=row(.99)
    assert retained(initial,after,initial,1e-6,1e-9)


def test_initial_limit_prevents_accumulated_tolerance_drift():
    initial={'a':row()};before={'a':row(1.0000008)};after={'a':row(1.0000015)}
    assert not retained(before,after,initial,1e-6,1e-9)


def test_panel_rejects_lost_edges_missing_member_and_nonfinite():
    initial={'a':row()};after=copy.deepcopy(initial);after['a']['log_edges']=[3]
    assert not retained(initial,after,initial,1e-6,1e-9)
    assert not retained(initial,{},initial,1e-6,1e-9)
    after=copy.deepcopy(initial);after['a']['detail']=float('nan')
    assert not retained(initial,after,initial,1e-6,1e-9)
