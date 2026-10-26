import numpy as np
from tools.audit_aces_learned_detail import common_detail


def test_all_methods_use_same_edges_when_one_prediction_is_nonpositive():
    ref=np.repeat(np.arange(1,17,dtype=np.float32).reshape(4,4,1),3,axis=2)
    bad=ref.copy();bad[0,0]=-1
    result=common_detail({'good':ref,'bad':bad},ref,np.ones((4,4),bool))
    assert result['excluded_pixels']==1
    assert result['methods']['good']['edges']==result['methods']['bad']['edges']
    assert result['methods']['good']['rmse_ev']==0
