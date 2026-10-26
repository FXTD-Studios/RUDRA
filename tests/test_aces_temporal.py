import numpy as np
import pytest
from tools.evaluate_aces_temporal import temporal_change


def test_source_motion_is_cancelled_but_added_exposure_pumping_is_detected():
    previous=np.array([[1.,2.],[3.,4.]])
    current=np.array([[2.,3.],[1.,5.]])
    assert temporal_change(previous,current,previous,current)['mean_abs_excess_ev']==0
    result=temporal_change(previous,current*2**.1,previous,current)
    assert result['mean_abs_excess_ev']==pytest.approx(.1)


def test_nonpositive_cells_are_counted_not_hidden_by_log_floor():
    p=np.array([0.,-1.,1.]);c=np.ones(3)
    result=temporal_change(p,c,c,c)
    assert result['valid_cells']==1 and result['excluded_cells']==2
    assert temporal_change(p[:2],c[:2],c[:2],c[:2])['mean_abs_excess_ev'] is None
