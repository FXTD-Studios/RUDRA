import numpy as np
import pytest
from tools.evaluate_run05_temporal_probe import excess


def test_baseline_motion_cancels_but_uniform_and_local_pumping_remain():
    a = np.zeros((2,2)); b = np.array([[1.,-1.],[.5,-.5]])
    assert excess(a,b,a,b) == {'global_excess_ev':0.,'cell_excess_ev':0.}
    assert excess(a,b+.25,a,b)['global_excess_ev'] == pytest.approx(.25)
    local=excess(a,b+np.array([[.2,-.2],[.2,-.2]]),a,b)
    assert local['global_excess_ev'] == pytest.approx(0)
    assert local['cell_excess_ev'] == pytest.approx(.2)
