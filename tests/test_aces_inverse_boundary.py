import numpy as np
from training.aces_colour_backtrack import constrain_colour
from training.aces_reference import reference_view


def test_fractional_saturated_rgb_repair_is_opt_in_and_only_changes_failures():
    view=reference_view();source=np.array([[[.9919890165,1.,1.],[.9918516874,1.,1.],[.4,.5,.6],[1.,1.,1.]]],np.float32)
    base=view.inverse(source);weight=np.array([[1.,1.,0.,1.]],np.float32)
    old,d=constrain_colour(source,base,base,weight,view,include_fully_clipped=True)
    assert d['remaining_display_errors']==2
    result,d=constrain_colour(source,base,base,weight,view,include_fully_clipped=True,repair_inverse_boundary=True)
    assert d['remaining_display_errors']==0 and d['inverse_boundary_repaired_pixels']==2
    np.testing.assert_array_equal(result[:,2:],old[:,2:])
    assert np.abs(view.forward(result)-source).max()*255<=1


def test_unsupported_inverse_fix_does_not_hide_remaining_error():
    class BadView:
        def forward(self,x):return np.zeros_like(x)
        def inverse(self,x):return np.ones_like(x)
    source=np.ones((1,1,3),np.float32);base=source.copy()
    _,d=constrain_colour(source,base,base,np.ones((1,1),np.float32),BadView(),include_fully_clipped=True,repair_inverse_boundary=True)
    assert d['remaining_display_errors']==1 and d['inverse_boundary_repaired_pixels']==0
