import numpy as np
import pytest
from rudra.aces_view import ACES2View
from training.aces_colour_backtrack import constrain_colour


def test_colour_backtracking_preserves_excluded_regions_and_meets_display_bound():
    view=ACES2View()
    source=np.array([[[1.,0.,0.],[0.,0.,1.],[1.,1.,1.],[.3,.4,.5],[0.,0.,0.]]],np.float32)
    baseline=view.inverse(source)
    prediction=baseline+np.array([-.08,-.06,.03],np.float32)
    weight=np.array([[1,1,1,0,1]],np.float32)
    result,stats=constrain_colour(source,prediction,baseline,weight,view)
    assert stats['changed_pixels']==2 and stats['remaining_display_errors']==0
    assert stats['outside_support_bit_exact']
    np.testing.assert_array_equal(result[:,2:],prediction[:,2:])
    assert np.abs(view.forward(result[:,:2])-source[:,:2]).max()*255<=1


def test_already_consistent_partial_highlights_stay_bit_exact():
    view=ACES2View();source=np.array([[[1.,.4,.2]]],np.float32)
    prediction=view.inverse(source)
    result,stats=constrain_colour(source,prediction,prediction,np.ones((1,1),np.float32),view)
    np.testing.assert_array_equal(result,prediction)
    assert stats['changed_pixels']==0


def test_full_highlight_policy_fixes_bad_white_without_changing_valid_white():
    view=ACES2View();source=np.ones((1,2,3),np.float32);baseline=view.inverse(source)
    prediction=baseline.copy();prediction[0,0]=[0,0,0]
    result,stats=constrain_colour(source,prediction,baseline,np.ones((1,2),np.float32),view,include_fully_clipped=True)
    assert stats['remaining_display_errors']==0 and stats['changed_pixels']==1
    assert np.abs(view.forward(result)-source).max()*255<=1
    np.testing.assert_array_equal(result[0,1],prediction[0,1])


def test_nonfinite_prediction_cannot_be_reported_as_zero_appearance_errors():
    view=ACES2View();source=np.ones((1,1,3),np.float32);base=view.inverse(source)
    prediction=base.copy();prediction[0,0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        constrain_colour(source,prediction,base,np.ones((1,1),np.float32),view)
