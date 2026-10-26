import numpy as np
from rudra.aces_view import ACES2View
from training.aces_reliable_highlights import partial_only,feather_full


def test_full_fallback_and_feather_preserve_ordinary_shadow_and_core():
    view=ACES2View();source=np.full((48,48,3),.3,np.float32);source[8:40,8:40]=1
    source[:4]=0;baseline=view.inverse(source);prediction=baseline+1
    weight=np.zeros((48,48),np.float32);weight[8:40,8:40]=1;weight[:4]=1
    fallback,_=partial_only(source,prediction,baseline,weight,view)
    feather,stats=feather_full(source,prediction,baseline,weight,view)
    np.testing.assert_array_equal(fallback[8:40,8:40],baseline[8:40,8:40])
    np.testing.assert_array_equal(feather[8,8:40],baseline[8,8:40])
    np.testing.assert_array_equal(feather[24,24],prediction[24,24])
    np.testing.assert_array_equal(feather[:8],prediction[:8])
    assert stats['outside_support_bit_exact'] and stats['feathered_pixels']>0
