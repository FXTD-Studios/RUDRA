import numpy as np
import pytest
pytest.importorskip('PyOpenColorIO')
from rudra.aces_view import ACES2View
from pipeline.aces_shadow_regions import bound_shadow_regions


def test_region_scale_bounds_brightness_and_preserves_relative_detail():
    view=ACES2View();source=np.full((12,20,3),.2,np.float32)
    mask=np.zeros((12,20),bool);mask[2:10,2:8]=True;mask[2:10,12:18]=True
    source[mask]=1/255
    target=view.inverse(np.full_like(source,.025))
    target[2:10,2:8]*=np.linspace(.4,1.5,6,dtype=np.float32)[None,:,None]
    out,stats=bound_shadow_regions(target,source,mask,view)
    np.testing.assert_array_equal(out[~mask],target[~mask])
    assert stats['regions']==2 and stats['scaled_regions']==2
    assert ((view.forward(out)[mask]-source[mask])*255).max()<.491
    for ys,xs in ((slice(2,10),slice(2,8)),(slice(2,10),slice(12,18))):
        ratios=out[ys,xs]/target[ys,xs]
        assert ratios.min()>0
        np.testing.assert_allclose(ratios,ratios.flat[0],rtol=2e-7)
