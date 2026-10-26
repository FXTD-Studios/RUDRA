import numpy as np
import pytest
pytest.importorskip('PyOpenColorIO')
from rudra.aces_view import ACES2View
from pipeline.aces_shadow_smooth import bound_smooth_shadows


def test_smooth_shadow_solver_preserves_outside_and_verified_brightness():
    view=ACES2View();source=np.full((24,24,3),.02,np.float32)
    mask=np.zeros((24,24),bool);mask[4:20,4:20]=True
    source[mask]=1/255
    source[10:14,10:14]=0
    base=view.inverse(source);reference=view.inverse(np.full_like(source,.025))
    output,stats=bound_smooth_shadows(reference,base,source,mask,view)
    np.testing.assert_array_equal(output[~mask],reference[~mask])
    assert ((view.forward(output)[mask]-source[mask])*255).max()<.491
    assert stats['iterations']>0 and np.isfinite(output).all()
    assert (output[mask]>0).all()


def test_exact_black_reference_is_preserved_without_radiance_floor():
    view=ACES2View();source=np.full((12,12,3),1/255,np.float32)
    base=view.inverse(source);reference=base.copy();reference[5,5]=0
    mask=np.zeros((12,12),bool);mask[3:9,3:9]=True
    output,stats=bound_smooth_shadows(reference,base,source,mask,view)
    np.testing.assert_array_equal(output[5,5],np.zeros(3))
    assert stats['exact_black_pixels']==1 and np.isfinite(output).all()
