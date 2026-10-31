import numpy as np
import pytest
from rudra.delivery.video import _encode_frame,TARGETS,EncodeError
from rudra.delivery.aces_video import ACES2HDR1000View


def test_signed_scene_values_reach_official_hdr_transform_unclipped():
    ocio=pytest.importorskip('PyOpenColorIO')
    view=ACES2HDR1000View()
    rgb=np.array([[[-.01,.2,.3],[0.,0.,0.],[2.,10.,1.]]],np.float32)
    expected=rgb.reshape(-1,3).copy()
    config=ocio.Config.CreateFromBuiltinConfig(view.CONFIG)
    config.getProcessor(ocio.DisplayViewTransform(src='ACEScg',display=view.DISPLAY,view=view.VIEW)).getDefaultCPUProcessor().applyRGB(expected)
    np.testing.assert_array_equal(view.forward(rgb),expected.reshape(rgb.shape))
    # The official display transform can map negative AP1 and clipped AP1 to
    # identical display values. Verify the wrapper passes signed data directly.
    class Capture:
        def applyRGB(self,array):
            np.testing.assert_array_equal(array,rgb.reshape(-1,3))
    view._forward=Capture()
    np.testing.assert_array_equal(view.forward(rgb),rgb)


def test_already_mastered_codes_preserve_all_16bit_levels():
    values=np.arange(65536,dtype=np.float64)/65535
    rgb=np.repeat(values[:,None],3,axis=1).reshape(256,256,3)
    coded=_encode_frame(rgb,TARGETS['ffv1rgb16'],1000,'rec2020',False,'pq_codes')
    np.testing.assert_array_equal(coded[:,:,0].ravel(),np.arange(65536,dtype=np.uint16))
    with pytest.raises(EncodeError):_encode_frame(rgb,TARGETS['ffv1rgb16'],1000,'rec2020',True,'pq_codes')
    with pytest.raises(EncodeError):_encode_frame(rgb+1,TARGETS['ffv1rgb16'],1000,'rec2020',False,'pq_codes')
