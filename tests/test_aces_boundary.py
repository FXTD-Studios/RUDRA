import numpy as np
import pytest
from pipeline.aces_boundary import match_boundary


def test_preserves_reference_detail_with_boundary_offset_and_outside_exact():
    y,x=np.mgrid[:20,:24]
    baseline=.03*x+.02*y
    base=np.repeat(np.sinh(baseline)[...,None],3,axis=2).astype(np.float32)
    mask=np.zeros((20,24),bool);mask[4:16,4:20]=True
    bump=np.zeros(mask.shape);bump[8:12,9:15]=.4
    reference=np.sinh(np.arcsinh(base.astype(float))+.7+bump[...,None]).astype(np.float32)
    output=match_boundary(base,reference,mask)
    np.testing.assert_array_equal(output[~mask],base[~mask])
    np.testing.assert_allclose(np.arcsinh(output),np.arcsinh(base)+bump[...,None],atol=1e-6)


def test_empty_mask_and_unanchored_mask():
    base=np.ones((4,5,3),np.float32)
    np.testing.assert_array_equal(match_boundary(base,base*2,np.zeros((4,5),bool)),base)
    with pytest.raises(ValueError,match='anchor'):
        match_boundary(base,base,np.ones((4,5),bool))
