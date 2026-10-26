import numpy as np
import pytest
from training.aces_context_batches import stack_context_arrays


def test_crop_first_preserves_all_five_fields_and_frame_order():
    rng=np.random.default_rng(7)
    cache=[tuple(rng.normal(size=(c,37,53)).astype(np.float32) for c in (3,3,3,1,3)) for _ in range(3)]
    indices=[2,0];crop=(4,8,29,40)
    expected=[np.stack([cache[i][k] for i in indices])[:,:,4:33,8:48] for k in range(5)]
    actual=stack_context_arrays(cache,indices,crop)
    for a,b in zip(actual,expected):
        np.testing.assert_array_equal(a,b)
        assert a.flags.c_contiguous
    actual[0][:]=0
    assert np.any(cache[2][0]!=0)


def test_uncropped_motion_pair_and_invalid_crop():
    cache=[tuple(np.full((c,10,12),i+k,np.float32) for k,c in enumerate((3,3,3,1,3))) for i in range(2)]
    for k,a in enumerate(stack_context_arrays(cache,[0,1])):
        np.testing.assert_array_equal(a,np.stack([cache[0][k],cache[1][k]]))
    with pytest.raises(ValueError):stack_context_arrays(cache,[0],(0,0,11,12))
    with pytest.raises(ValueError):stack_context_arrays(cache,[])


@pytest.mark.parametrize('height,width',[(320,512),(321,513),(323,515),(1025,2049),(1024,2048)])
def test_replay_v2_retains_highlights_near_every_image_boundary(height,width):
    from training.aces_context_batches import sample_replay_crop
    rng=np.random.default_rng(11)
    for py in (4,65,height//2,height-5):
        for px in (4,65,width//2,width-5):
            for _ in range(8):
                (y,x,h,w),(top,bottom,left,right)=sample_replay_crop(height,width,np.array([[py,px]]),rng)
                assert y%4==x%4==0
                assert y+top<=py<y+h-bottom and x+left<=px<x+w-right
                assert y+h<=height and x+w<=width
