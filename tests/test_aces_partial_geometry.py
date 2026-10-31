import pytest
from training.aces_partial_geometry import aligned_partial_origin


def test_origins_stay_in_bounds_and_on_stride_grid():
    for cy in (0,79,160,319):
        for cx in (0,87,256,511):
            y,x=aligned_partial_origin(cy,cx,320,512,192,64)
            assert y%4==x%4==0
            assert 0<=y<=128 and 0<=x<=320


@pytest.mark.parametrize('h,w,crop,border',[(321,512,192,64),(320,513,192,64),(320,512,191,64),(320,512,192,48),(320,512,128,64)])
def test_unsupported_geometry_fails_explicitly(h,w,crop,border):
    with pytest.raises(ValueError):aligned_partial_origin(160,256,h,w,crop,border)
