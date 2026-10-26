import numpy as np
import pytest
pytest.importorskip('PyOpenColorIO')
from rudra.aces_view import ACES2View


def test_reference_view_preserves_greys_and_does_not_mutate_inputs():
    view = ACES2View()
    source = np.repeat((np.arange(256,dtype=np.float32)/255)[:,None],3,axis=1)
    saved = source.copy()
    scene = view.inverse(source)
    np.testing.assert_array_equal(source,saved)
    restored = view.forward(scene)
    assert np.max(np.abs(restored-source))*255 < .01
    assert scene[-1].max() > 1  # no SDR-like ceiling imposed on ACES values


@pytest.mark.parametrize('value',[np.nan,np.inf,-.1,1.1])
def test_invalid_inverse_codes_are_not_silently_clipped(value):
    with pytest.raises(ValueError):
        ACES2View().inverse(np.full((2,3),value,np.float32))
