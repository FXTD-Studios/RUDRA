import numpy as np
import pytest
from rudra.interval_dequant import bounded_interval_dequant


def test_ramp_improves_without_leaving_quantization_intervals():
    truth = np.broadcast_to(np.linspace(.2,.4,512,dtype=np.float32)[None,:,None], (24,512,3)).copy()
    quantized = np.rint(truth*255)/255
    result = bounded_interval_dequant(quantized)
    assert np.max(np.abs(result-quantized))*255 <= .4901
    assert np.mean((np.diff(result,axis=1)-np.diff(truth,axis=1))**2) < .5*np.mean((np.diff(quantized,axis=1)-np.diff(truth,axis=1))**2)
    assert np.mean((result-truth)**2) < np.mean((quantized-truth)**2)
    np.testing.assert_array_equal(result,bounded_interval_dequant(quantized))


def test_sharp_edges_flat_fields_and_lost_range_are_unchanged():
    source = np.zeros((16,32,3),np.float32)
    source[:,:8] = 1
    source[:,8:16] = 60/255
    source[:,16:24] = 120/255
    result = bounded_interval_dequant(source)
    np.testing.assert_array_equal(result,source)
    with pytest.raises(ValueError):
        bounded_interval_dequant(source,step=.5)
