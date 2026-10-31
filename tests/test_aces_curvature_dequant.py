import numpy as np
import pytest
from training.aces_curvature_dequant import dequantize


def test_constant_edges_and_clipped_support_stay_exact():
    x=np.full((32,64,3),.4,np.float32);x[:,32:]=.8;x[:4]=1;x[-4:]=0
    y,mask=dequantize(x)
    np.testing.assert_array_equal(x,y)
    assert not mask[:4].any() and not mask[-4:].any()


def test_quantized_ramp_curvature_and_reference_error_improve_within_bins():
    truth=np.broadcast_to(np.linspace(.2,.3,256,dtype=np.float32)[None,:,None],(16,256,3)).copy()
    source=np.round(truth*255)/255
    result,_=dequantize(source)
    assert np.max(np.abs(result-source))*255<=.49002
    assert np.mean((result-truth)**2)<np.mean((source-truth)**2)*.3
    assert np.mean(np.diff(result,2,axis=1)**2)<np.mean(np.diff(source,2,axis=1)**2)*.1


def test_invalid_source_rejected():
    with pytest.raises(ValueError):dequantize(np.full((2,2,3),np.nan))


def test_coherent_guard_preserves_fine_texture_and_still_smooths_a_ramp():
    from training.aces_coherent_dequant import coherent_dequantize
    y,x=np.mgrid[:64,:256].astype(np.float32)
    v=140+5*x/256+.8*np.sin(x*1.3)+.7*np.cos(y*1.1)
    texture=np.round(np.repeat(v[...,None],3,axis=-1))/255
    result,allowed=coherent_dequantize(texture)
    np.testing.assert_array_equal(result[8:-8,8:-8],texture[8:-8,8:-8])
    truth=np.repeat((230+23*x/256+y/64)[...,None],3,axis=-1)/255
    ramp=np.round(truth*255)/255;result,allowed=coherent_dequantize(ramp)
    assert allowed[8:-8,8:-8].mean()>.5
    assert np.mean((result-truth)**2)<np.mean((ramp-truth)**2)*.5
