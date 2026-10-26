"""Research-only source-evidence guard for curvature dequantization."""
import numpy as np
from scipy.ndimage import uniform_filter,maximum_filter,minimum_filter
from training.aces_curvature_dequant import dequantize


def coherent_dequantize(source,iterations=200):
    result,allowed=dequantize(source,iterations)
    codes=np.asarray(source,dtype=np.float32)*255
    gx=np.zeros_like(codes);gy=np.zeros_like(codes)
    gx[:,:-1]=np.diff(codes,axis=1);gy[:-1]=np.diff(codes,axis=0)
    numerator=np.zeros(codes.shape[:2],np.float32);denominator=numerator.copy()
    for c in range(3):
        numerator+=np.abs(uniform_filter(gx[...,c],31))+np.abs(uniform_filter(gy[...,c],31))
        denominator+=uniform_filter(np.abs(gx[...,c])+np.abs(gy[...,c]),31)
    coherence=numerator/np.maximum(denominator,1e-6)
    span=(maximum_filter(codes,size=(31,31,1))-minimum_filter(codes,size=(31,31,1))).max(-1)
    jumps=np.maximum(np.abs(gx).max(-1),np.abs(gy).max(-1))
    allowed&=(coherence>=.8)&(span>=1.99)&(maximum_filter(jumps,31)<=2.01)
    result[~allowed]=np.asarray(source,dtype=np.float32)[~allowed]
    return result,allowed
