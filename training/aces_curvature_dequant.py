"""Research-only constrained source-code curvature minimization, not HDR recovery.

Original implementation: projected accelerated gradient on squared Laplacian,
with fixed source bins and a source-defined edge/clip guard. No learned weights.
"""
import numpy as np
from scipy.ndimage import laplace,maximum_filter


def dequantize(source,iterations=200):
    source=np.asarray(source,dtype=np.float32)
    if source.ndim!=3 or source.shape[-1]!=3 or not np.isfinite(source).all() or source.min()<0 or source.max()>1:
        raise ValueError('finite HWC RGB source in [0,1] required')
    if not isinstance(iterations,int) or iterations<0:raise ValueError('nonnegative integer iterations required')
    codes=source*255
    # Keep observed edges and clipped/black support fixed. The guard never uses
    # reference data or model scores. Channels are independent within half-code bins.
    jump=np.zeros(source.shape[:2],np.float32)
    dx=np.abs(np.diff(codes,axis=1)).max(-1);dy=np.abs(np.diff(codes,axis=0)).max(-1)
    jump[:,:-1]=np.maximum(jump[:,:-1],dx);jump[:,1:]=np.maximum(jump[:,1:],dx)
    jump[:-1]=np.maximum(jump[:-1],dy);jump[1:]=np.maximum(jump[1:],dy)
    allowed=(maximum_filter(jump,size=5,mode='reflect')<=2.01)&(codes.max(-1)<254.5)&(codes.max(-1)>3)
    lower=np.where(allowed[...,None],np.maximum(codes-.49,0),codes)
    upper=np.where(allowed[...,None],np.minimum(codes+.49,255),codes)
    u=codes.copy();y=u.copy();t=1.
    # ||spatial Laplacian|| <=8, hence a step1/64 for L.T L is nonexpansive.
    for _ in range(iterations):
        gradient=np.stack([laplace(laplace(y[...,c],mode='reflect'),mode='reflect') for c in range(3)],axis=-1)
        updated=np.clip(y-gradient/64,lower,upper)
        next_t=(1+np.sqrt(1+4*t*t))/2
        y=updated+((t-1)/next_t)*(updated-u);u=updated;t=next_t
    result=(u/255).astype(np.float32)
    result[~allowed]=source[~allowed]
    return result,allowed
