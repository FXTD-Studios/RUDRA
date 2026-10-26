"""Experimental common exposure reduction for connected shadow components."""
import numpy as np
from scipy.ndimage import label


def bound_shadow_regions(target,source,mask,view):
    if target.shape!=source.shape or target.shape!=(*mask.shape,3):
        raise ValueError('matching RGB arrays and spatial mask required')
    if not np.isfinite(target).all() or not np.isfinite(source).all():
        raise ValueError('finite inputs required')
    labels,count=label(mask)
    if count==0:return target.copy(),{'regions':0,'min_scale':1.,'scaled_regions':0}
    ids=labels[mask]-1;values=target[mask];limit=source[mask]+.49/255
    def feasible(scale):
        error=(view.forward((values*scale[ids,None]).astype(np.float32))-limit).max(-1)
        worst=np.full(count,-np.inf);np.maximum.at(worst,ids,error)
        return worst<=0
    lo=np.zeros(count);hi=np.ones(count)
    if not feasible(lo).all():raise ValueError('zero exposure is not feasible')
    unchanged=feasible(hi);lo[unchanged]=1
    # Each accepted lower endpoint was actually evaluated. We do not assume
    # monotonicity to claim an optimum: this finds a verified feasible scale.
    for _ in range(22):
        mid=(lo+hi)*.5;ok=feasible(mid)
        lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid)
    output=target.copy();output[mask]=values*lo[ids,None]
    if ((view.forward(output[mask])-source[mask])*255).max()>.4901:
        raise ValueError('shadow bound verification failed')
    return output,{'regions':int(count),'min_scale':float(lo.min()),
                   'scaled_regions':int((lo<1).sum()),'zero_regions':int((lo==0).sum())}
