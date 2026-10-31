"""Prepare identical context tensors without copying unused full-frame pixels."""
import numpy as np


def sample_replay_crop(height,width,points,rng):
    """V2: keep the chosen interior highlight inside the supervised crop.

    Origin remains divisible by four. Snap the far edge to the real image edge
    when only1..3 pixels remain, rather than creating a false64px exclusion.
    Points must already exclude the outer four pixels of the original image.
    """
    if height<320 or width<512 or not len(points):raise ValueError('eligible replay frame required')
    py,px=points[int(rng.integers(len(points)))]
    if not (4<=py<height-4 and 4<=px<width-4):raise ValueError('interior highlight point required')
    y=int(np.clip(py-rng.integers(64,252),0,height-320))//4*4
    x=int(np.clip(px-rng.integers(64,444),0,width-512))//4*4
    end_y=y+320;end_x=x+512
    if height-end_y<4:end_y=height
    if width-end_x<4:end_x=width
    borders=[64 if y else 4,64 if end_y<height else 4,64 if x else 4,64 if end_x<width else 4]
    top,bottom,left,right=borders
    if not (y+top<=py<end_y-bottom and x+left<=px<end_x-right):raise AssertionError('selected highlight excluded')
    return (y,x,end_y-y,end_x-x),borders


def stack_context_arrays(cache,indices,crop=None):
    if not indices:raise ValueError('at least one cached frame required')
    if crop is not None:
        y,x,h,w=crop
        if min(y,x)<0 or min(h,w)<=0:raise ValueError('positive in-frame crop required')
        for i in indices:
            if any(a.ndim!=3 or y+h>a.shape[-2] or x+w>a.shape[-1] for a in cache[i]):raise ValueError('crop outside cached frame')
    return [np.stack([cache[i][k] if crop is None else cache[i][k][:,y:y+h,x:x+w] for i in indices]) for k in range(5)]
