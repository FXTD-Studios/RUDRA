"""Research ablation: retain learned highlights only with an unclipped channel."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from training.aces_colour_backtrack import constrain_colour


def partial_only(source,prediction,baseline,weight,view):
    # The completely clipped source has no measured RGB channel to anchor detail.
    # This ablation preserves analytic radiance there; it does not recover texture.
    result,diagnostic=constrain_colour(source,prediction,baseline,weight,view)
    full=(weight>0)&(source.min(-1)>=254.5/255)
    result[full]=baseline[full]
    high=(weight>0)&(source.max(-1)>.5)
    diagnostic['fully_clipped_analytic_pixels']=int(full.sum())
    diagnostic['outside_support_bit_exact']=bool(np.array_equal(result[~high],prediction[~high]))
    diagnostic['outside_support_definition']='outside all highlight support; fully clipped highlights explicitly use analytic fallback'
    return result,diagnostic


def feather_full(source,prediction,baseline,weight,view,radius=8):
    """Ablation: smoothly admit learned radiance inside fully-clipped regions."""
    if not np.isfinite(radius) or radius<=0:raise ValueError('positive finite radius required')
    result,diagnostic=constrain_colour(source,prediction,baseline,weight,view)
    full=(weight>0)&(source.min(-1)>=254.5/255)
    if full.any() and not full.all():
        distance=distance_transform_edt(np.pad(full,1,constant_values=False))[1:-1,1:-1]
        t=np.clip((distance-1)/radius,0,1).astype(np.float32)
        alpha=t*t*(3-2*t)
        result[full]=baseline[full]+alpha[full,None]*(result[full]-baseline[full])
        diagnostic['feathered_pixels']=int((full&(alpha<1)).sum())
    else:diagnostic['feathered_pixels']=0
    high=(weight>0)&(source.max(-1)>.5)
    diagnostic['outside_support_bit_exact']=bool(np.array_equal(result[~high],prediction[~high]))
    diagnostic['outside_support_definition']='outside all highlight support; fully-clipped boundary explicitly feathered'
    diagnostic['fully_clipped_feather_radius']=radius
    return result,diagnostic
