"""Masked signed-asinh spatial detail objective; no clipping of model radiance."""
import torch


def highlight_gradient_loss(predicted,target,mask):
    """Equal-image mean squared gradient error, only edges with both ends valid.

    Inputs NCHW, mask N1HW. Empty images contribute zero. Artificial crop borders
    must be excluded by the caller. Signed asinh avoids a luminance floor.
    """
    residual=torch.asinh(predicted)-torch.asinh(target)
    total=predicted.new_zeros(predicted.shape[0]);count=total.clone()
    for dim in (2,3):
        left=[slice(None)]*4;right=left.copy();left[dim]=slice(None,-1);right[dim]=slice(1,None)
        valid=mask[tuple(left)]&mask[tuple(right)]
        error=torch.diff(residual,dim=dim).square().mean(1,keepdim=True)
        total=total+(error*valid).flatten(1).sum(1)
        count=count+valid.flatten(1).sum(1)
    return (total/count.clamp_min(1)).mean()
