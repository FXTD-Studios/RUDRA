"""Experimental ACES reconstruction objectives with explicit shadow scale."""
from functools import lru_cache
import numpy as np
import torch
from training.aces_reference import reference_view


def absolute_reconstruction_loss(output,target):
    error=(torch.asinh(output.acescg)-torch.asinh(target)).square().mean(1,keepdim=True)
    weight=output.reconstruction_weight
    return (error*weight).sum()/weight.sum().clamp_min(1)+error.mean()


@lru_cache(maxsize=1)
def shadow_reference_scale():
    # A view-defined numerical coordinate, not recovered physical exposure.
    neutral=reference_view().inverse(np.full((1,3),5/255,np.float32))
    return float(neutral.mean())


def shadow_relative_loss(output,target,source):
    scale=shadow_reference_scale()
    weight=output.reconstruction_weight*(source.amax(1,keepdim=True)<=.5)
    error=(torch.asinh(output.acescg/scale)-torch.asinh(target/scale)).square().mean(1,keepdim=True)
    return (error*weight).sum()/weight.sum().clamp_min(1)


def restoration_code_loss(output,target):
    """Direct ordinary-pixel restoration supervision in 8-bit code units.

    The target is fixed data, so its exact OCIO display transform needs no
    gradient. Reconstruction pixels are excluded using source-defined weights.
    """
    values=target.detach().float().permute(0,2,3,1).cpu().contiguous().numpy()
    codes=reference_view().forward(values)
    codes=torch.from_numpy(np.ascontiguousarray(codes)).permute(0,3,1,2).to(target.device)
    ordinary=(output.reconstruction_weight==0).to(output.restored_source.dtype)
    error=((output.restored_source-codes)*255).square().mean(1,keepdim=True)
    return (error*ordinary).sum()/ordinary.sum().clamp_min(1)
