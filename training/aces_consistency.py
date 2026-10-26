"""Shadow error-consistency regularization under subpixel translations."""
import torch
from torch.nn import functional as F
from training.aces_losses import shadow_reference_scale


def translate_constant(tensor,shifts):
    """CPU bilinear translations of fixed source/target data; no learned warp."""
    if tensor.device.type!='cpu' or tensor.requires_grad:raise ValueError('fixed CPU tensors required')
    b,_,h,w=tensor.shape
    theta=torch.eye(2,3,dtype=tensor.dtype)[None].repeat(b,1,1)
    theta[:,0,2]=2*shifts[:,0]/w;theta[:,1,2]=2*shifts[:,1]/h
    grid=F.affine_grid(theta,tensor.shape,align_corners=False)
    return F.grid_sample(tensor,grid,mode='bilinear',padding_mode='border',align_corners=False)


def quantize_source8(tensor):
    """Reapply an 8-bit display-code quantizer to fixed transformed inputs."""
    if tensor.requires_grad or not torch.isfinite(tensor).all():raise ValueError('finite fixed input required')
    if ((tensor<0)|(tensor>1)).any():raise ValueError('source codes in[0,1] required')
    return torch.round(tensor*255)/255


def shadow_error_consistency(pred,target,shift_pred,shift_target,weight,shift_weight,border=10):
    """Match transformed and original reconstruction errors, not raw images.

    Sparse-Jacobian-style loss (Eilertsen et al. CVPR2019 Eq8), adapted to
    signed asinh coordinates and common source-defined shadow support.
    """
    s=shadow_reference_scale()
    error=torch.asinh(pred/s)-torch.asinh(target/s)
    shifted_error=torch.asinh(shift_pred/s)-torch.asinh(shift_target/s)
    area=slice(border,-border) if border else slice(None)
    valid=torch.minimum(weight,shift_weight)[:,:,area,area]
    distance=(shifted_error-error).square().mean(1,keepdim=True)[:,:,area,area]
    return (distance*valid).sum()/valid.sum().clamp_min(1)
