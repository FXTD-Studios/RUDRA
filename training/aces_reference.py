"""Experimental exact OCIO forward with finite-difference first derivatives.

CPU reference bridge for feasibility only. Transfers CUDA tensors to CPU and
back; not TorchScript/native-exportable, and no higher-order differentiation.
The backward is a finite-difference approximation, not an analytic ACES Jacobian.
"""
import math
from functools import lru_cache
import numpy as np
import torch
from torch.autograd.function import once_differentiable
from rudra.aces_view import ACES2View


@lru_cache(maxsize=1)
def reference_view():
    return ACES2View()


def _numpy(source):
    return source.detach().float().permute(0,2,3,1).cpu().contiguous().numpy()


def _tensor(array, source):
    return torch.from_numpy(np.ascontiguousarray(array)).permute(0,3,1,2).to(source.device, source.dtype)


class _Inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, step):
        if source.ndim != 4 or source.shape[1] != 3 or source.dtype != torch.float32:
            raise ValueError('expected float32 BCHW RGB codes')
        ctx.save_for_backward(source)
        ctx.step = step
        return _tensor(reference_view().inverse(_numpy(source)), source)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        source, = ctx.saved_tensors
        values = _numpy(source)
        upstream = _numpy(grad_output)
        gradient = np.empty_like(values)
        for channel in range(3):
            plus, minus = values.copy(), values.copy()
            plus[...,channel] = np.minimum(values[...,channel]+ctx.step,1)
            minus[...,channel] = np.maximum(values[...,channel]-ctx.step,0)
            width = plus[...,channel]-minus[...,channel]
            jacobian = (reference_view().inverse(plus)-reference_view().inverse(minus))/width[...,None]
            gradient[...,channel] = np.sum(jacobian*upstream,axis=-1)
        if not np.isfinite(gradient).all():
            raise ValueError('nonfinite ACES finite-difference gradient')
        return _tensor(gradient,source), None


def inverse_aces2_reference(source, step=1/4096):
    if not math.isfinite(step) or step <= 0 or step > .01:
        raise ValueError('finite-difference step must be in (0,.01]')
    return _Inverse.apply(source, float(step))
