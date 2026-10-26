"""Experimental target-anchored shadow colour loss, disabled by default.

Signed asinh compresses radiance without erasing negative source-gamut channels.
The 0.1-nit scale sets the linear/log transition; it is not a clipping ceiling.
"""
import torch
from torch import nn
from torch.nn import functional as F
from training.dequantization import DequantizationLoss


class SignedShadowLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.view = DequantizationLoss(policy='observed_clip_v1')

    def forward(self, output, target, source):
        if output.shape != target.shape or source.shape != output.shape:
            raise ValueError('expected matching BCHW tensors')
        with torch.autocast(device_type=output.device.type, enabled=False):
            with torch.no_grad():
                mask = (self.view.source_code(source).amax(1, keepdim=True) <= 16/255).float()
                truth = torch.einsum('ij,bjhw->bihw', self.view.to_source, target.float())
            pred = torch.einsum('ij,bjhw->bihw', self.view.to_source, output.float())
            error = F.smooth_l1_loss(torch.asinh(pred / 1e-5), torch.asinh(truth / 1e-5),
                                    reduction='none', beta=.25)
            return self.view._mean(error, mask)
