"""Opt-in reference-gradient supervision around observed missing-range regions."""
import torch
from torch import nn
from torch.nn import functional as F

from training.dequantization import DequantizationLoss


class ReconstructionGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.view = DequantizationLoss(policy='observed_clip_v1')

    def forward(self, output, target, source):
        if output.shape != target.shape or source.shape != output.shape:
            raise ValueError('expected matching BCHW source, output and target')
        with torch.autocast(device_type=output.device.type, enabled=False):
            with torch.no_grad():
                top = self.view.source_code(source).amax(1, keepdim=True)
                missing = ((top >= 254.5/255) | (top <= 3/255)).float()
                support = F.max_pool2d(missing, 5, stride=1, padding=2)
            loss = output.float().sum()*0
            count = 0
            for ev in (-3, 0, 3):
                with torch.no_grad():
                    truth = self.view.source_code(self.view.forward_view(target * 2.**ev))*255
                pred = self.view.source_code(self.view.forward_view(output * 2.**ev))*255
                for axis in (-1,-2):
                    if output.shape[axis] < 2:
                        continue
                    pair = torch.maximum(support.narrow(axis,0,support.shape[axis]-1),
                                         support.narrow(axis,1,support.shape[axis]-1))
                    error = F.smooth_l1_loss(pred.diff(dim=axis),truth.diff(dim=axis),reduction='none',beta=.25)
                    loss = loss + self.view._mean(error,pair)
                    count += 1
            return {'total': loss/max(count,1), 'support_fraction': support.mean()}
