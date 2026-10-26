"""Opt-in experimental code-domain supervision; disabled by default.

Inspired by SingleHDR (Liu et al., CVPR 2020), section 3.2: explicitly
supervise dequantization before radiance expansion. This version uses our
declared picker view, source-code units and reference gradients. It does not
claim to implement a standard ACES display transform.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class DequantizationLoss(nn.Module):
    def __init__(self, view='bt1886', gradient_weight=0.5, policy='legacy_nearwhite'):
        super().__init__()
        from training.contract_checks import _view_table
        from rudra.delivery.colorspace import rgb_to_rgb_matrix
        from rudra.reconstruction_policy import reconstruction_bands
        self.hi_band, self.lo_band = reconstruction_bands(policy)
        if not math.isfinite(gradient_weight) or gradient_weight < 0:
            raise ValueError('gradient_weight must be finite and nonnegative')
        codes, values = _view_table(view)
        self.view = view
        self.gradient_weight = float(gradient_weight)
        self.register_buffer('codes', torch.tensor(codes, dtype=torch.float32))
        self.register_buffer('values', torch.tensor(values, dtype=torch.float32))
        self.register_buffer('to_source', torch.tensor(
            rgb_to_rgb_matrix('rec2020', 'rec709'), dtype=torch.float32))

    def source_code(self, canonical):
        from rudra.sdr2hdr import srgb_to_linear
        linear = torch.einsum('ij,bjhw->bihw', self.to_source, srgb_to_linear(canonical.float()))
        linear = linear.clamp(0, 1)
        # torch.where evaluates both branches: pow(0, 1/2.4) has an infinite
        # derivative even when the linear branch is selected. Bound ONLY the
        # unused nonlinear branch to keep exact black and finite derivatives.
        return torch.where(linear <= .0031308, 12.92 * linear,
                           1.055 * linear.clamp_min(.0031308).pow(1 / 2.4) - .055).clamp(0, 1)

    def forward_view(self, radiance):
        # Invert the same sampled inverse-view used by the independent numpy
        # contract checker. Interpolation remains differentiable in radiance.
        x = radiance.float().clamp(self.values[0], self.values[-1])
        index = torch.searchsorted(self.values, x.contiguous()).clamp(1, self.values.numel() - 1)
        lo, hi = self.values[index - 1], self.values[index]
        frac = (x - lo) / (hi - lo).clamp_min(1e-12)
        return self.codes[index - 1] + frac * (self.codes[index] - self.codes[index - 1])

    @staticmethod
    def _mean(error, mask):
        return (error * mask).sum() / (mask.sum() * error.shape[1]).clamp_min(1)

    def forward(self, output, target, canonical_input):
        if output.shape != target.shape or output.shape != canonical_input.shape or output.ndim != 4 or output.shape[1] != 3:
            raise ValueError('expected matching BCHW RGB tensors')
        with torch.autocast(device_type=output.device.type, enabled=False):
            with torch.no_grad():
                source = self.source_code(canonical_input)
                top = source.amax(1, keepdim=True)
                # Entirely outside the bounded head's highlight/shadow blend.
                keep = ((top >= self.lo_band[1]) & (top <= self.hi_band[0])).float()
                truth = self.source_code(self.forward_view(target)) * 255
            pred = self.source_code(self.forward_view(output)) * 255
            pixel = self._mean(F.smooth_l1_loss(pred, truth, reduction='none', beta=.25), keep)
            gradient = output.float().sum() * 0
            for axis in (-1, -2):
                if output.shape[axis] < 2:
                    continue
                pair = keep.narrow(axis, 0, keep.shape[axis] - 1) * keep.narrow(axis, 1, keep.shape[axis] - 1)
                error = F.smooth_l1_loss(pred.diff(dim=axis), truth.diff(dim=axis), reduction='none', beta=.25)
                gradient = gradient + self._mean(error, pair)
            return {'total': pixel + self.gradient_weight * gradient,
                    'code': pixel, 'gradient': gradient, 'ordinary_fraction': keep.mean()}


def validate_training_contract(args, experiment):
    """The auxiliary view must match the actual targets and picker, without regrading."""
    import json
    from pathlib import Path
    from rudra.source_baseline import baseline_curve_index
    from training.sdr2hdr_dataset import effective_baseline_curve
    if not getattr(args, 'bounded_source', False) or not getattr(args, 'baseline_source', False):
        raise ValueError('dequantization loss requires a bounded picker model')
    if any(getattr(args, name, 0) != 0 for name in
           ('baseline_dropout', 'degradation_probability', 'augmentation_strength')) or experiment.get('shadow_codec_p', 0) != 0:
        raise ValueError('dequantization loss requires fixed, unaugmented source-preserving pairs')
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if not row.get('excluded')]
    for row in rows:
        if row.get('target_mode') != 'source_preserving' or row.get('declared_view') != 'bt1886':
            raise ValueError('dequantization loss currently requires declared BT1886 source-preserving targets')
        if effective_baseline_curve(row, False, False, False, 0) != baseline_curve_index('bt1886'):
            raise ValueError('dequantization target view does not match the sample picker')
