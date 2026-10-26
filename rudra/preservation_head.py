"""Differentiable source-code-bounded restoration, separate from HDR reconstruction."""
import torch


def bounded_source_head(canonical, raw_correction, reconstructed, inverse_view, policy='legacy_nearwhite'):
    """BCHW BT.2020 sRGB in; radiance out via a supplied differentiable inverse view.

    Ordinary source channels change by at most 0.49/255. Lost-range regions
    blend into the separate HDR prediction. No weights or display curve hidden
    here: callers explicitly provide the inverse matching their checkpoint.
    """
    from rudra.sdr2hdr import srgb_to_linear, linear_to_srgb
    from rudra.delivery.colorspace import rgb_to_rgb_matrix
    from rudra.reconstruction_policy import reconstruction_bands
    hi_band, lo_band = reconstruction_bands(policy)
    if canonical.shape != raw_correction.shape or canonical.shape != reconstructed.shape:
        raise ValueError('head inputs must have identical BCHW shapes')
    with torch.autocast(device_type=canonical.device.type, enabled=False):
        x, raw, hdr = canonical.float(), raw_correction.float(), reconstructed.float()
        to_source = torch.as_tensor(rgb_to_rgb_matrix('rec2020', 'rec709'), device=x.device, dtype=x.dtype)
        to_canon = torch.as_tensor(rgb_to_rgb_matrix('rec709', 'rec2020'), device=x.device, dtype=x.dtype)
        source = linear_to_srgb(torch.einsum('ij,bjhw->bihw', to_source, srgb_to_linear(x))).clamp(0, 1)
        restored = (source + (0.49 / 255.0) * torch.tanh(raw)).clamp(0, 1)
        restored_canon = linear_to_srgb(torch.einsum('ij,bjhw->bihw', to_canon, srgb_to_linear(restored))).clamp(0, 1)
        bounded = inverse_view(restored_canon)
        top = source.amax(1, keepdim=True)
        def smooth(v, lo, hi):
            t = ((v - lo) / (hi - lo)).clamp(0, 1)
            return t.square() * (3 - 2 * t)
        mask = torch.maximum(smooth(top, *hi_band), 1 - smooth(top, *lo_band))
        weight = ((mask - .5) * 2).clamp(0, 1)
        return bounded * (1 - weight) + hdr * weight
