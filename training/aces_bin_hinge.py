"""The bin hinge (0 EV training kit, 2 Nov 2026).

On partially clipped highlight pixels, penalise the RAW prediction's 0 EV
ACES view only where it sits more than `tol_codes` from the SOURCE codes,
squared beyond it:

    bin = mean_mask( relu( max_c |view(pred) - source| * 255 - tol )^2 )

Why: at inference the shipped 1-code constraint acts per frame on raw pixels
more than a code from the source, and per-frame action is what broke the
aligned-residual motion gate whenever it was tightened (RESULT_PARTIAL_TIGHTEN_
2026_11_02.md). A model that lands inside half a code by itself leaves the
constraint nothing to do, and the temporal term already acts on what it
outputs. Inference is unchanged.

Also returns the shares of masked pixels over the tolerance and over 1 code
(how often the shipped constraint would have to act): the training
diagnostics the registration names.
"""
import torch


def bin_hinge(view0, source, mask, tol_codes=0.5):
    """view0: the raw prediction's 0 EV view, BCHW display codes; source: BCHW
    source codes; mask: B1HW in {0,1} (partial highlights, crop interior).
    -> (value, share_over_tol, share_over_one_code)."""
    dev = ((view0 - source).abs() * 255.0).amax(1, keepdim=True)
    n = mask.sum().clamp_min(1)
    value = ((torch.relu(dev - tol_codes) ** 2) * mask).sum() / n
    with torch.no_grad():
        over_tol = float(((dev > tol_codes).float() * mask).sum() / n)
        over_one = float(((dev > 1.0).float() * mask).sum() / n)
    return value, over_tol, over_one
