"""Frozen preservation-pair review. No optimizer, training or checkpoint export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline.commercial_preflight import sha256
from rudra.sdr2hdr import sdr2hdr_loss
from tools.diagnose_preservation import diagnose
from training.bench_val import CLIPPED, CRUSHED, LUMA_2020, pu21


def gradient_relation(a, b):
    na, nb = a.norm(), b.norm()
    return {"cosine": float(torch.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else None,
            "norm_ratio": float(na / nb) if nb > 0 else None}


def error_partition(pred, baseline, reference, sdr):
    """Additive RGB PU squared error, grouped by reference luminance or SDR code.

    Each contribution divides by the full frame's channel count; sums recover
    full-frame MSE. Ratios of these contributions are not PSNR deltas.
    """
    ep = (pu21(pred * 10000) - pu21(reference * 10000)).square()
    eb = (pu21(baseline * 10000) - pu21(reference * 10000)).square()
    y = (reference * LUMA_2020.to(reference.device)[None, :, None, None]).sum(1, keepdim=True) * 10000
    top = sdr.amax(1, keepdim=True)
    bands = {f"{lo}-{hi}": (y >= lo) & (y < hi)
             for lo, hi in ((0, 1), (1, 10), (10, 100), (100, 1000), (1000, float("inf")))}
    regions = {"crushed": top <= CRUSHED, "clipped": top >= CLIPPED,
               "ordinary": (top > CRUSHED) & (top < CLIPPED)}
    def measure(groups):
        return {name: {"pixel_fraction": float(mask.float().mean()),
                       "model_mse": float((ep * mask).mean()),
                       "baseline_mse": float((eb * mask).mean())}
                for name, mask in groups.items()}
    return {"luminance_bands": measure(bands), "sdr_regions": measure(regions),
            "full_model_mse": float(ep.mean()), "full_baseline_mse": float(eb.mean())}


def parameter_probe(model, sdr, target, curve, *, amp=False):
    """Compare parameter gradients at fixed weights, without an optimizer.

    Both loss domains are evaluated at every checkpoint. Region focus is an
    existing-option counterfactual, not a proposed or fitted training recipe.
    """
    flags = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(True)
    try:
        with torch.autocast(device_type=sdr.device.type, dtype=torch.bfloat16, enabled=amp):
            output = model(sdr, baseline_curve=curve, record_preclamp=True)
            post = sdr2hdr_loss(output, sdr, target, crushed_hold=1)
            pre = sdr2hdr_loss(output, sdr, target, crushed_hold=1, crushed_hold_domain="preclamp")
            region = sdr2hdr_loss(output, sdr, target, crushed_hold=1,
                                 crushed_hold_domain="preclamp", region_focus=.5)
            terms = {"task_without_hold": post["total"] - post["crushed_change"],
                     "post_hold": post["crushed_change"], "pre_hold": pre["crushed_change"],
                     "post_total": post["total"], "pre_total": pre["total"],
                     "region_half_pre_total": region["total"],
                     "weighted_chroma": .1 * post["chroma"],
                     "weighted_highlight": .5 * post["highlight"],
                     "log_l1": post["log_l1"],
                     "pu21_mse": (pu21(output.hdr * 10000) - pu21(target * 10000)).square().mean()}
        parameters = tuple(model.parameters())
        vectors, stats = {}, {}
        for name, value in terms.items():
            grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                for p, g in zip(parameters, grads)])
            if not torch.isfinite(vector).all() or not torch.isfinite(value):
                raise ValueError("nonfinite parameter diagnostic")
            vectors[name] = vector
            norm = float(vector.norm())
            stats[name] = {"value": float(value.detach()), "gradient_norm": norm,
                           "unit_norm_clip_scale": min(1., 1. / (norm + 1e-6))}
        pairs = [(a, b) for a, b in (
            ("pre_hold", "task_without_hold"), ("post_hold", "task_without_hold"),
            ("pre_total", "post_total"), ("pre_total", "pu21_mse"),
            ("post_total", "pu21_mse"), ("region_half_pre_total", "pu21_mse"),
            ("weighted_chroma", "pu21_mse"), ("weighted_highlight", "pu21_mse"),
            ("weighted_chroma", "pre_total"))]
        return {"terms": stats, "relations": {f"{a}__{b}": gradient_relation(vectors[a], vectors[b])
                                                for a, b in pairs}}
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=300)
    args = parser.parse_args()
    diagnose(args.root, args.recipe, args.out, args.seconds, arms=("control", "preclamp"),
             parameter_probe=parameter_probe, frame_probe=error_partition)
    with (args.out / "extension.json").open("x", encoding="utf-8") as handle:
        json.dump({"algorithm_sha256": sha256(Path(__file__)),
                   "limitations": ["single-crop FP32 parameter gradients; actual training used batches of four and BF16",
                                   "gradient cosines are local first-order diagnostics, not AdamW update predictions",
                                   "default loss coefficients match the frozen recipe; no ceiling metadata in these draws"]},
                  handle, indent=2)
