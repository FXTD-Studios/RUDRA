"""Read-only checkpoint diagnosis: no optimizer, training, or weight export.

Reuses an inspected validation set; inference interventions are diagnostic only.
Output-space gradients on fingerprint-verified training crops do not predict
parameter updates or establish a successful training objective.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline.commercial_preflight import sha256
from rudra.sdr2hdr import SDR2HDRNet, luminance, sdr2hdr_loss
from rudra.source_baseline import baseline_curve_index
from training.bench_val import BenchVal, CLIPPED, CRUSHED, LUMA_2020, pu21, pu_psnr, region_error
from training.routing_replay import augmentation_rng, fingerprint
from training.sdr2hdr_dataset import SDRHDRDataset, exposure_push, read_jsonl


def residual_gate(sdr, mode="all"):
    y = luminance(sdr.float().clamp(0, 1))
    hi, sh = torch.sigmoid((y - .82) * 24), torch.sigmoid((.10 - y) * 24)
    if mode == "highlights":
        return hi
    if mode != "all":
        raise ValueError("unsupported diagnostic mode")
    return torch.maximum(hi, sh)


def reconstruct(output, sdr, model, mode="all"):
    if model.gate is not None or model.shadow_gate is not None or model.curve is not None:
        raise ValueError("diagnostic reconstruction supports the frozen pilot only")
    pre = torch.log1p(output.baseline * model.log_scale) + output.log_residual * residual_gate(sdr, mode)
    upper = torch.log1p(torch.tensor(model.max_hdr * model.log_scale, device=sdr.device, dtype=sdr.dtype))
    return torch.expm1(pre.clamp(0, upper)) / model.log_scale, pre


def masks(sdr):
    top = sdr.clamp(0, 1).amax(1, keepdim=True)
    crush = F.max_pool2d((top <= 2 / 255).float(), 3, 1, 1)
    destroyed = F.max_pool2d(((top >= CLIPPED) | (top <= CRUSHED)).float(), 7, 1, 3)
    clipped = F.max_pool2d((top >= CLIPPED).float(), 7, 1, 3)
    return crush, destroyed, clipped


def diagnostic_variants(output, sdr, model):
    pred, pre = reconstruct(output, sdr, model)
    torch.testing.assert_close(pred, output.hdr, rtol=0, atol=0)
    crush, _, clipped = masks(sdr)
    highlights, _ = reconstruct(output, sdr, model, "highlights")
    return {
        "original": pred,
        "existing_highlights_mode": highlights,
        "existing_preserve_outside": output.baseline + torch.maximum(output.highlight_mask, output.shadow_mask) * (pred - output.baseline),
        "diagnostic_crushed_hold": torch.where(crush.bool(), output.baseline, pred),
        "diagnostic_clip_only": torch.where(clipped.bool(), pred, output.baseline),
    }, pre


def score(pred, base, ref, sdr):
    top = sdr[0].amax(0)
    p, b, r = pred[0], base[0], ref[0]
    out = {"pu_gain_db": pu_psnr(p * 10000, r * 10000) - pu_psnr(b * 10000, r * 10000)}
    for name, mask in (("clip", top >= CLIPPED), ("crushed", top <= CRUSHED)):
        ep, eb = region_error(p, r, mask), region_error(b, r, mask)
        out[f"{name}_gain_stops"] = None if ep is None else eb - ep
    return out


def dark_stats(pred, base, ref, sdr, pre):
    mask = sdr[0].amax(0) <= CRUSHED
    count = int(mask.sum())
    if not count:
        return {"pixels": 0, "negative_channels": 0, "zero_rgb_pixels": 0}
    y = lambda x: (x[0].double() * LUMA_2020.to(x.device).double()[:, None, None]).sum(0)[mask] * 10000
    p, b, r = y(pred), y(base), y(ref)
    return {"pixels": count,
            "negative_channels": int((pre[0][:, mask] < 0).sum()),
            "zero_rgb_pixels": int((pred[0][:, mask] == 0).all(0).sum()),
            "reference_above_floor_pixels": int((r > .005).sum()),
            "black_but_reference_above_floor_pixels": int(((p < .005) & (r > .005)).sum()),
            "prediction_sum_nits": float(p.sum()), "baseline_sum_nits": float(b.sum()),
            "reference_sum_nits": float(r.sum())}


def pu_bands(pred, base, ref):
    ep = (pu21(pred * 10000) - pu21(ref * 10000)).square()
    eb = (pu21(base * 10000) - pu21(ref * 10000)).square()
    out = {}
    for lo, hi in ((0, 1), (1, 10), (10, 100), (100, 1000), (1000, float("inf"))):
        mask = (ref * 10000 >= lo) & (ref * 10000 < hi)
        out[f"{lo}-{hi}"] = {"fraction": float(mask.float().mean()),
                             "model_mse_contribution": float((ep * mask).mean()),
                             "baseline_mse_contribution": float((eb * mask).mean())}
    return out


def gradient_probe(output, sdr, target, model):
    """Frozen features; differentiate with respect to an independent raw residual.

    Pre-clamp hold is a diagnostic candidate in the same log units and mask as
    crushed_hold. It is NOT installed in the training loss or used for fitting.
    """
    raw = output.log_residual.detach().clone().requires_grad_(True)
    working = replace(output, log_residual=raw)
    hdr, pre = reconstruct(working, sdr, model)
    working = replace(working, hdr=hdr)
    hold, destroyed, _ = masks(sdr)
    denominator = hold.sum() * 3 + 1e-6
    # Algebraically pre - base_log, avoiding cancellation near black.
    pre_hold = ((raw * residual_gate(sdr)).abs() * hold).sum() / denominator
    losses = sdr2hdr_loss(working, sdr, target, region_focus=0, crushed_hold=1)
    region = sdr2hdr_loss(working, sdr, target, region_focus=.5, crushed_hold=1)
    terms = {"original_total": losses["total"], "region_focus_half_total": region["total"],
             "post_clamp_hold": losses["crushed_change"], "diagnostic_pre_clamp_hold": pre_hold,
             "log_l1": losses["log_l1"], "weighted_chroma": .1 * losses["chroma"],
             "pu21_mse": (pu21(hdr * 10000) - pu21(target * 10000)).square().mean()}
    negative = (pre < 0)
    negative_held = negative & hold.bool()
    out = {"channels": raw.numel(), "negative_channels": int(negative.sum()),
           "held_channels": int(hold.sum()) * 3, "negative_held_channels": int(negative_held.sum()),
           "region_focus_exempts_held_negative_channels": int((negative_held & destroyed.bool()).sum()),
           "terms": {}}
    for name, term in terms.items():
        grad = torch.autograd.grad(term, raw, retain_graph=True)[0]
        if not torch.isfinite(grad).all() or not torch.isfinite(term):
            raise ValueError("nonfinite diagnostic gradient")
        out["terms"][name] = {"value": float(term.detach()), "gradient_l1": float(grad.abs().sum()),
                              "negative_held_gradient_l1": float(grad[negative_held].abs().sum()),
                              "negative_held_nonzero_gradients": int((grad[negative_held] != 0).sum())}
    return out


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["condition"], row["ev"])].append(row)
    results = []
    for (arm, condition, ev), group in groups.items():
        variants = {}
        for name in group[0]["variants"]:
            values = {}
            for metric in group[0]["variants"][name]:
                samples = [r["variants"][name][metric] for r in group if r["variants"][name][metric] is not None]
                values[metric] = {"mean": float(np.mean(samples)) if samples else None, "frames": len(samples)}
            variants[name] = values
        dark = {key: sum(r["dark"].get(key, 0) for r in group) for key in set().union(*(r["dark"] for r in group))}
        results.append({"arm": arm, "condition": condition, "ev": ev, "frames": len(group),
                        "variants": variants, "dark_pixel_sums": dark})
    return results


def verify_historical(rows, root, arms=("control", "corrected")):
    """Fail rather than silently diagnosing different inputs or metrics."""
    worst = 0.0
    for arm in arms:
        history = read_jsonl(root / arm / "bench_val.jsonl")
        expected = next(r for r in history if r["step"] == 200)["frames"]
        actual = {(r["asset_id"], r["condition"], r["ev"]): r["variants"]["original"] for r in rows if r["arm"] == arm}
        if len(actual) != len(expected):
            raise ValueError("historical coverage mismatch")
        for row in expected:
            match = actual[(row["asset_id"], row["condition"], row["ev"])]
            for key in ("pu_gain_db", "clip_gain_stops", "crushed_gain_stops"):
                if key not in row:
                    continue
                if row[key] is None or match[key] is None:
                    if row[key] != match[key]:
                        raise ValueError("historical region coverage mismatch")
                else:
                    difference = abs(row[key] - match[key])
                    worst = max(worst, difference)
                    if difference > 1e-5:
                        raise ValueError(f"historical score mismatch: {arm} {key} {difference}")
    return worst


def diagnose(root, recipe_path, out, seconds=600, *, arms=("control", "corrected"),
             parameter_probe=None, frame_probe=None):
    if out.exists() or not out.resolve().is_relative_to((REPO / "reports").resolve()):
        raise ValueError("use a fresh output directory under repository reports")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if arms not in (("control", "corrected"), ("control", "preclamp")):
        raise ValueError("unsupported paired experiment")
    preservation = arms == ("control", "preclamp")
    verified = json.loads((root / "verified_summary.json").read_text()) if preservation else None
    manifest = Path(recipe["manifest"])
    pins = {REPO / p: h for p, h in recipe["code_sha256"].items()}
    pins[manifest] = recipe["manifest_sha256"]
    for p, h in pins.items():
        if sha256(p) != h:
            raise ValueError(f"frozen input changed: {p}")
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started > seconds:
            raise TimeoutError("diagnostic cap reached; partial journals retained")
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, hashes = {}, {}
    for arm in (*arms, "initial"):
        checkpoint = root / (arms[-1] if arm == "initial" else arm) / ("step_0000000.pt" if arm == "initial" else "step_0000200.pt")
        if verified and arm != "initial" and sha256(checkpoint) != verified["checkpoints"][arm]["step_200_sha256"]:
            raise ValueError("verified checkpoint hash mismatch")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if preservation and parameter_probe is not None:
            expected = {"shadow_chroma_weight": .15, "shadow_smoothness_weight": .02,
                        "region_focus": 0., "crushed_hold": 1., "curve_head": False,
                        "source_curve": False, "max_grad_norm": 1.}
            if any(saved["config"].get(k) != v for k, v in expected.items()):
                raise ValueError("parameter probe does not match checkpoint loss configuration")
        model = SDR2HDRNet.from_config(saved["config"]).to(device).eval().requires_grad_(False)
        model.load_state_dict(saved["model"], strict=True)
        if verified and arm == "initial" and fingerprint(saved["model"]) != verified["checkpoints"][arms[-1]]["initial_model_sha256"]:
            raise ValueError("verified initialization mismatch")
        if saved["step"] != (0 if arm == "initial" else 200):
            raise ValueError("wrong checkpoint step")
        models[arm], hashes[str(checkpoint)] = model, sha256(checkpoint)
    bv = BenchVal.from_manifest(str(manifest), real_frames=48, rendered_frames=0)
    bv.load()
    rows = []
    with (out / "frames.jsonl").open("x", encoding="utf-8") as journal, torch.no_grad():
        for i, (record, frame) in enumerate(zip(bv.records, bv.frames)):
            deadline()
            conditions = [("real", None, frame.sdr.float(), frame.hdr.float(), frame.baseline_curve)]
            for ev in bv.push_evs:
                sdr, hdr, _ = exposure_push(frame.hdr.float(), ev)
                conditions.append(("push", ev, sdr, hdr, baseline_curve_index("aces")))
            for condition, ev, x, ref, curve in conditions:
                x, ref = x[None].to(device), ref[None].to(device)
                for arm in arms:
                    model = models[arm]
                    output = model(x, baseline_curve=curve)
                    variants, pre = diagnostic_variants(output, x, model)
                    row = {"arm": arm, "asset_id": record["asset_id"], "scene_id": record["scene_id"],
                           "condition": condition, "ev": ev,
                           "variants": {k: score(v, output.baseline, ref, x) for k, v in variants.items()},
                           "dark": dark_stats(output.hdr, output.baseline, ref, x, pre)}
                    if condition == "real":
                        row["bands"] = pu_bands(output.hdr, output.baseline, ref)
                    if frame_probe is not None:
                        row["error_partition"] = frame_probe(output.hdr, output.baseline, ref, x)
                    rows.append(row)
                    journal.write(json.dumps(row, allow_nan=False) + "\n"); journal.flush()
            if (i + 1) % 8 == 0:
                print(f"paired diagnosis {i+1}/48 frames", flush=True)
    parity = verify_historical(rows, root, arms)
    config = json.loads((root / arms[-1] / "config.json").read_text())
    dataset = SDRHDRDataset(manifest, source_curve_dropout=(config["source_curve_dropout"] if config["source_curve"] else 0),
                           **{k: config[k] for k in (
        "crop_size", "augmentation_strength", "degradation_probability",
        "exposure_push", "exposure_push_max_ev", "baseline_dropout")})
    trace = read_jsonl(root / f"{arms[-1]}_samples.jsonl")
    chosen = sorted([r for curve in ("bt1886", "aces") for r in
                     [r for r in trace if r["baseline_curve"] == baseline_curve_index(curve)][:8]], key=lambda r: r["draw"])
    probes = []
    with (out / "gradients.jsonl").open("x", encoding="utf-8") as journal:
        for draw in chosen:
            deadline()
            with augmentation_rng(draw["augmentation_seed"]):
                sample = dataset[draw["record_index"]]
            if fingerprint({k: v for k, v in sample.items() if k != "baseline_curve"}) != draw["sample_sha256"] or int(sample["baseline_curve"]) != draw["baseline_curve"]:
                raise ValueError("training crop fingerprint mismatch")
            sdr, target = sample["sdr"][None].to(device), sample["hdr"][None].to(device)
            if parameter_probe is not None and torch.isfinite(sample["ceiling"]).any():
                raise ValueError("parameter probe requires uncensored training draws")
            for arm, model in models.items():
                deadline()
                curve = baseline_curve_index("bt1886") if arm == "control" and not preservation else draw["baseline_curve"]
                with torch.no_grad():
                    output = model(sdr, baseline_curve=curve)
                row = {"arm": arm, **draw, "evaluated_baseline_curve": curve,
                       "probe": (gradient_probe(output, sdr, target, model) if parameter_probe is None else
                                 parameter_probe(model, sdr, target, curve))}
                probes.append(row); journal.write(json.dumps(row, allow_nan=False) + "\n"); journal.flush()
            print(f"verified training crop draw {draw['draw']}", flush=True)
    deadline()
    for p, h in {**pins, **{Path(p): h for p, h in hashes.items()}}.items():
        if sha256(p) != h:
            raise ValueError(f"frozen input changed during diagnosis: {p}")
    summary = {"elapsed_seconds": time.monotonic() - started, "device": str(device),
               "checkpoint_sha256": hashes, "manifest_sha256": sha256(manifest),
               "algorithm_sha256": sha256(Path(__file__)), "frozen_code_verified": True,
               "probe_space": "parameters" if parameter_probe is not None else "output residual",
               "historical_max_absolute_metric_difference": parity,
               "aggregates": aggregate(rows), "gradient_probes": len(probes),
               "verified_training_draws": len(chosen), "training_draw_ids": [r["draw"] for r in chosen],
               "journals_sha256": {p.name: sha256(p) for p in (out / "frames.jsonl", out / "gradients.jsonl")},
               "limitations": ["same inspected 48 Sparks frames; diagnostic, not promotion or generalization",
                   "16 deliberately stratified training crops, not representative training gradients",
                   "FP32 diagnostic gradients; no optimizer/parameter update, not the actual BF16 training trajectory",
                   "hard-mask interventions diagnose spatial damage; not shipping inference changes",
                   "source dataset untouched; historical full-media preflight not repeated", "no new training"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("aggregates", "checkpoint_sha256")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=600)
    a = p.parse_args()
    diagnose(a.root, a.recipe, a.out, a.seconds)
