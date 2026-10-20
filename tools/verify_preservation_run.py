"""Verify a completed preservation pair without fitting or changing weights."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline.commercial_preflight import sha256
from training.bounded_pilot import footprint
from training.routing_ab import (PUSH_SCOPE, approved_recipe, preservation_comparison, push_baseline_ab,
                                 push_baseline_comparison, region_comparison,
                                 region_focus, arms, repeatability, repeatability_comparison)
from training.routing_replay import fingerprint
from training.sdr2hdr_dataset import read_jsonl


def training_summary(rows, steps):
    rows = [r for r in rows if "gradient_norm_before_clip" in r]
    if [r["step"] for r in rows] != list(range(1, steps + 1)):
        raise ValueError("missing or duplicate training diagnostic steps")
    keys = ("gradient_norm_before_clip", "held_channels", "negative_held_channels",
            "negative_held_fraction", "hold_preclamp", "hold_postclamp", "total")
    for row in rows:
        if not all(math.isfinite(row[k]) for k in keys):
            raise ValueError("nonfinite training diagnostic")
        if not 0 <= row["negative_held_channels"] <= row["held_channels"]:
            raise ValueError("invalid held-region counts")
        fraction = row["negative_held_channels"] / max(1, row["held_channels"])
        if abs(fraction - row["negative_held_fraction"]) > 1e-6:
            raise ValueError("inconsistent held-region fraction")
    def summarize(group):
        held = sum(r["held_channels"] for r in group)
        negative = sum(r["negative_held_channels"] for r in group)
        return {"steps": len(group), "held_channels": int(held), "negative_held_channels": int(negative),
                "negative_held_fraction": negative / held if held else None,
                "gradient_norm_mean": float(np.mean([r["gradient_norm_before_clip"] for r in group])),
                "gradient_norm_max": max(r["gradient_norm_before_clip"] for r in group),
                "hold_preclamp_mean": float(np.mean([r["hold_preclamp"] for r in group])),
                "hold_postclamp_mean": float(np.mean([r["hold_postclamp"] for r in group]))}
    return {"overall": summarize(rows), "windows": [
        {"first_step": group[0]["step"], "last_step": group[-1]["step"], **summarize(group)}
        for start in range(0, steps, 50) if (group := rows[start:start + 50])]}


def luminance_effect(reports, manifest_rows, pair):
    """Pre-registered dark-frame split (tools/frame_luminance_breakdown.py)."""
    from tools.frame_luminance_breakdown import DARK_NITS, breakdown
    split = {arm: breakdown(reports[arm], manifest_rows) for arm in pair}
    real = {arm: next(c for c in split[arm]["conditions"] if c["condition"] == "real") for arm in pair}
    dark = {arm: real[arm]["bins"][0] for arm in pair}
    if dark[pair[0]]["frames"] != dark[pair[1]]["frames"]:
        raise ValueError("unpaired dark-frame coverage")
    delta = (dark[pair[1]]["mean"] - dark[pair[0]]["mean"]) if dark[pair[0]]["frames"] else None
    return {"dark_nits_threshold": DARK_NITS, "dark_real_frames": dark[pair[0]]["frames"],
            "dark_real_pu_delta_db": delta, "arms": split}


def verify(recipe_path, approval_path):
    recipe = approved_recipe(recipe_path, approval_path)
    if recipe["scope"] not in ("internal-preservation-ab", "internal-region-focus-ab", "internal-repeatability",
                               PUSH_SCOPE):
        raise ValueError("preservation recipe required")
    root, steps = Path(recipe["output_root"]), recipe["arm_steps"]
    pair = arms(recipe)
    candidate = pair[1]
    is_region = region_focus(recipe)
    is_repeat = repeatability(recipe)
    is_push = push_baseline_ab(recipe)
    if (root / "verified_summary.json").exists():
        raise ValueError("verification summary already exists")
    states, traces, checkpoints, reports, config, gradients = {}, {}, {}, {}, {}, {}
    torch.set_num_threads(2)
    source_records = read_jsonl(recipe["manifest"])
    media_count = len({str(Path(r[k]).resolve()) for r in source_records for k in ("sdr_path", "hdr_path")})
    for arm in pair:
        state = json.loads((root / f"{arm}_status.json").read_text())
        if (state["state"] != "completed" or state["last_step"] != steps
                or state["sample_count"] != steps * 4
                or (state["changed_baselines"] != 0) != (is_push and arm != "control")
                or not 0 < state["elapsed_seconds_after_preflight"] <= recipe["arm_seconds"]):
            raise ValueError(f"incomplete or over-budget arm: {arm}")
        if f"verified {media_count}/{media_count} media files" not in (root / f"{arm}.log").read_text():
            raise ValueError("missing full-media preflight completion")
        states[arm] = state
        trace = read_jsonl(root / f"{arm}_samples.jsonl")
        if len(trace) != steps * 4 or [r["draw"] for r in trace] != list(range(steps * 4)):
            raise ValueError("incomplete replay")
        traces[arm] = trace
        initial = torch.load(root / arm / "step_0000000.pt", map_location="cpu", weights_only=False)
        final_path = root / arm / f"step_{steps:07d}.pt"
        final = torch.load(final_path, map_location="cpu", weights_only=False)
        best = torch.load(root / arm / "best.pt", map_location="cpu", weights_only=False)
        c = final["config"]
        if (initial["step"] != 0 or final["step"] != steps or final["scheduler"]["T_max"] != 1000
                or c["training_use"] != "internal-pilot" or c["resume"] or c["init_checkpoint"]
                or c["commercial_provenance"]["initialization"] != "scratch"
                or c["commercial_provenance"]["release_authorized"] is not False
                or c["manifest_sha256"] != recipe["manifest_sha256"]
                or c["crushed_hold_domain"] != ("postclamp" if arm == "control" and not (is_region or is_repeat or is_push)
                                                else "preclamp")
                or not c["preservation_diagnostics"]):
            raise ValueError("checkpoint ancestry/configuration mismatch")
        if is_region and (c["region_focus"] != (0. if arm == "control" else .5)
                          or not c.get("bench_push_pu") or c["crushed_hold"] != 1.):
            raise ValueError("region-focus configuration mismatch")
        if is_repeat and (c["region_focus"] != 0 or not c.get("bench_push_pu")
                          or c["crushed_hold"] != 1. or not c.get("deterministic")):
            raise ValueError("repeatability configuration mismatch")
        if is_push and (c["region_focus"] != 0 or not c.get("bench_push_pu")
                        or c["crushed_hold"] != 1. or not c.get("deterministic")
                        or c.get("push_baseline") != ("picker" if arm == "control" else "render")):
            raise ValueError("push-baseline configuration mismatch")
        for checkpoint in (initial, final, best):
            if not all(torch.isfinite(t).all() for t in checkpoint["model"].values()):
                raise ValueError("nonfinite checkpoint weights")
        initial_hash = fingerprint(initial["model"])
        if initial_hash != state["initial_model_sha256"]:
            raise ValueError("initial checkpoint fingerprint mismatch")
        config[arm] = c
        checkpoints[arm] = {"step_200_sha256": sha256(final_path), "best_step": best["step"],
                            "initial_model_sha256": initial_hash, "finite_weights": True,
                            "training_use": c["training_use"], "scheduler_horizon": final["scheduler"]["T_max"],
                            "replay_sha256": sha256(root / f"{arm}_samples.jsonl")}
        gradients[arm] = training_summary(read_jsonl(root / arm / "train.jsonl"), steps)
        reports[arm] = next(r for r in read_jsonl(root / arm / "bench_val.jsonl") if r["step"] == steps)
        if reports[arm]["per_ev"] != state["per_ev"]:
            raise ValueError("state/report EV mismatch")
        overall = gradients[arm]["overall"]
        if (overall["held_channels"] != state["held_channels_seen"]
                or overall["negative_held_channels"] != state["negative_held_channels_seen"]
                or overall["steps"] != state["gradient_observations"]):
            raise ValueError("state/training coverage mismatch")
    changed = 0
    if is_push:
        from rudra.source_baseline import baseline_curve_index
        allowed = (baseline_curve_index("bt1886"), baseline_curve_index("aces"))
        for a, b in zip(traces["control"], traces[candidate]):
            if {k: v for k, v in a.items() if k != "baseline_curve"} != {k: v for k, v in b.items() if k != "baseline_curve"}:
                raise ValueError("paired samples differ")
            if a["baseline_curve"] != b["baseline_curve"]:
                if (a["baseline_curve"], b["baseline_curve"]) != allowed:
                    raise ValueError("undeclared baseline change")
                changed += 1
        if not changed or changed != states[candidate]["changed_baselines"]:
            raise ValueError("push-baseline change count mismatch")
    elif traces["control"] != traces[candidate]:
        raise ValueError("paired samples differ")
    if checkpoints["control"]["initial_model_sha256"] != checkpoints[candidate]["initial_model_sha256"]:
        raise ValueError("paired initial weights differ")
    excluded = {"output_dir", "preservation_ab_arm",
                "push_baseline" if is_push else "region_focus" if is_region else "crushed_hold_domain"}
    if {k: v for k, v in config["control"].items() if k not in excluded} != {k: v for k, v in config[candidate].items() if k not in excluded}:
        raise ValueError("more than the declared treatment differs")
    identities = lambda r: (r["asset_id"], r["scene_id"], r["condition"], r["ev"])
    a, b = ([identities(r) for r in reports[arm]["frames"]] for arm in pair)
    if a != b or len(set(a)) != len(a):
        raise ValueError("unpaired/duplicate validation observations")
    scenes = sorted({r["scene_id"] for r in reports["control"]["frames"]})
    by_scene = [{"scene_id": scene, **{arm: float(np.mean([
        r["pu_gain_db"] for r in reports[arm]["frames"] if r["scene_id"] == scene and r["condition"] == "real"]))
        for arm in pair}} for scene in scenes]
    elapsed = sum(s["elapsed_seconds_after_preflight"] for s in states.values())
    size = footprint(root)
    if elapsed > 2 * recipe["arm_seconds"] or size >= recipe["storage_stop_bytes"]:
        raise ValueError("combined budget exceeded")
    result = {"scope": recipe["scope"] + "; verified early comparison, no promotion",
              "recipe_sha256": sha256(recipe_path), "approval_sha256": sha256(approval_path),
              "manifest_sha256": recipe["manifest_sha256"], "code_sha256": recipe["code_sha256"],
              "verification_code_sha256": sha256(Path(__file__)), "checkpoints": checkpoints,
              "matched_draws": len(traces["control"]), "unique_record_indices": len({r["record_index"] for r in traces["control"]}),
              "baseline_counts": dict(Counter(r["baseline_curve"] for r in traces["control"])),
              "media_files_verified_per_arm": media_count,
              "elapsed_seconds_after_preflight": {arm: s["elapsed_seconds_after_preflight"] for arm, s in states.items()},
              "combined_seconds_after_preflight": elapsed, "artifact_bytes_before_summary": size,
              "metrics": {arm: s["metrics"] for arm, s in states.items()},
              "per_ev": {arm: s["per_ev"] for arm, s in states.items()}, "real_per_scene": by_scene,
              "training_diagnostics": gradients, "effect": (repeatability_comparison(root, recipe) if is_repeat else
                  push_baseline_comparison(states) if is_push else
                  region_comparison(states) if is_region else preservation_comparison(states)),
              "release_authorized": False,
              "limitations": ["one seed, 200 steps, already inspected single-title validation",
                               "picker baseline only, no shipping clip-fill/competitor/temporal evaluation",
                               "training pixel counts are correlated sample draws, not independent trials",
                               "no convergence or commercial release claim"]}
    if is_push:
        result["baseline_counts_by_arm"] = {arm: dict(Counter(r["baseline_curve"] for r in traces[arm])) for arm in pair}
        result["changed_baselines"] = changed
        result["luminance_breakdown"] = luminance_effect(reports, source_records, pair)
    text = json.dumps(result, indent=2, allow_nan=False)
    with (root / "verified_summary.json").open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(json.dumps({k: result[k] for k in ("matched_draws", "combined_seconds_after_preflight", "effect")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--approval", required=True, type=Path)
    args = parser.parse_args()
    verify(args.recipe, args.approval)
