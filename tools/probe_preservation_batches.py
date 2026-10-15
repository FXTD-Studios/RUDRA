"""Verify the frozen gradient diagnosis on recorded batches at training precision."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline.commercial_preflight import sha256
from rudra.sdr2hdr import SDR2HDRNet
from tools.diagnose_preservation_tradeoff import parameter_probe
from training.routing_replay import augmentation_rng, fingerprint
from training.sdr2hdr_dataset import SDRHDRDataset, read_jsonl


def probe(root, recipe_path, out, seconds=180):
    if out.exists() or not out.resolve().is_relative_to((REPO / "reports").resolve()):
        raise ValueError("fresh output directory under reports required")
    if not torch.cuda.is_available():
        raise ValueError("CUDA required to reproduce the original training precision")
    started = time.monotonic()
    recipe = json.loads(recipe_path.read_text())
    verified = json.loads((root / "verified_summary.json").read_text())
    if verified["recipe_sha256"] != sha256(recipe_path):
        raise ValueError("verified recipe mismatch")
    pins = {REPO / k: v for k, v in recipe["code_sha256"].items()}
    pins[Path(recipe["manifest"])] = recipe["manifest_sha256"]
    config = json.loads((root / "preclamp/config.json").read_text())
    expected = {"shadow_chroma_weight": .15, "shadow_smoothness_weight": .02,
                "region_focus": 0., "crushed_hold": 1., "curve_head": False,
                "source_curve": False, "batch_size": 4, "grad_accum": 1, "max_grad_norm": 1.}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported frozen recipe")
    torch.set_num_threads(2)
    models = {}
    for arm in ("initial", "control", "preclamp"):
        path = root / ("preclamp" if arm == "initial" else arm) / ("step_0000000.pt" if arm == "initial" else "step_0000200.pt")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        actual = sha256(path)
        if arm == "initial":
            if fingerprint(saved["model"]) != verified["checkpoints"]["preclamp"]["initial_model_sha256"]:
                raise ValueError("initial tensor mismatch")
        elif actual != verified["checkpoints"][arm]["step_200_sha256"]:
            raise ValueError("checkpoint mismatch")
        pins[path] = actual
        model = SDR2HDRNet.from_config(saved["config"]).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved["model"], strict=True)
        models[arm] = model
    for path, expected_hash in pins.items():
        if sha256(path) != expected_hash:
            raise ValueError(f"changed frozen input: {path}")
    dataset = SDRHDRDataset(recipe["manifest"], source_curve_dropout=0, **{k: config[k] for k in (
        "crop_size", "augmentation_strength", "degradation_probability", "exposure_push", "exposure_push_max_ev", "baseline_dropout")})
    trace = read_jsonl(root / "preclamp_samples.jsonl")
    out.mkdir(parents=True)
    first_step_check = {}
    with (out / "batches.jsonl").open("x", encoding="utf-8") as journal:
        # Complete historical batches containing the previously stratified crops.
        for start in (0, 4, 8, 32, 40, 44):
            samples = []
            for draw in trace[start:start + 4]:
                with augmentation_rng(draw["augmentation_seed"]):
                    sample = dataset[draw["record_index"]]
                if fingerprint({k: v for k, v in sample.items() if k != "baseline_curve"}) != draw["sample_sha256"] or int(sample["baseline_curve"]) != draw["baseline_curve"]:
                    raise ValueError("batch replay mismatch")
                if torch.isfinite(sample["ceiling"]).any():
                    raise ValueError("uncensored draws required")
                samples.append(sample)
            sdr, target = (torch.stack([s[k] for s in samples]).cuda() for k in ("sdr", "hdr"))
            curves = torch.tensor([int(s["baseline_curve"]) for s in samples], device="cuda")
            for arm, model in models.items():
                if time.monotonic() - started > seconds:
                    raise TimeoutError("batch diagnostic budget exceeded")
                measured = parameter_probe(model, sdr, target, curves, amp=True)
                journal.write(json.dumps({"arm": arm, "first_draw": start,
                                          "probe": measured}, allow_nan=False) + "\n")
                journal.flush()
                if arm == "initial" and start == 0:
                    for prior_arm, term in (("control", "post_total"), ("preclamp", "pre_total")):
                        history = next(r for r in read_jsonl(root / prior_arm / "train.jsonl") if r.get("step") == 1 and "total" in r)
                        result = measured["terms"][term]
                        first_step_check[prior_arm] = {"loss_difference": abs(result["value"] - history["total"]),
                            "gradient_norm_difference": abs(result["gradient_norm"] - history["gradient_norm_before_clip"])}
                        if not math.isclose(result["value"], history["total"], rel_tol=1e-5, abs_tol=1e-7) or not math.isclose(result["gradient_norm"], history["gradient_norm_before_clip"], rel_tol=1e-5, abs_tol=1e-6):
                            raise ValueError("original first-step loss/gradient not reproduced")
            print(f"verified BF16 batch at draws {start}-{start+3}", flush=True)
    for path, expected_hash in pins.items():
        if sha256(path) != expected_hash:
            raise ValueError("frozen input changed during diagnosis")
    result = {"elapsed_seconds": time.monotonic() - started, "optimizer_steps": 0,
              "batches": 6, "verified_draws": 24, "probes": 18,
              "first_training_step_parity": first_step_check,
              "input_hashes": {str(p): h for p, h in pins.items()},
              "algorithm_sha256": {p: sha256(Path(p)) for p in (
                  "tools/probe_preservation_batches.py", "tools/diagnose_preservation_tradeoff.py")},
              "journal_sha256": sha256(out / "batches.jsonl"),
              "limitations": ["six selected early batches, no population inference",
                              "weights frozen at initialization or step 200; no intermediate trajectory or AdamW updates"]}
    with (out / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("elapsed_seconds", "first_training_step_parity")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    args = parser.parse_args()
    probe(args.root, args.recipe, args.out, args.seconds)
