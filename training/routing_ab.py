"""Run an explicitly approved, sequential scratch routing, preservation or push-baseline A/B.

No source media cache or native changes. Each dedicated child owns one arm;
the parent enforces wall time and shared artifact limits, including during a
hung training call. A pending draft cannot launch either arm.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.commercial_preflight import sha256
from training.bounded_pilot import footprint

CODE_FILES = ("training/routing_ab.py", "training/routing_replay.py",
              "training/train_sdr2hdr.py", "training/sdr2hdr_dataset.py",
              "training/bench_val.py", "training/bounded_pilot.py",
              "training/preservation.py",
              "pipeline/commercial_preflight.py", "pipeline/hdr_io.py",
              "rudra/sdr2hdr.py", "rudra/source_baseline.py")


class ArmComplete(Exception):
    pass


PUSH_SCOPE = "internal-push-baseline-ab"
SCOPES = ("internal-routing-ab", "internal-preservation-ab", "internal-region-focus-ab",
          "internal-repeatability", PUSH_SCOPE)


def preservation(recipe):
    return recipe.get("scope") in ("internal-preservation-ab", "internal-region-focus-ab", "internal-repeatability",
                                   PUSH_SCOPE)


def push_baseline_ab(recipe):
    """Single factor: pushed renders keep their own ACES baseline (render) vs picker routing."""
    return recipe.get("scope") == PUSH_SCOPE


def strict_pair(recipe):
    """Scopes run on the verified strict-deterministic, every-step-logged recipe."""
    return repeatability(recipe) or push_baseline_ab(recipe)


def repeatability(recipe):
    return recipe.get("scope") == "internal-repeatability"


def region_focus(recipe):
    return recipe.get("scope") == "internal-region-focus-ab"


def arms(recipe):
    if repeatability(recipe):
        return ("control", "repeat")
    if region_focus(recipe):
        return ("control", "region")
    if push_baseline_ab(recipe):
        return ("control", "render")
    return ("control", "preclamp") if preservation(recipe) else ("control", "corrected")


def repeatability_comparison(root, recipe):
    """Exact repeatability verdict; never a model-quality promotion gate."""
    import torch
    from training.routing_replay import fingerprint
    root = Path(root)
    records, checkpoints = [], []
    for arm in arms(recipe):
        rows = [json.loads(line) for line in (root / arm / "train.jsonl").read_text().splitlines()]
        if [r["step"] for r in rows if "total" in r] != list(range(1, recipe["arm_steps"] + 1)):
            raise ValueError("incomplete every-step repeatability logs")
        records.append(rows)
        checkpoints.append(torch.load(root / arm / f"step_{recipe['arm_steps']:07d}.pt",
                                      map_location="cpu", weights_only=False))
    configs = [{k:v for k,v in c["config"].items() if k not in ("output_dir", "preservation_ab_arm")}
               for c in checkpoints]
    if configs[0] != configs[1] or not configs[0].get("runtime", {}).get("deterministic_algorithms"):
        raise ValueError("repeatability configs/runtime differ or strict mode absent")
    hashes = [fingerprint(c["model"]) for c in checkpoints]
    initial = [fingerprint(torch.load(root / arm / "step_0000000.pt", map_location="cpu",
                                     weights_only=False)["model"]) for arm in arms(recipe)]
    inputs = [(root / f"{arm}_samples.jsonl").read_bytes() for arm in arms(recipe)]
    if initial[0] != initial[1] or inputs[0] != inputs[1]:
        raise ValueError("repeatability initialization or input mismatch")
    reports = [[json.loads(line) for line in (root / arm / "bench_val.jsonl").read_text().splitlines()]
               for arm in arms(recipe)]
    checks = {"training_and_eval_logs_equal": records[0] == records[1],
              "final_weights_equal": hashes[0] == hashes[1],
              "benchmark_reports_equal": reports[0] == reports[1]}
    return {**checks, "repeatability_pass": all(checks.values()), "model_sha256": hashes,
            "production_pass": False, "release_authorized": False}


def _paired_deltas(control, candidate, label):
    a, b = ({r["ev"]: r for r in s["per_ev"]} for s in (control, candidate))
    if set(a) != {1., 2.} or a.keys() != b.keys():
        raise ValueError(f"paired {label} EV coverage mismatch")
    for ev in a:
        for key in ("clip_frames", "crushed_frames", "pu_frames"):
            if a[ev][key] <= 0 or a[ev][key] != b[ev][key]:
                raise ValueError(f"paired {label} coverage mismatch")
        for row in (a[ev], b[ev]):
            if any(row.get(k) is None or not math.isfinite(float(row[k]))
                   for k in ("clip_gain_stops", "crushed_gain_stops", "pu_gain_db")):
                raise ValueError(f"invalid {label} scores")
    real_delta = candidate["metrics"]["bench_real_gain_db"] - control["metrics"]["bench_real_gain_db"]
    deltas = [{"ev": ev, "clipped_delta_stops": b[ev]["clip_gain_stops"] - a[ev]["clip_gain_stops"],
               "crushed_delta_stops": b[ev]["crushed_gain_stops"] - a[ev]["crushed_gain_stops"],
               "pushed_pu_delta_db": b[ev]["pu_gain_db"] - a[ev]["pu_gain_db"]} for ev in sorted(a)]
    return real_delta, deltas


def push_baseline_comparison(states):
    """Pre-registered early test of the 14 Oct diagnosis; never a promotion gate.

    Supported only if real PU21 improves AND pushed crushed error is not worse
    at either EV. Clipped and pushed full-frame changes are reported as
    tradeoffs; the dark-frame split is added by the post-run verifier.
    """
    control, candidate = states["control"], states["render"]
    if control.get("changed_baselines") != 0 or not candidate.get("changed_baselines"):
        raise ValueError("push-baseline treatment was not exercised exactly in the candidate arm")
    real_delta, deltas = _paired_deltas(control, candidate, "push-baseline")
    return {"real_pu_delta_db": real_delta, "per_ev": deltas,
            "changed_baselines": candidate["changed_baselines"],
            "early_hypothesis_supported": real_delta > 0 and all(r["crushed_delta_stops"] >= 0 for r in deltas),
            "production_pass": False,
            "interpretation": "early paired effect only; relative improvement does not establish baseline superiority"}


def region_comparison(states):
    control, candidate = states["control"], states["region"]
    real_delta, deltas = _paired_deltas(control, candidate, "region-focus")
    return {"real_pu_delta_db": real_delta, "per_ev": deltas,
            "early_hypothesis_supported": real_delta > 0 and all(v >= 0 for r in deltas for k,v in r.items() if k != "ev"),
            "production_pass": False,
            "interpretation": "early paired effect only; relative improvement does not establish baseline superiority"}


def preservation_comparison(states):
    control, candidate = states["control"], states["preclamp"]
    a, b = ({r["ev"]: r for r in s["per_ev"]} for s in (control, candidate))
    if a.keys() != b.keys() or not a:
        raise ValueError("paired per-EV coverage mismatch")
    deltas = [{"ev": ev,
               "clipped_delta_stops": b[ev]["clip_gain_stops"] - a[ev]["clip_gain_stops"],
               "crushed_delta_stops": b[ev]["crushed_gain_stops"] - a[ev]["crushed_gain_stops"]}
              for ev in sorted(a)]
    return {"real_pu_delta_db": candidate["metrics"]["bench_real_gain_db"] - control["metrics"]["bench_real_gain_db"],
            "per_ev": deltas,
            "both_pushed_crushed_means_improved": all(r["crushed_delta_stops"] > 0 for r in deltas),
            "production_pass": False,
            "interpretation": "paired early diagnostic effect only; report real and clipped tradeoffs separately"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def approved_recipe(recipe_path, approval_path):
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if (approval.get("approved") is not True or not approval.get("approved_by")
            or not approval.get("approved_at") or approval.get("release_authorized") is not False
            or approval.get("recipe_sha256") != sha256(recipe_path)):
        raise ValueError("explicit approval bound to this exact recipe is required")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if (recipe.get("scope") not in SCOPES or recipe.get("release_authorized") is not False
            or not 1 <= recipe["arm_steps"] <= 200
            or not 0 < recipe["arm_seconds"] <= (900 if strict_pair(recipe) else 600)
            or recipe["storage_stop_bytes"] != 900_000_000):
        raise ValueError("invalid paired experiment scope or limits")
    root = Path(recipe["output_root"]).resolve()
    if not root.is_relative_to((REPO / "reports").resolve()) or root == (REPO / "reports").resolve():
        raise ValueError("output must be a new child of repository reports")
    if set(recipe["code_sha256"]) != set(CODE_FILES):
        raise ValueError("incomplete experiment code pins")
    for name, digest in recipe["code_sha256"].items():
        if sha256(REPO / name) != digest:
            raise ValueError(f"experiment code changed: {name}")
    for name in ("manifest", "evidence"):
        if sha256(Path(recipe[name])) != recipe[name + "_sha256"]:
            raise ValueError(f"experiment {name} changed")
    receipt = json.loads(Path(recipe["evidence"]).read_text(encoding="utf-8"))
    if (receipt.get("scope") != "internal-pilot" or receipt.get("release_authorized") is not False
            or not receipt.get("reviewed_by") or not receipt.get("reviewed_at")
            or any(s.get("weight_distribution") is not False for s in receipt.get("sources", {}).values())):
        raise ValueError("reviewed internal-only evidence required")
    return recipe


def arm_args(recipe, arm):
    if arm not in arms(recipe):
        raise ValueError("unknown experiment arm")
    from training import train_sdr2hdr as trainer
    previous = sys.argv
    try:
        sys.argv = ["train_sdr2hdr.py", *recipe["trainer_args"],
                    "--manifest", recipe["manifest"], "--commercial-evidence", recipe["evidence"],
                    "--output-dir", str(Path(recipe["output_root"]) / arm)]
        args = trainer.parse_args()
    finally:
        sys.argv = previous
    if (args.mode != "image" or args.workers != 0 or args.steps != 1000
            or args.eval_every != recipe["arm_steps"] or args.save_every != recipe["arm_steps"]
            or args.batch_size != 4 or args.grad_accum != 1 or not args.baseline_source
            or args.resume or args.init_checkpoint or args.image_checkpoint
            or args.best_metric != "bench_gain"):
        raise ValueError("requires scratch picker image training, zero workers, batch 4, 1000-step schedule")
    if (args.crushed_hold_domain != "postclamp" or args.preservation_diagnostics or args.bench_push_pu
            or args.push_baseline != "picker"):
        raise ValueError("experiment controls the hold domain, push baseline and diagnostic flag")
    if preservation(recipe):
        if (args.crushed_hold != 1.0 or args.region_focus != 0.0
                or args.curve_head or args.source_curve or args.gate_conditioning
                or args.exposure_push != .5 or args.exposure_push_max_ev != 2.0
                or args.baseline_dropout != .5 or args.degradation_probability != .65
                or args.shadow_chroma_weight != .15 or args.shadow_smoothness_weight != .02
                or not args.scene_balanced_sampling or args.real_corpus_ev != 0.0):
            raise ValueError("preservation comparison must retain the frozen objective and routing recipe")
        args.crushed_hold_domain = "preclamp" if arm == "preclamp" or region_focus(recipe) or strict_pair(recipe) else "postclamp"
        if region_focus(recipe) or strict_pair(recipe):
            if args.bench_val_push != "1,2":
                raise ValueError("region-focus comparison requires +1/+2 EV")
            args.region_focus = .5 if arm == "region" else 0.
            args.bench_push_pu = True
        if strict_pair(recipe) and (not args.deterministic or args.log_every != 1
                or args.crop_size != 256 or args.base_channels != 32):
            raise ValueError("repeatability requires deterministic full-size training and every-step logs")
        if push_baseline_ab(recipe):
            args.push_baseline = "render" if arm == "render" else "picker"
        args.preservation_diagnostics = True
        args.preservation_ab_arm = arm
        args.preservation_ab_stop_step = recipe["arm_steps"]
    else:
        args.routing_ab_arm = arm
        args.routing_ab_stop_step = recipe["arm_steps"]
    return args


def check_limits(root, started, seconds, storage_limit, now=None):
    if started is not None and (time.monotonic() if now is None else now) - started >= seconds:
        raise TimeoutError("experiment arm exceeded its post-preflight time cap")
    if footprint(root) >= storage_limit:
        raise RuntimeError("shared experiment artifact limit reached")


def run_arm(recipe, arm):
    import torch
    from training import train_sdr2hdr as trainer
    from training.routing_replay import ReplayDataset, fingerprint
    if arm not in arms(recipe):
        raise ValueError("unknown arm")
    args = arm_args(recipe, arm)
    root, out = Path(recipe["output_root"]), Path(args.output_dir)
    if out.exists():
        raise ValueError("arm output already exists")
    replay = []
    state = {"arm": arm, "state": "preflight", "last_step": 0,
             "started_monotonic": None, "release_authorized": False}
    if preservation(recipe):
        state.update(hold_domain=args.crushed_hold_domain, gradient_observations=0,
                     held_channels_seen=0, negative_held_channels_seen=0, nonzero_hold_steps=0,
                     pre_post_hold_difference_steps=0)
    if region_focus(recipe):
        state["region_focus"] = args.region_focus
    state_path = root / f"{arm}_status.json"
    original_dataset, original_save = trainer.SDRHDRDataset, trainer._atomic_save
    control_state = json.loads((root / "control_status.json").read_text()) if arm != "control" else None
    if control_state and control_state.get("state") != "completed":
        raise ValueError("control must complete before corrected arm")

    def limits():
        check_limits(root, state["started_monotonic"], recipe["arm_seconds"], recipe["storage_stop_bytes"])

    done = threading.Event()

    def guard():
        while not done.wait(0.25):
            try:
                limits()
            except Exception as exc:
                state.update(state="stopped", reason=str(exc))
                try:
                    write_json(state_path, state)
                finally:
                    os._exit(124)  # Dedicated child; workers=0, no loader children.

    def dataset(*a, **kw):
        data = original_dataset(*a, **kw)
        if data.augment:
            from rudra.source_baseline import baseline_curve_index
            change = ((baseline_curve_index("bt1886"), baseline_curve_index("aces"))
                      if push_baseline_ab(recipe) and control_state else None)
            data = ReplayDataset(data, "corrected" if control_state else "control", args.seed,
                                 root / f"{arm}_samples.jsonl",
                                 root / "control_samples.jsonl" if control_state else None,
                                 matched_baselines=preservation(recipe), allowed_baseline_change=change)
            replay.append(data)
        return data

    def save(payload, path):
        limits()
        if not all(torch.isfinite(t).all() for t in payload["model"].values()):
            raise ValueError("nonfinite checkpoint")
        if payload["step"] == 0:
            initial = fingerprint(payload["model"])
            if control_state and initial != control_state["initial_model_sha256"]:
                raise ValueError("arms have different starting weights")
            state["initial_model_sha256"] = initial
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        size = buffer.tell()
        if footprint(root) + size * (2 if payload["step"] == 0 else 1) >= recipe["storage_stop_bytes"]:
            raise RuntimeError("checkpoint would exceed shared artifact limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(buffer.getbuffer())
        os.replace(temporary, path)
        if payload["step"] == 0:
            (out / "step_0000000.pt").write_bytes(buffer.getbuffer())

    def observe(event, step, values):
        state["last_step"] = step
        if event == "ready":
            state["started_monotonic"] = time.monotonic()
            state["state"] = "running"
            write_json(state_path, state)
        limits()
        if event == "eval" or (event == "gradients" and step % 10 == 0):
            # Wall-clock milestones only, kept out of numerical comparison logs.
            with (root / f"{arm}_timing.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": event, "step": step,
                    "elapsed_seconds_after_preflight": time.monotonic() - state["started_monotonic"]}) + "\n")
        if event == "loss" and not all(torch.isfinite(v).all() for v in values.values()):
            raise ValueError("nonfinite training values")
        if event == "gradients" and preservation(recipe):
            if not all(torch.isfinite(v).all() for v in values.values()):
                raise ValueError("nonfinite preservation gradients/diagnostics")
            state["gradient_observations"] += 1
            state["held_channels_seen"] += int(values["held_channels"])
            state["negative_held_channels_seen"] += int(values["negative_held_channels"])
            state["nonzero_hold_steps"] += int(float(values[f"hold_{args.crushed_hold_domain}"]) > 0)
            state["pre_post_hold_difference_steps"] += int(abs(float(values["hold_preclamp"] - values["hold_postclamp"])) > 1e-7)
        if event == "eval":
            if not all(math.isfinite(float(v)) for v in values.values()):
                raise ValueError("nonfinite evaluation")
            for key in ("bench_real_frames", "bench_clip_frames", "bench_crushed_frames"):
                if values.get(key, 0) <= 0:
                    raise ValueError(f"missing evaluation coverage: {key}")
            report = json.loads((out / "bench_val.jsonl").read_text().splitlines()[-1])
            if (not report["per_ev"] or any(r["clip_frames"] <= 0 or r["crushed_frames"] <= 0
                                           for r in report["per_ev"])):
                raise ValueError("missing per-EV clipped/crushed coverage")
            if preservation(recipe):
                configured_evs = [float(e) for e in args.bench_val_push.split(",") if e]
                if ([r["ev"] for r in report["per_ev"]] != configured_evs
                        or any(r.get(k) is None or not math.isfinite(float(r[k]))
                               for r in report["per_ev"] for k in ("clip_gain_stops", "crushed_gain_stops"))):
                    raise ValueError("missing/invalid configured per-EV preservation scores")
            state["per_ev"] = report["per_ev"]
            if (region_focus(recipe) or strict_pair(recipe)) and any(r.get("pu_frames", 0) != values["bench_real_frames"]
                    or r.get("pu_gain_db") is None or not math.isfinite(float(r["pu_gain_db"]))
                    for r in report["per_ev"]):
                raise ValueError("missing/invalid pushed full-frame PU21 coverage")
            state["metrics"] = values
            write_json(state_path, state)
            if step == recipe["arm_steps"]:
                if len(replay) != 1:
                    raise ValueError("exactly one training replay is required")
                replay[0].finish(step * args.batch_size)
                state["sample_count"] = replay[0].count
                state["changed_baselines"] = replay[0].changed_baselines
                if control_state and not preservation(recipe) and not replay[0].changed_baselines:
                    raise ValueError("corrected routing was never exercised")
                if push_baseline_ab(recipe) and bool(control_state) != bool(replay[0].changed_baselines):
                    raise ValueError("push-baseline treatment must change baselines in the candidate arm only")
                if preservation(recipe) and (state["gradient_observations"] != step
                                             or state["held_channels_seen"] <= 0
                                             or state["nonzero_hold_steps"] <= 0):
                    raise ValueError("preservation objective/gradient coverage missing")
                raise ArmComplete()

    trainer.SDRHDRDataset, trainer._atomic_save = dataset, save
    write_json(state_path, state)
    watchdog = threading.Thread(target=guard, daemon=True)
    watchdog.start()
    try:
        trainer.train(args, observer=observe)
        raise RuntimeError("trainer returned without the bounded stop")
    except ArmComplete:
        state["state"] = "completed"
        return_code = 0
    except BaseException as exc:
        state.update(state="stopped", reason=f"{type(exc).__name__}: {exc}")
        return_code = 1
    finally:
        done.set()
        watchdog.join(timeout=1)
        trainer.SDRHDRDataset, trainer._atomic_save = original_dataset, original_save
        for data in replay:
            data.close()
        state["elapsed_seconds_after_preflight"] = (
            time.monotonic() - state["started_monotonic"] if state["started_monotonic"] is not None else None)
        write_json(state_path, state)
    return return_code


def run(recipe_path, approval_path):
    recipe = approved_recipe(recipe_path, approval_path)
    # Validate all arguments before creating files or launching a child.
    for arm in arms(recipe):
        arm_args(recipe, arm)
    root = Path(recipe["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    try:
        for arm in arms(recipe):
            with (root / f"{arm}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "--recipe", str(recipe_path.resolve()),
                     "--approval", str(approval_path.resolve()), "--worker", arm],
                    cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                try:
                    while child.poll() is None:
                        status_path = root / f"{arm}_status.json"
                        state = json.loads(status_path.read_text()) if status_path.exists() else {}
                        check_limits(root, state.get("started_monotonic"), recipe["arm_seconds"], recipe["storage_stop_bytes"])
                        time.sleep(0.25)
                    if child.returncode:
                        raise RuntimeError(f"{arm} failed; see its log/status")
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait()
        states = {arm: json.loads((root / f"{arm}_status.json").read_text()) for arm in arms(recipe)}
        if any(s["state"] != "completed" for s in states.values()):
            raise RuntimeError("both arms must complete")
        write_json(root / "comparison.json", {"arms": states, "release_authorized": False,
                   "scope": recipe["scope"],
                   **({"repeatability": repeatability_comparison(root, recipe)} if repeatability(recipe) else
                      {"effect": push_baseline_comparison(states)} if push_baseline_ab(recipe) else
                      {"effect": region_comparison(states)} if region_focus(recipe) else
                      {"effect": preservation_comparison(states)} if preservation(recipe) else {}),
                   "interpretation": "early paired comparison only; no quality promotion or automatic continuation"})
        return 0
    except BaseException as exc:
        write_json(root / "stopped.json", {"reason": f"{type(exc).__name__}: {exc}", "release_authorized": False})
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--approval", required=True, type=Path)
    parser.add_argument("--worker", choices=("control", "corrected", "preclamp", "region", "repeat", "render"), help=argparse.SUPPRESS)
    a = parser.parse_args()
    if a.worker:
        raise SystemExit(run_arm(approved_recipe(a.recipe, a.approval), a.worker))
    raise SystemExit(run(a.recipe, a.approval))
