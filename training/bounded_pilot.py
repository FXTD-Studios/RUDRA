"""Run an approved scratch pilot with a smoke gate and hard local limits.

Uses the existing trainer and objective. Internal-only; no resume or release.
Workers must be zero so the watchdog never leaves data-loader child processes.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class PilotStop(RuntimeError):
    pass


def smoke_gate(metrics):
    if not all(math.isfinite(float(v)) for v in metrics.values()):
        raise PilotStop("nonfinite smoke metric")
    for key in ("bench_real_frames", "bench_clip_frames", "bench_crushed_frames"):
        if metrics.get(key, 0) <= 0:
            raise PilotStop(f"smoke lacks required coverage: {key}")
    if metrics.get("bench_real_gain_db", -math.inf) <= 0:
        raise PilotStop("smoke did not improve sampled real PU21")
    for key in ("bench_clip_gain_stops", "bench_crushed_gain_stops"):
        if metrics.get(key, -math.inf) < -1e-6:
            raise PilotStop(f"smoke regressed sampled region: {key}")


class Budget:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = None
        self.extended = None

    def start(self):
        self.started = self.clock()

    def extend(self):
        self.check()
        self.extended = self.clock()

    def check(self):
        if self.started is None:
            return
        now = self.clock()
        if now - self.started >= 5400:
            raise PilotStop("90-minute total budget exhausted")
        if self.extended is None and now - self.started >= 1800:
            raise PilotStop("30-minute smoke budget exhausted")
        if self.extended is not None and now - self.extended >= 3600:
            raise PilotStop("60-minute continuation budget exhausted")


def footprint(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def run(recipe):
    import torch
    from training import train_sdr2hdr as trainer

    argv = recipe["trainer_args"]
    previous = sys.argv
    try:
        sys.argv = ["train_sdr2hdr.py", *argv]
        args = trainer.parse_args()
    finally:
        sys.argv = previous
    if (args.mode != "image" or args.workers != 0 or args.steps != 1000
            or args.eval_every != 200 or args.save_every != 200
            or args.resume or args.init_checkpoint or args.image_checkpoint
            or not args.commercial_evidence):
        raise ValueError("pilot requires reviewed scratch image run, zero workers, 1000 steps, eval/save every 200")
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise ValueError("pilot output must be a new directory")
    budget = Budget()
    done = threading.Event()
    last_step = 0

    def status(state, reason=None):
        if out.exists():
            elapsed = budget.clock() - budget.started if budget.started is not None else 0
            (out / "pilot_status.json").write_text(json.dumps({
                "state": state, "reason": reason, "last_step": last_step,
                "elapsed_seconds_after_preflight": elapsed,
                "continuation_entered": budget.extended is not None,
                "release_authorized": False,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            }, indent=2), encoding="utf-8")

    def guard():
        while not done.wait(0.25):
            try:
                budget.check()
                if footprint(out) >= 900_000_000:
                    raise PilotStop("storage safety threshold reached (900 MB; approval cap 1 GB)")
            except PilotStop as exc:
                status("stopped", str(exc))
                print(f"[pilot-stop] {exc}", flush=True)
                os._exit(124)  # This dedicated process owns no loader children.

    def save(payload, path):
        budget.check()
        if not all(torch.isfinite(t).all() for t in payload["model"].values()):
            raise PilotStop("nonfinite checkpoint parameters")
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        size = buffer.tell()
        if footprint(out) + size * (2 if payload["step"] == 0 else 1) >= 900_000_000:
            raise PilotStop("checkpoint would exceed storage safety threshold")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(buffer.getbuffer())
        os.replace(temporary, path)
        if payload["step"] == 0:
            (out / "step_0000000.pt").write_bytes(buffer.getbuffer())

    def observe(event, step, values):
        nonlocal last_step
        last_step = step
        if event == "ready":
            budget.start()
            print("[pilot] media preflight passed; 30-minute smoke timer started", flush=True)
        budget.check()
        if event == "loss":
            if not all(torch.isfinite(v).all() for v in values.values()):
                raise PilotStop(f"nonfinite training value at step {step}")
        if event == "eval":
            if not all(math.isfinite(float(v)) for v in values.values()):
                raise PilotStop(f"nonfinite evaluation at step {step}")
            if step == 200:
                smoke_gate(values)
                budget.extend()
                print("[pilot] smoke gate passed; up to 60 additional minutes authorized", flush=True)
            status("running")

    original = trainer._atomic_save
    trainer._atomic_save = save
    watchdog = threading.Thread(target=guard, daemon=True)
    watchdog.start()
    try:
        result = trainer.train(args, observer=observe)
        status("completed")
        print(f"[pilot] completed: {result}; internal-only, no promotion", flush=True)
        return 0
    except Exception as exc:
        status("stopped", f"{type(exc).__name__}: {exc}")
        print(f"[pilot-stop] {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        done.set()
        trainer._atomic_save = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    a = parser.parse_args()
    raise SystemExit(run(json.loads(a.recipe.read_text(encoding="utf-8"))))
