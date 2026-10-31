"""Resume-capable image training for FUTURE runs: exact continuation or explicit warm start.

`train_sdr2hdr.train()` (and every launcher built on it) is scratch-only: its
numbered checkpoints carry model/optimizer/scheduler but no RNG state, no data
position and no selection history, so a crash means starting again. This module
is a separate workflow; it leaves that trainer and all existing runs untouched.

What a resume state holds (schema ``rudra-resume-v1``)
    model, optimizer, scheduler, step, best score, best weights and best step,
    the trailing selection-score history, the AMP configuration (bf16 autocast,
    no GradScaler) and a GradScaler state if one is ever used, Python / NumPy /
    torch CPU / CUDA RNG states, the train sampler's epoch-start generator state
    plus the batches already consumed in that epoch, the full trainer config,
    dataset identity (manifest SHA256, commercial evidence SHA256 and provenance),
    SHA256 of every code file that shapes training or selection, torch version,
    and the run's lineage (run id, segments, which file each segment resumed from).

How it is written
    every numbered save (``step_NNNNNNN.pt``) is a full state; between them,
    ``--resume-save-every`` and graceful stops write one of two ping-pong slots
    under ``resume/``. Each file is written to a temp file, fsynced, reloaded
    and checked, then renamed; its ``.json`` sidecar (SHA256, step, run id) is
    written after it the same way. A half-written file or a stale sidecar fails
    verification and the next-newest valid state is used, so a valid previous
    copy always exists. Nothing is deleted: superseded logs and best.pt are moved
    aside with an ``orphaned`` suffix.

Exact continuation (``--resume``)
    same run directory, same run id, identical invariant config, identical code
    hashes, identical dataset/provenance (commercial evidence is fully
    re-validated, media rehash included), same torch version, workers=0.
    On CPU with ``--deterministic`` an interrupted-and-resumed run is bitwise
    identical to an uninterrupted one (tests). On CUDA the restored state is
    exact but kernels are not bitwise deterministic, so the continuation is as
    close as two uninterrupted CUDA runs are to each other -- not bitwise.

Warm start (``--warm-start-from``)
    model weights only, fresh optimizer/schedule/RNG, NEW run, lineage records
    the parent. Not a continuation. Refused for commercial runs: the commercial
    preflight has no ancestor clearance (pipeline/commercial_preflight.py).

Checkpoints written by train_sdr2hdr (including fxtd_native1080_hdr_run02) have
no resume schema: they are refused for exact resume and accepted only as a
warm-start parent for research runs.

    python training/resumable_training.py <train_sdr2hdr args> [--resume]
        [--resume-save-every N] [--max-hours H] [--recover-stale-lock]
    python training/resumable_training.py inspect <checkpoint.pt>
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import signal
import socket
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = "rudra-resume-v1"
EXIT_STOPPED_RESUMABLE = 3

# Files whose bytes decide what one training step or one selection does.
CODE_FILES = (
    "rudra/preservation_head.py",
    "rudra/reconstruction_policy.py",
    "training/dequantization.py", "training/reconstruction_gradient.py", "training/signed_shadow.py", "training/contract_checks.py", "pipeline/run05_corpus.py",
    "pipeline/sdr_render.py", "rudra/delivery/colorspace.py",
    "training/run05_objective.py",
    "training/train_sdr2hdr.py", "training/resumable_training.py", "training/sdr2hdr_dataset.py",
    "training/preservation.py", "training/bench_val.py", "rudra/sdr2hdr.py",
    "rudra/source_baseline.py", "rudra/radiometry.py", "pipeline/hdr_io.py",
    "pipeline/commercial_preflight.py", "training/dark_experiment.py",
)
# Trainer options that may differ between segments without changing the math.
MUTABLE_OPTIONS = {"log_every", "output_dir", "resume", "init_checkpoint", "reset_best", "device"}


class ResumeError(RuntimeError):
    """Refusal to continue; the message says what differs and what to do instead."""


# ----------------------------------------------------------------- utilities
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hashes() -> dict:
    """Line-ending independent, so a CRLF re-checkout on Windows is the same code."""
    return {name: hashlib.sha256((REPO / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in CODE_FILES if (REPO / name).is_file()}


def _fsync_write(path: Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def atomic_write_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    _fsync_write(tmp, json.dumps(value, indent=2, default=str).encode("utf-8"))
    os.replace(tmp, path)


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _move_aside(path: Path, label: str) -> Path | None:
    """Rename instead of delete (the repo mount cannot unlink; evidence is kept)."""
    if not path.exists():
        return None
    target = path.with_name(f"{path.stem}.{label}_{_utc_stamp()}{path.suffix}")
    n = 0
    while target.exists():
        n += 1
        target = path.with_name(f"{path.stem}.{label}_{_utc_stamp()}_{n}{path.suffix}")
    os.replace(path, target)
    return target


# ------------------------------------------------------------------ RNG state
def capture_rng() -> dict:
    py = random.getstate()
    np_state = np.random.get_state()
    return {
        "python": [py[0], list(py[1]), py[2]],
        "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict, skip_cuda: bool = False) -> None:
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.random.set_rng_state(state["torch_cpu"])
    cuda = [] if skip_cuda else (state.get("torch_cuda") or [])
    if cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(cuda):
            raise ResumeError("saved CUDA RNG states do not match the visible GPUs; exact resume needs the same device count")
        torch.cuda.set_rng_state_all(cuda)


# ------------------------------------------------------------------- sampling
class ResumableSampler(Sampler):
    """Index stream identical to WeightedRandomSampler(weights, n, True, generator),
    plus the state needed to continue an epoch part-way through.

    Without weights it draws a randperm from its own generator (record-uniform).
    That differs from the legacy RandomSampler draw, which is seeded from the
    global RNG, but it is exact and resumable within this workflow.
    """

    def __init__(self, num_samples: int, generator: torch.Generator, weights: torch.Tensor | None = None):
        self.num_samples = int(num_samples)
        self.generator = generator
        self.weights = weights
        self.epoch = 0
        self.epoch_start_state: torch.Tensor | None = None
        self._skip = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        self.epoch_start_state = self.generator.get_state()
        if self.weights is not None:
            order = torch.multinomial(self.weights, self.num_samples, True, generator=self.generator).tolist()
        else:
            order = torch.randperm(self.num_samples, generator=self.generator).tolist()
        skip, self._skip = self._skip, 0
        self.epoch += 1
        yield from order[skip:]

    def state(self, batches_consumed: int, batch_size: int) -> dict:
        if self.epoch_start_state is None:
            raise ResumeError("sampler has not started an epoch")
        return {"epoch": self.epoch, "epoch_start_generator": self.epoch_start_state.clone(),
                "samples_consumed": int(batches_consumed * batch_size),
                "batches_consumed": int(batches_consumed), "weighted": self.weights is not None}

    def resume(self, state: dict) -> None:
        if bool(state["weighted"]) != (self.weights is not None):
            raise ResumeError("sampling mode differs from the saved run")
        self.generator.set_state(state["epoch_start_generator"])
        self.epoch = int(state["epoch"]) - 1
        self._skip = int(state["samples_consumed"])


def build_resumable_loaders(args):
    from training import train_sdr2hdr as legacy
    train_loader, val_loader, hard_loader = legacy.build_loaders(args)
    old = train_loader.sampler
    generator = torch.Generator().manual_seed(args.seed)
    weights = getattr(old, "weights", None) if args.scene_balanced_sampling else None
    sampler = ResumableSampler(len(train_loader.dataset), generator, weights)
    loader = DataLoader(train_loader.dataset, sampler=sampler, batch_size=args.batch_size,
                        num_workers=0, pin_memory=torch.cuda.is_available(),
                        drop_last=train_loader.drop_last)
    return loader, val_loader, hard_loader


# --------------------------------------------------------------------- locks
def _process_alive(pid: int, created: float | None) -> bool:
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a project dependency
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        proc = psutil.Process(pid)
        return created is None or abs(proc.create_time() - created) < 1.0
    except psutil.Error:
        return False


class RunLock:
    """O_EXCL lock with owner identity; stale locks are moved aside only on request."""

    def __init__(self, directory: Path, run_id: str | None):
        self.path = directory / "train.lock"
        self.token = uuid.uuid4().hex
        self.run_id = run_id
        self.held = False

    def _owner(self) -> dict:
        created = None
        with contextlib.suppress(Exception):
            import psutil
            created = psutil.Process().create_time()
        return {"pid": os.getpid(), "host": socket.gethostname(), "created_unix": time.time(),
                "process_create_time": created, "run_id": self.run_id, "token": self.token}

    def inspect(self) -> dict:
        if not self.path.exists():
            return {"state": "free"}
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(owner["pid"])
        except Exception:
            return {"state": "unreadable"}
        if owner.get("host") != socket.gethostname():
            return {"state": "foreign_host", "owner": owner}
        alive = _process_alive(pid, owner.get("process_create_time"))
        return {"state": "live" if alive else "stale", "owner": owner}

    def acquire(self, recover_stale: bool = False) -> None:
        info = self.inspect()
        if info["state"] == "live":
            raise ResumeError(f"run is locked by live process {info['owner']['pid']}; it is still training")
        if info["state"] == "foreign_host":
            raise ResumeError(f"lock belongs to host {info['owner'].get('host')}; cannot verify it from here")
        if info["state"] in ("stale", "unreadable"):
            if not recover_stale:
                raise ResumeError(f"{info['state']} lock at {self.path}: its owner is gone. Confirm no trainer "
                                  "is running, then rerun with --recover-stale-lock (the lock is moved, not deleted)")
            stale_dir = self.path.parent / "_stale_locks"
            stale_dir.mkdir(exist_ok=True)
            os.replace(self.path, stale_dir / f"train.lock.{_utc_stamp()}.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ResumeError("another process took the lock first") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self._owner(), f)
            f.flush()
            os.fsync(f.fileno())
        self.held = True

    def release(self) -> None:
        if not self.held:
            return
        with contextlib.suppress(Exception):
            if json.loads(self.path.read_text(encoding="utf-8")).get("token") == self.token:
                os.replace(self.path, self.path.with_name("train.lock.released"))
        self.held = False


# ---------------------------------------------------------------- state files
def save_state(payload: dict, path: Path) -> dict:
    """Write, fsync, reload-verify, rename; then the sidecar. Returns the sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    check = torch.load(tmp, map_location="cpu", weights_only=True)
    if check.get("resume_schema") != SCHEMA or check.get("step") != payload["step"]:
        raise ResumeError(f"verification of {tmp} failed after writing")
    del check
    sidecar = {"schema": SCHEMA, "file": path.name, "sha256": sha256_file(tmp), "bytes": tmp.stat().st_size,
               "step": int(payload["step"]), "run_id": payload["lineage"]["run_id"],
               "written_unix": time.time()}
    os.replace(tmp, path)
    atomic_write_json(path.with_name(path.name + ".json"), sidecar)
    return sidecar


def verify_state(path: Path, run_id: str | None = None) -> tuple[dict | None, str]:
    """(payload, "ok") for a complete, matching state file; (None, reason) otherwise."""
    sidecar_path = path.with_name(path.name + ".json")
    if not sidecar_path.is_file():
        return None, "no sidecar"
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None, "unreadable sidecar"
    if sidecar.get("schema") != SCHEMA:
        return None, "sidecar schema"
    if sha256_file(path) != sidecar.get("sha256"):
        return None, "sha256 mismatch (partial write or later modification)"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return None, f"unloadable: {type(exc).__name__}"
    if payload.get("resume_schema") != SCHEMA or payload.get("step") != sidecar.get("step"):
        return None, "payload/sidecar mismatch"
    if run_id is not None and payload["lineage"]["run_id"] != run_id:
        return None, "different run"
    return payload, "ok"


def classify_checkpoint(path: Path) -> dict:
    """What a checkpoint can honestly be used for."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    keys = sorted(payload) if isinstance(payload, dict) else []
    if isinstance(payload, dict) and payload.get("resume_schema") == SCHEMA:
        state, reason = verify_state(Path(path))
        kind = "exact_resume_state" if state is not None else f"damaged_resume_state ({reason})"
        uses = ["exact resume within its run", "warm start"] if state is not None else []
    elif isinstance(payload, dict) and {"optimizer", "scheduler"} <= set(keys):
        kind, uses = "legacy_optimizer_state", ["warm start (weights only)"]
    elif isinstance(payload, dict) and "model" in payload:
        kind, uses = "weights_only", ["warm start (weights only)", "evaluation"]
    else:
        kind, uses = "unknown", []
    missing = [] if kind == "exact_resume_state" else [
        "RNG states", "sampler epoch/position", "selection history", "best weights", "code/dataset identity",
        "lineage"]
    return {"path": str(path), "sha256": sha256_file(Path(path)), "kind": kind, "step": payload.get("step"),
            "keys": keys, "supported_uses": uses, "missing_for_exact_resume": missing}


def find_latest_state(output_dir: Path, run_id: str) -> tuple[Path, dict, list[dict]]:
    candidates = sorted([*output_dir.glob("step_*.pt"), *(output_dir / "resume").glob("slot_*.pt")])
    checked, valid = [], []
    for path in candidates:
        payload, reason = verify_state(path, run_id)
        checked.append({"path": str(path), "status": reason,
                        "step": payload["step"] if payload is not None else None})
        if payload is not None:
            valid.append((payload["step"], path.name.startswith("step_"), path, payload))
    if not valid:
        raise ResumeError(f"no valid resume state in {output_dir}: {checked}")
    valid.sort(key=lambda v: (v[0], v[1]))
    step, _, path, payload = valid[-1]
    return path, payload, checked


# ------------------------------------------------------------- compatibility
def invariant_config(config: dict) -> dict:
    skip = MUTABLE_OPTIONS | {"runtime", "resumable"}
    return {k: v for k, v in config.items() if k not in skip}


def check_compatible(saved: dict, current: dict, saved_code: dict, current_code: dict,
                     saved_torch: str, allow_device_change: bool = False) -> None:
    a, b = invariant_config(saved), invariant_config(current)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if diff:
        raise ResumeError("configuration differs from the saved run: "
                          + ", ".join(f"{k}: {a.get(k)!r} -> {b.get(k)!r}" for k in diff[:8])
                          + ". Exact resume needs identical settings; start a new run (optionally --warm-start-from).")
    code_diff = sorted(k for k in set(saved_code) | set(current_code) if saved_code.get(k) != current_code.get(k))
    if code_diff:
        raise ResumeError(f"training code changed since the state was saved: {code_diff}. "
                          "Restore those files or start a new run.")
    if saved_torch.split("+")[0].rsplit(".", 1)[0] != torch.__version__.split("+")[0].rsplit(".", 1)[0]:
        raise ResumeError(f"torch {saved_torch} -> {torch.__version__}: optimizer/RNG semantics may differ")
    if not allow_device_change and saved.get("device") != current.get("device"):
        raise ResumeError(f"device {saved.get('device')} -> {current.get('device')}")


# -------------------------------------------------------------------- driver
class Control:
    """Graceful stop requests: STOP file, signal, or the wall-time budget."""

    def __init__(self, stop_file: Path, max_seconds: float | None, stop_at_step: int | None = None):
        self.stop_file = stop_file
        self.deadline = time.monotonic() + max_seconds if max_seconds else None
        self.stop_at_step = stop_at_step  # tests and controlled interruption drills
        self.reason = None

    def install_signals(self) -> None:
        def handler(signum, frame):
            self.reason = self.reason or f"signal {signum}"
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(getattr(signal, name), handler)

    def should_stop(self, step: int) -> str | None:
        if self.reason:
            return self.reason
        if self.stop_file.exists():
            self.reason = f"stop file {self.stop_file.name}"
        elif self.deadline is not None and time.monotonic() > self.deadline:
            self.reason = "wall-time budget"
        elif self.stop_at_step is not None and step >= self.stop_at_step:
            self.reason = f"requested stop at step {self.stop_at_step}"
        return self.reason


def _reconcile_log(log_path: Path, step: int) -> Path | None:
    """Keep records up to `step`; move the lost segment's later records aside."""
    if not log_path.exists():
        return None
    keep, lost = [], []
    for line in log_path.read_text(encoding="utf-8").splitlines(keepends=True):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            lost.append(line)
            continue
        (keep if int(record.get("step", -1)) <= step else lost).append(line)
    orphan = None
    if lost:
        orphan = log_path.with_name(f"train.orphaned_after_step{step}_{_utc_stamp()}.jsonl")
        _fsync_write(orphan, "".join(lost).encode("utf-8"))
    tmp = log_path.with_name(log_path.name + ".tmp")
    _fsync_write(tmp, "".join(l if l.endswith("\n") else l + "\n" for l in keep).encode("utf-8"))
    os.replace(tmp, log_path)
    return orphan


def train(args: argparse.Namespace, resume: bool = False, warm_start_from: str | None = None,
          resume_save_every: int = 0, control: Control | None = None, recover_stale_lock: bool = False,
          allow_device_change: bool = False, observer=None, experiment: dict | None = None) -> dict:
    """Run or continue an image training run. Returns a result dict; never promotes."""
    from training import train_sdr2hdr as legacy
    from training.preservation import validate_args
    from rudra.sdr2hdr import SDR2HDRNet, sdr2hdr_loss
    from training.sdr2hdr_dataset import corpus_ev_of

    # -- refuse what this workflow does not continue exactly
    if args.mode != "image":
        raise ResumeError("resumable training supports --mode image only")
    if args.resume or args.init_checkpoint or args.image_checkpoint:
        raise ResumeError("use --resume / --warm-start-from of this workflow, not the legacy flags")
    if args.workers != 0:
        raise ResumeError("exact data position needs --workers 0 (prefetching workers consume samples ahead)")
    if args.best_metric == "bench_gain" or args.bench_val_frames > 0:
        raise ResumeError("bench validation is not supported by the resumable workflow yet")
    if getattr(args, "preservation_diagnostics", False) or args.freeze_except_gate:
        raise ResumeError("preservation diagnostics / gate freezing are not supported here")
    if resume and warm_start_from:
        raise ResumeError("--resume and --warm-start-from are different things; choose one")
    commercial = bool(getattr(args, "commercial_evidence", None))
    if warm_start_from and commercial:
        raise ResumeError("commercial warm start refused: pipeline/commercial_preflight.py has no ancestor "
                          "clearance. Commercial runs start from scratch or resume themselves exactly.")
    validate_args(args)
    policy = legacy.check_reconstruction_policy_args(args)
    experiment = dict(experiment or {})
    dequant_weight = float(experiment.get('dequant_code_weight', 0))
    if not math.isfinite(dequant_weight) or dequant_weight < 0:
        raise ValueError('dequant_code_weight must be finite and nonnegative')
    reconstruction_gradient_weight = float(experiment.get('reconstruction_gradient_weight', 0))
    if not math.isfinite(reconstruction_gradient_weight) or reconstruction_gradient_weight < 0:
        raise ValueError('reconstruction_gradient_weight must be finite and nonnegative')
    if reconstruction_gradient_weight and getattr(args, 'reconstruction_policy', '') != 'observed_clip_v1':
        raise ValueError('reconstruction gradient requires observed_clip_v1')
    signed_shadow_weight = float(experiment.get('signed_shadow_weight', 0))
    if not math.isfinite(signed_shadow_weight) or signed_shadow_weight < 0:
        raise ValueError('signed_shadow_weight must be finite and nonnegative')
    if signed_shadow_weight and policy != 'observed_clip_v1':
        raise ValueError('signed shadow requires observed_clip_v1')
    if dequant_weight or reconstruction_gradient_weight or signed_shadow_weight:
        from training.dequantization import validate_training_contract
        validate_training_contract(args, experiment)
    output_dir = Path(args.output_dir)
    identity_path = output_dir / "run_identity.json"
    control = control or Control(output_dir / "STOP", None)

    # -- identity, lock and provenance
    if resume:
        if not identity_path.is_file():
            raise ResumeError(f"{output_dir} has no run_identity.json; it was not started by this workflow")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        lock = RunLock(output_dir, identity["run_id"])
        lock.acquire(recover_stale_lock)
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ResumeError(f"{output_dir} is not empty; a new run needs a new directory (or use --resume)")
        identity = None
        lock = None
    try:
        legacy.configure_determinism(getattr(args, "deterministic", False))
        provenance = None
        if commercial:
            from pipeline.commercial_preflight import training_preflight, validate
            # Fresh: the unchanged scratch preflight (empty dir, scratch only).
            # Resume: the same full evidence + media validation, then equality.
            provenance = (validate(Path(args.manifest), Path(args.commercial_evidence),
                                   progress=lambda m: print(m, flush=True))
                          if resume else training_preflight(args))
        if not resume:
            identity = {"run_id": uuid.uuid4().hex, "created_utc": _utc_stamp(), "schema": SCHEMA,
                        "host": socket.gethostname(), "initialization": "warm_start" if warm_start_from else "scratch"}
            output_dir.mkdir(parents=True, exist_ok=True)
            lock = RunLock(output_dir, identity["run_id"])
            lock.acquire(False)
            atomic_write_json(identity_path, identity)

        legacy.seed_everything(args.seed)
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda" and not getattr(args, "deterministic", False):
            torch.backends.cuda.matmul.allow_tf32 = True
        train_loader, val_loader, val_hard_loader = build_resumable_loaders(args)
        sampler: ResumableSampler = train_loader.sampler
        corpus_ev = corpus_ev_of(args.manifest, real_only_ev=getattr(args, "real_corpus_ev", None),
                                 pipeline_only_ev=getattr(args, "pipeline_corpus_ev", None))
        if args.source_curve:
            legacy.check_source_curve_args(args)
        model = SDR2HDRNet(base_channels=args.base_channels, gate_conditioning=args.gate_conditioning,
                           curve_head=args.curve_head, source_curve=args.source_curve,
                           baseline_source=args.baseline_source, bounded_source=args.bounded_source,
                           reconstruction_policy=policy,
                           corpus_ev=corpus_ev).to(device)
        dequant_loss = None
        reconstruction_gradient_loss = None
        signed_shadow_loss = None
        if signed_shadow_weight:
            from training.signed_shadow import SignedShadowLoss
            with torch.random.fork_rng(devices=[]):
                signed_shadow_loss = SignedShadowLoss().to(device)
        if reconstruction_gradient_weight:
            from training.reconstruction_gradient import ReconstructionGradientLoss
            with torch.random.fork_rng(devices=[]):
                reconstruction_gradient_loss = ReconstructionGradientLoss().to(device)
        if dequant_weight:
            from training.dequantization import DequantizationLoss
            # Isolate lookup initialization from sampler RNG, including future
            # changes to how the baseline lookup is constructed.
            with torch.random.fork_rng(devices=[]):
                dequant_loss = DequantizationLoss(policy=policy).to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.lr, weight_decay=args.weight_decay)
        optimizer_steps = max(math.ceil(args.steps / args.grad_accum), 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimizer_steps,
                                                               eta_min=args.lr * 0.05)
        amp_config = {"autocast": "bfloat16" if device.type == "cuda" else "disabled", "grad_scaler": None}

        config = vars(args).copy()
        config["runtime"] = legacy.runtime_record(device)
        config["training_use"] = "research"
        if provenance:
            config["training_use"] = "internal-pilot" if provenance.get("scope") == "internal-pilot" \
                else "commercial-candidate"
            config["commercial_provenance"] = provenance
        experiment = dict(experiment or {})
        config["experiment"] = experiment
        augment = None
        if experiment.get("shadow_codec_p", 0) > 0:
            from training.dark_experiment import ShadowCodecAugment
            augment = ShadowCodecAugment(experiment["shadow_codec_p"], seed=args.seed + 3)
        config.update({"manifest_sha256": legacy.manifest_hash(args.manifest), "corpus_ev": corpus_ev,
                       "architecture": type(model).__name__, "input_contract": "8-bit normalized sRGB RGB",
                       "output_contract": "scene-linear RGB normalized to 10000 nits",
                       "initialization": identity["initialization"],
                       "resumable": {"schema": SCHEMA, "run_id": identity["run_id"]}})
        current_code = code_hashes()
        log_path = output_dir / "train.jsonl"
        start_step, best, best_step, best_model = 0, math.inf, 0, None
        score_history: list[float] = []
        baseline_counts: Counter = Counter()
        lineage = {"run_id": identity["run_id"], "initialization": identity["initialization"],
                   "segments": [], "warm_start_parent": None}
        resumed_from = None

        if warm_start_from:
            parent_path = Path(warm_start_from)
            parent = torch.load(parent_path, map_location="cpu", weights_only=True)
            model.load_state_dict(parent.get("model", parent), strict=True)
            lineage["warm_start_parent"] = {
                "path": str(parent_path), "sha256": sha256_file(parent_path), "step": parent.get("step"),
                "run_id": (parent.get("lineage") or {}).get("run_id"),
                "training_use": (parent.get("config") or {}).get("training_use"),
                "manifest_sha256": (parent.get("config") or {}).get("manifest_sha256"),
                "meaning": "weights only; optimizer, schedule, RNG and data order start fresh"}
            del parent

        if resume:
            path, state, checked = find_latest_state(output_dir, identity["run_id"])
            check_compatible(state["config"], config, state["code_sha256"], current_code,
                             state["torch_version"], allow_device_change)
            if commercial and state["config"].get("commercial_provenance") != provenance:
                raise ResumeError("commercial provenance differs from the saved run; resume refused")
            if state["dataset"]["manifest_sha256"] != config["manifest_sha256"]:
                raise ResumeError("manifest bytes changed since the state was saved")
            if state["amp"] != amp_config:
                raise ResumeError(f"AMP configuration differs: {state['amp']} -> {amp_config}")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_step, best, best_step = int(state["step"]), float(state["best"]), int(state["best_step"])
            best_model = state["best_model"]
            score_history = [float(x) for x in state["score_history"]]
            baseline_counts = Counter(state.get("baseline_counts") or {})
            lineage = state["lineage"]
            config = dict(config, runtime=config["runtime"])
            resumed_from = {"path": str(path), "sha256": sha256_file(path), "step": start_step,
                            "candidates_checked": checked}
            # Reconcile files a crash left ahead of the restored step.
            orphaned_log = _reconcile_log(log_path, start_step)
            best_path = output_dir / "best.pt"
            orphaned_best = None
            if best_path.exists():
                on_disk = torch.load(best_path, map_location="cpu", weights_only=True)
                if int(on_disk.get("step", -1)) != best_step:
                    orphaned_best = _move_aside(best_path, f"orphaned_after_step{start_step}")
                del on_disk
            if not best_path.exists():
                legacy._atomic_save({"model": best_model, "step": best_step, "best": best,
                                     "config": state["config"]}, best_path)
            resumed_from.update(orphaned_log=str(orphaned_log) if orphaned_log else None,
                                orphaned_best=str(orphaned_best) if orphaned_best else None)
            config = state["config"] if not allow_device_change else dict(state["config"], device=args.device)

        segment = {"index": len(lineage["segments"]), "start_step": start_step, "started_utc": _utc_stamp(),
                   "host": socket.gethostname(), "pid": os.getpid(), "resumed_from": resumed_from,
                   "runtime": legacy.runtime_record(device)}
        lineage["segments"].append(segment)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not resume:
            (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        atomic_write_json(output_dir / "lineage.json", lineage)

        def image_eval():
            clean = legacy.evaluate_image(model, val_loader, device, args.eval_batches,
                                          args.shadow_chroma_weight, args.shadow_smoothness_weight)
            merged = {f"clean_{k}": v for k, v in clean.items()}
            chosen = clean
            if val_hard_loader is not None:
                hard = legacy.evaluate_image(model, val_hard_loader, device, args.eval_batches,
                                             args.shadow_chroma_weight, args.shadow_smoothness_weight)
                merged.update({f"hard_{k}": v for k, v in hard.items()})
                if args.best_eval == "hard":
                    chosen = hard
            if args.best_metric in ("composite_gain", "preserved_composite_gain") and val_hard_loader is not None:
                key = "preserved_gain_db" if args.best_metric == "preserved_composite_gain" else "gain_db"
                merged["composite_gain"] = merged[f"hard_{key}"] + min(0.0, merged[f"clean_{key}"])
                if experiment.get("selection") == "pu_min":
                    from training.dark_experiment import pu_min_score
                    return merged, pu_min_score(merged)
                return merged, -merged["composite_gain"]
            if experiment.get("selection") == "pu_min":
                from training.dark_experiment import pu_min_score
                return merged, pu_min_score(merged)
            return merged, chosen[args.best_metric]

        batches_in_epoch = 0

        def full_state(step: int) -> dict:
            return {"resume_schema": SCHEMA, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "best": best, "best_step": best_step,
                    "best_model": best_model, "score_history": list(score_history),
                    "baseline_counts": dict(baseline_counts), "amp": amp_config, "grad_scaler": None,
                    "rng": capture_rng(), "sampler": sampler.state(batches_in_epoch, args.batch_size),
                    "augment_rng": augment.state() if augment is not None else None,
                    "config": config, "code_sha256": current_code, "torch_version": str(torch.__version__),
                    "dataset": {"manifest_sha256": config["manifest_sha256"],
                                "commercial_evidence_sha256": (provenance or {}).get("evidence_sha256"),
                                "commercial_provenance": provenance},
                    "lineage": lineage}

        def write_slot(step: int) -> dict:
            slots = [output_dir / "resume" / f"slot_{s}.pt" for s in ("a", "b")]
            ages = []
            for slot in slots:
                payload, _ = verify_state(slot, identity["run_id"]) if slot.exists() else (None, "")
                ages.append(payload["step"] if payload is not None else -1)
            target = slots[0] if ages[0] <= ages[1] else slots[1]
            return save_state(full_state(step), target)

        def status(state: str, step: int, reason: str | None = None) -> None:
            atomic_write_json(output_dir / "resume_status.json", {
                "state": state, "reason": reason, "step": step, "target_steps": args.steps,
                "run_id": identity["run_id"], "pid": os.getpid(), "updated_unix": time.time(),
                "release_authorized": False})

        # -- data iterator: created where the legacy trainer creates it, so the
        # global RNG draw it makes lands in the same place in the stream.
        if resume:
            sampler.resume(state["sampler"])
            batches_in_epoch = int(state["sampler"]["batches_consumed"])
        data_iter = iter(train_loader)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        if resume:
            restore_rng(state["rng"], skip_cuda=allow_device_change)
            if augment is not None:
                augment.restore(state["augment_rng"])
            del state
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": start_step, "resume": {
                    "segment": segment["index"], "from": resumed_from["path"],
                    "sha256": resumed_from["sha256"]}}) + "\n")
        started = time.time()
        if start_step == 0:
            initial_metrics, best = image_eval()
            score_history.append(best)
            best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
            legacy._atomic_save({"model": model.state_dict(), "step": 0, "best": best, "config": config},
                                output_dir / "best.pt")
            print(f"[baseline] {json.dumps(initial_metrics)}", flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": 0, "eval": initial_metrics, "baseline": True}) + "\n")
            if observer is not None:
                observer("eval", 0, initial_metrics)
        status("training", start_step)

        step = start_step
        stop_reason = None
        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batches_in_epoch = 0
                batch = next(data_iter)
            batches_in_epoch += 1
            if augment is not None:
                batch["sdr"] = augment(batch["sdr"])
            sdr, target = batch["sdr"].to(device, non_blocking=True), batch["hdr"].to(device, non_blocking=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" \
                else contextlib.nullcontext()
            with amp:
                capture = getattr(args, "crushed_hold_domain", "postclamp") == "preclamp"
                output = model(sdr, source_curve=legacy._source_of(model, batch, device),
                               **({"record_preclamp": True} if capture else {}),
                               **legacy._picker_of(model, batch, device))
                losses = sdr2hdr_loss(
                    output, sdr, target, args.shadow_chroma_weight, args.shadow_smoothness_weight,
                    target_ceiling=(batch["ceiling"].to(device, non_blocking=True) if "ceiling" in batch else None),
                    baseline_weight=args.baseline_weight if args.curve_head else 0.0,
                    region_focus=args.region_focus, crushed_hold=args.crushed_hold,
                    crushed_hold_domain=getattr(args, "crushed_hold_domain", "postclamp"))
                if experiment.get("near_black_weight", 0) > 0:
                    from training.dark_experiment import near_black_log_loss
                    losses["near_black"] = near_black_log_loss(
                        output.hdr.float(), target.float(), sdr.float(),
                        experiment.get("near_black_code", 16) / 255.0)
                    losses["total"] = losses["total"] + experiment["near_black_weight"] * losses["near_black"]
                if experiment.get("run05_restoration", False):
                    from training.run05_objective import restoration_total
                    losses["total"] = restoration_total(losses, args.region_focus, args.crushed_hold)
                if dequant_loss is not None:
                    dq = dequant_loss(output.hdr, target, sdr)
                    losses['dequant_code'] = dq['code']
                    losses['dequant_gradient'] = dq['gradient']
                    losses['total'] = losses['total'] + dequant_weight * dq['total']
                if reconstruction_gradient_loss is not None:
                    rg = reconstruction_gradient_loss(output.hdr, target, sdr)
                    losses['reconstruction_gradient'] = rg['total']
                    losses['reconstruction_support'] = rg['support_fraction']
                    losses['total'] = losses['total'] + reconstruction_gradient_weight * rg['total']
                if signed_shadow_loss is not None:
                    losses['signed_shadow'] = signed_shadow_loss(output.hdr, target, sdr)
                    losses['total'] = losses['total'] + signed_shadow_weight * losses['signed_shadow']
                loss = losses["total"]
                scaled_loss = loss / args.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}; last valid state is kept on disk")
            if observer is not None:
                observer("loss", step, {"loss": loss, "output": output.hdr})
            scaled_loss.backward()
            if step % args.grad_accum == 0 or step == args.steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm,
                                               error_if_nonfinite=observer is not None)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            record = {"step": step, "lr": scheduler.get_last_lr()[0],
                      **{key: float(value.detach()) for key, value in losses.items()}}
            if args.baseline_source:
                from rudra.source_baseline import BASELINE_CURVES
                baseline_counts.update(BASELINE_CURVES[int(i)] for i in batch["baseline_curve"].tolist())
                record["baseline_curve_counts_since_start"] = dict(baseline_counts)
            if step == 1 or step % args.log_every == 0:
                rate = (step - start_step) * args.batch_size / max(time.time() - started, 1e-6)
                print(f"[image] step {step}/{args.steps} loss={record['total']:.5f} samples/s={rate:.2f}", flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if step % args.eval_every == 0 or step == args.steps:
                metrics, score = image_eval()
                print(f"[eval] step {step}: {json.dumps(metrics)}", flush=True)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step, "eval": metrics}) + "\n")
                score_history.append(score)
                smoothed = legacy.selection_score(score_history, args.best_smoothing)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step, "score_raw": score, "score_smoothed": smoothed,
                                             "best_smoothing": args.best_smoothing}) + "\n")
                if smoothed < best:
                    best, best_step = smoothed, step
                    best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    legacy._atomic_save({"model": model.state_dict(), "step": step, "best": best,
                                         "config": config}, output_dir / "best.pt")
                if observer is not None:
                    observer("eval", step, metrics)
            if step % args.save_every == 0 or step == args.steps:
                save_state(full_state(step), output_dir / f"step_{step:07d}.pt")
            elif resume_save_every and step % resume_save_every == 0:
                write_slot(step)
            if step % 20 == 0:
                status("training", step)
            stop_reason = control.should_stop(step) if step < args.steps else None
            if stop_reason:
                if not (step % args.save_every == 0):
                    write_slot(step)
                break
        if stop_reason:
            status("stopped_resumable", step, stop_reason)
            print(f"[stopped] {stop_reason}; resume state saved at step {step}", flush=True)
            return {"state": "stopped_resumable", "step": step, "reason": stop_reason,
                    "run_id": identity["run_id"], "best": str(output_dir / "best.pt")}
        status("completed", step)
        return {"state": "completed", "step": step, "run_id": identity["run_id"],
                "best": str(output_dir / "best.pt"), "release_authorized": False}
    finally:
        if lock is not None:
            lock.release()


# ----------------------------------------------------------------------- CLI
def parse(argv: list[str]):
    from training import train_sdr2hdr as legacy
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--run05-restoration", action="store_true",
                     help="remove the zero-residual prior for run05 restoration targets")
    own.add_argument("--resume", dest="resume_run", action="store_true",
                     help="exactly continue the run in --output-dir from its newest valid state")
    own.add_argument("--warm-start-from", help="research only: weights-only start of a NEW run")
    own.add_argument("--resume-save-every", type=int, default=0,
                     help="extra ping-pong resume saves between numbered checkpoints (0 = numbered only)")
    own.add_argument("--max-hours", type=float, default=None, help="graceful, resumable stop after this long")
    own.add_argument("--recover-stale-lock", action="store_true")
    own.add_argument("--shadow-codec-p", type=float, default=0.0,
                     help="run03: probability of shadow-only JPEG/crush degradation of a training crop")
    own.add_argument("--near-black-weight", type=float, default=0.0,
                     help="run03: weight of the log-domain near-black luminance+chroma term")
    own.add_argument("--near-black-code", type=int, default=16, help="run03: SDR max code (0-255) defining near-black")
    own.add_argument('--dequant-code-weight', type=float, default=0.0,
                     help='experimental code/reference-gradient loss for fixed source-preserving pairs')
    own.add_argument('--reconstruction-gradient-weight', type=float, default=0.0,
                     help='experimental reference-gradient loss near missing-range regions at -3/0/+3 EV')
    own.add_argument('--signed-shadow-weight', type=float, default=0.0,
                     help='experimental signed source-gamut radiance supervision in dark regions')
    own.add_argument("--selection", choices=("legacy", "pu_min"), default="legacy",
                     help="run03: pu_min selects best.pt on -min(clean, hard PU gain), trailing median")
    own.add_argument("--allow-device-change", action="store_true",
                     help="continue on a different device; recorded, and no longer an exact continuation")
    mine, rest = own.parse_known_args(argv)
    previous = sys.argv
    try:
        sys.argv = ["resumable_training.py", *rest]
        args = legacy.parse_args()
    finally:
        sys.argv = previous
    return mine, args


def experiment_of(mine) -> dict:
    """Only non-default experiment options are recorded, so a plain run keeps an empty dict."""
    exp = {}
    if not math.isfinite(mine.signed_shadow_weight) or mine.signed_shadow_weight < 0:
        raise ValueError('signed-shadow-weight must be finite and nonnegative')
    if mine.signed_shadow_weight:
        exp['signed_shadow_weight'] = mine.signed_shadow_weight
    if not math.isfinite(mine.reconstruction_gradient_weight) or mine.reconstruction_gradient_weight < 0:
        raise ValueError('reconstruction-gradient-weight must be finite and nonnegative')
    if mine.reconstruction_gradient_weight:
        exp['reconstruction_gradient_weight'] = mine.reconstruction_gradient_weight
    if not math.isfinite(mine.dequant_code_weight) or mine.dequant_code_weight < 0:
        raise ValueError('dequant-code-weight must be finite and nonnegative')
    if mine.dequant_code_weight:
        exp['dequant_code_weight'] = mine.dequant_code_weight
    if mine.run05_restoration:
        exp["run05_restoration"] = True
    if mine.shadow_codec_p > 0:
        exp["shadow_codec_p"] = mine.shadow_codec_p
    if mine.near_black_weight > 0:
        exp.update(near_black_weight=mine.near_black_weight, near_black_code=mine.near_black_code)
    if mine.selection != "legacy":
        exp["selection"] = mine.selection
    return exp


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["inspect"]:
        for p in argv[1:]:
            print(json.dumps(classify_checkpoint(Path(p)), indent=2))
        return 0
    mine, args = parse(argv)
    control = Control(Path(args.output_dir) / "STOP", mine.max_hours * 3600 if mine.max_hours else None)
    control.install_signals()
    result = train(args, resume=mine.resume_run, warm_start_from=mine.warm_start_from,
                   resume_save_every=mine.resume_save_every, control=control,
                   recover_stale_lock=mine.recover_stale_lock, allow_device_change=mine.allow_device_change,
                   experiment=experiment_of(mine))
    print(json.dumps(result), flush=True)
    return EXIT_STOPPED_RESUMABLE if result["state"] == "stopped_resumable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
