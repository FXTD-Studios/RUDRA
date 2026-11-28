"""AI fill for the OFX plug-in (phase 3, 26 Nov 2026): one stored shot through AI fill v5, as the RUDRA app runs it.

The plug-in (native/ofx/fill_store.cpp) stores every frame the host renders while it collects:

    <dir>/request.json      the clip's frames, the route's condition, the prompt, the seed, the luminance limit
    <dir>/codes/f.NNNNNN.tif   the SDR codes, 16-bit
    <dir>/master/f.NNNNNN.exr  the master the plug-in shows, ACEScg, 1.0 = diffuse white, 0 EV

and, once all are in, starts this beside the host:

    python rudra/genfill/ofx_job.py --dir <dir> --repo <RUDRA's Python source> --base <Wan2.1-VACE-1.3B> --lora <v5>

It runs `python -m rudra.genfill.run` with the app's command line for AI fill v5 (recipe(), the same flags as
native/app/genfill_job.cpp genfill_arguments, region re-render), into <dir>/fill.part, renamed <dir>/fill when
whole. Its progress goes to <dir>/status.json, which the plug-in reads:

    {"state": "running" | "done" | "failed" | "cancelled", "done": k, "of": n, "message": "...", "updated": <time>}

written at least every 30 s while it runs (the plug-in takes a silence of five minutes as a stop). A file
<dir>/cancel (the plug-in's Cancel) stops the fill at its next step. Nothing is written to stdout: the plug-in
does not read it.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# native/app/genfill_job.hpp: AI fill v5 and the finish 1.3 runs it with (kGenFillV5Sha256, kGenFillV5Hold,
# kGenFillV5BlendStill, kGenFillV5HoldCalm). tests/test_ofx_job.py checks they are the app's.
V5_SHA256 = "5c5d4ab4107088f86bdc8213ae2ec21c1a882cc6d942141f0e12fb7733fda5e6"
V5_HOLD = "thin"
V5_BLEND_STILL = "0.95"
V5_HOLD_CALM = "0.95"
STEPS = 30          # 1.3.1: 8 steps drew stripes on 09V (G:\rudra_training\shot_check_1_3_1); the app's 30
HEARTBEAT = 30.0    # seconds between status writes at the least
# 1.3.2: ofx_prompt.py embed reads a new prompt while the frames are collected and holds this file in the prompt
# cache meanwhile, touching it every EMBED_BEAT seconds; the fill waits for it (one 11 GB encoder at a time).
EMBED_LOCK = ".embedding.lock"
EMBED_BEAT = 10.0
EMBED_STALE = 90.0  # a lock not touched for this long was left by a process that ended
EMBED_WAIT = 1800.0
# 1.3.2: the GPU memory mode (run.py --low-memory) from the memory free when the fill starts, as the host beside it
# (Resolve, Nuke) holds part of the card. tools/bench_genfill_memory.py (16 Nov 2026, RTX 4080 SUPER, 33 frames,
# reports/genfill_memory_2026-11-16_1801): 'off', as gated, peaked at 16.2 GB reserved; 'swap' at 10.8 GB, the same
# fill to the last bit and no slower; 'on' at 6.7 GB, the same there too. 1 GB is left for the CUDA context. A fill
# that runs out of memory all the same is run once more with 'on'.
OFF_FREE_GB = 17.2
SWAP_FREE_GB = 11.8
MEMORY_MODES = ("free", "auto", "off", "swap", "tile", "on")


def recipe(master: str, out: str, source: str, base: str, lora: str, condition: str, seed: int, prompt: str,
           region_json: str, steps: int = STEPS, prompt_cache: str = "", prompt_strength: float = 1.0,
           low_memory: str = "auto") -> list[str]:
    """The command line after the Python: genfill_job.cpp genfill_arguments with rerender on and AI fill v5.
    prompt_strength (1.3.2): the plug-in's Prompt strength, as the app's Deliver passes it (only when not 1).
    low_memory (1.3.2): run.py's --low-memory, given when it is not 'auto' (memory_mode_for)."""
    a = ["-m", "rudra.genfill.run", "--master", master, "--out", out, "--base", base, "--lora", lora,
         "--region-rerender", "--source", source, "--condition", condition,
         "--tau-high", "240", "--tau-low", "-1", "--feather", "8", "--brightness-model", "none",
         "--seed", str(seed), "--steps", str(steps),
         "--rerender-hue", "source", "--rerender-colour", "0.3", "--rerender-level", "3.3",
         "--rerender-smooth", "0.8",
         "--rerender-dip", "0.3",
         "--rerender-independent", "--rerender-calm", "0.9",
         "--rerender-full-last", "--json"]
    if prompt.strip():
        a += ["--prompt-highlight", prompt.strip()]
        if prompt_strength != 1.0:
            a += ["--prompt-strength", f"{prompt_strength:g}"]
    if prompt_cache:
        a += ["--prompt-cache", prompt_cache]
    if region_json:
        a += ["--region", region_json]
    a += ["--glob", "f.*.exr"]
    a += ["--rerender-hold-clipped", V5_HOLD]
    a += ["--rerender-blend-still", V5_BLEND_STILL, "--rerender-hold-calm", V5_HOLD_CALM]
    if condition == "aces":
        a += ["--rerender-cap-master", "--mask-first-frame"]
    if low_memory != "auto":
        a += ["--low-memory", low_memory]
    return a


def free_gpu_gb(run=subprocess.run) -> float | None:
    """The memory free now on the first NVIDIA GPU, in GB, from nvidia-smi (it holds none itself); None when it
    cannot say (no NVIDIA driver, another GPU)."""
    try:
        r = run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                text=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return float(r.stdout.split()[0]) / 1024.0
    except (OSError, subprocess.SubprocessError, IndexError, ValueError, AttributeError):
        return None


def memory_mode_for(free_gb: float | None) -> str:
    """run.py's --low-memory for a fill beside a host: 'off' (as gated) when the memory free holds it, else 'swap'
    (the same fill) when that fits, else 'on'. Not known: 'swap', which on a GPU with room costs nothing."""
    if free_gb is None:
        return "swap"
    if free_gb >= OFF_FREE_GB:
        return "off"
    return "swap" if free_gb >= SWAP_FREE_GB else "on"


def fill_env(inherited, repo: Path) -> dict:
    """The fill's environment: the host's (Resolve's, Nuke's: this job inherits it), without what points a Python at
    the host's own Python (PYTHONHOME, its PYTHONPATH), so the fill imports only its own and RUDRA's (1.3.2)."""
    env = {k: v for k, v in inherited.items() if k.upper() not in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP")}
    env["PYTHONPATH"] = str(repo)
    return env


def stderr_is(path: Path) -> bool:
    """Whether this process's stderr is the file at `path`."""
    try:
        a, b = os.fstat(sys.stderr.fileno()), os.stat(path)
    except (OSError, ValueError, AttributeError, io.UnsupportedOperation):
        return False
    return a.st_ino != 0 and (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def open_log(path: Path):
    """The job's log, to append to. The plug-in starts the job with job.log as its stderr, which on Windows it opens
    for itself (shared for reading only: platform/process.cpp), so a second handle for writing is refused: then the
    log is stderr itself, the same file. 1.3.2 (27 Nov 2026): every fill started from Resolve or Nuke stopped on that
    refusal before it began ("the AI fill ended without finishing"), as only the tests opened the log themselves."""
    if stderr_is(path):
        with contextlib.suppress(AttributeError, ValueError, io.UnsupportedOperation):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        return contextlib.nullcontext(sys.stderr)
    try:
        return open(path, "a", encoding="utf-8", errors="replace")
    except PermissionError:
        return contextlib.nullcontext(sys.stderr)


def out_of_memory(result: dict | None) -> bool:
    """Whether the fill's last word says the GPU ran out of memory."""
    text = f"{(result or {}).get('error', '')} {(result or {}).get('detail', '')}".lower()
    return "outofmemory" in text or "out of memory" in text


class Status:
    """status.json, written whole (a .part renamed over it), at most every HEARTBEAT seconds apart."""

    def __init__(self, folder: Path, of: int):
        self.path, self.of = folder / "status.json", of
        self.state = {"state": "running", "done": 0, "of": of, "message": "starting"}
        self.lock = threading.Lock()

    def set(self, **kw):
        with self.lock:
            self.state.update(kw)
            self.state["updated"] = time.time()
            part = self.path.with_name(self.path.name + ".part")
            part.write_text(json.dumps(self.state), encoding="utf-8")
            os.replace(part, self.path)

    def beat(self, stop: threading.Event):
        while not stop.wait(HEARTBEAT):
            self.set()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prompt_cache_for(folder: Path, lora: Path) -> Path:
    """One prompt cache for every OFX fill (27 Nov 2026): <ofx_fill>/prompt_cache. A prompt not in it loads umT5
    (11 GB) to embed it, once; the cache beside the LoRA (the app's presets, the prompts the app has embedded) is
    copied in first, so a prompt the app knows costs nothing."""
    return seed_cache(folder.parent.parent / "prompt_cache", lora)


def seed_cache(cache: Path, lora: Path) -> Path:
    """`cache`, made, with what the cache beside the LoRA has that it has not."""
    cache.mkdir(parents=True, exist_ok=True)
    beside = lora.parent / "prompt_cache"
    if beside.is_dir():
        for f in beside.iterdir():
            if f.is_file() and not f.name.startswith(".") and not (cache / f.name).exists():
                try:
                    shutil.copy2(f, cache / f.name)
                except OSError:
                    pass
    return cache


def embedding_running(cache: Path) -> bool:
    """Whether ofx_prompt.py embed holds the cache (its lock touched within EMBED_STALE seconds)."""
    try:
        return time.time() - (cache / EMBED_LOCK).stat().st_mtime < EMBED_STALE
    except OSError:
        return False


def wait_for_embedding(cache: Path, status, limit: float = EMBED_WAIT, poll: float = 2.0) -> bool:
    """Waits while the prompt is read beside the collection (status says so); False when it gave up waiting."""
    t0 = time.time()
    while embedding_running(cache):
        if time.time() - t0 > limit:
            return False
        status.set(message="reading the prompt (started when Generate was pressed; once per new prompt)")
        time.sleep(poll)
    return True


def strength_of(req: dict) -> float:
    """The request's prompt strength (the plug-in's Prompt strength), 0..2; 1 when it has none."""
    try:
        v = float(req.get("prompt_strength", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(2.0, v)) if v == v else 1.0


def steps_of(req: dict) -> int:
    """The request's steps (the plug-in's Steps), 10..50; 30 when it has none."""
    try:
        return max(10, min(50, int(req.get("steps", STEPS))))
    except (TypeError, ValueError):
        return STEPS


def check_frames(folder: Path, n: int) -> str | None:
    """Why the stored frames cannot be filled, or None."""
    for sub, ext in (("codes", ".tif"), ("master", ".exr")):
        have = sorted((folder / sub).glob(f"f.*{ext}"))
        if len(have) != n or [p.stem for p in have] != [f"f.{i:06d}" for i in range(n)]:
            return f"{len(have)} of {n} frames stored in {sub}"
    return None


def run_fill(cmd: list[str], repo: Path, env: dict, folder: Path, status: Status, n: int, log) -> tuple[int, dict | None, bool]:
    """One run of the fill: its exit code, its last word (the done or error line) and whether Cancel stopped it.
    Its progress goes to status.json as it comes."""
    log.write(f"== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
    log.flush()
    child = subprocess.Popen(cmd, cwd=str(repo), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
    stop = threading.Event()
    threading.Thread(target=status.beat, args=(stop,), daemon=True).start()
    cancelled = threading.Event()

    def watch_cancel():
        while not stop.wait(1.0):
            if (folder / "cancel").exists() and not cancelled.is_set():
                cancelled.set()
                status.set(message="stopping")
                try:
                    child.stdin.write("cancel\n")
                    child.stdin.flush()
                except OSError:
                    pass
    threading.Thread(target=watch_cancel, daemon=True).start()

    result = None
    for line in child.stdout:
        log.write(line)
        try:
            e = json.loads(line)
        except ValueError:
            continue
        ev = e.get("event")
        if ev == "embedding":
            status.set(message="reading a new prompt (its 11 GB text model loads once)")
        elif ev == "ready":
            status.set(message="model loaded")
        elif ev == "progress":
            # 1.3.2: the step too, so the plug-in's bar moves within a window ("window 2 of 5, step 12 of 30")
            step = f", step {e.get('step')} of {e.get('steps')}" if e.get("steps") else ""
            status.set(message=f"window {e.get('window')} of {e.get('windows')}{step}")
        elif ev == "frame":
            status.set(done=int(e.get("index", 0)), of=int(e.get("of", n)))
        elif ev in ("done", "error"):
            result = e
    code = child.wait()
    stop.set()
    log.flush()
    return code, result, cancelled.is_set()


def main(argv: list[str] | None = None, runner: list[str] | None = None) -> int:
    """runner: the program that fills (the tests' stand-in); default this Python with `-m rudra.genfill.run`."""
    out, sys.stdout = sys.stdout, sys.stderr   # the plug-in does not read stdout: nothing may fill its pipe
    try:
        return _main(argv, runner)
    finally:
        sys.stdout = out


def _main(argv: list[str] | None, runner: list[str] | None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--skip-lora-check", action="store_true", help="tests only: any LoRA")
    ap.add_argument("--low-memory", default="free", choices=MEMORY_MODES,
                    help="run.py's GPU memory mode; 'free' (the default) chooses it from the memory free now")
    a = ap.parse_args(argv)
    folder, repo = Path(a.dir), Path(a.repo)
    req = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    n = int(req["frames"])
    status = Status(folder, n)
    status.set(state="running", done=0, message="checking the stored frames")
    why = check_frames(folder, n)
    if why is None and not a.skip_lora_check and sha256(Path(a.lora)) != V5_SHA256:
        why = f"{a.lora} is not AI fill v5 (File > Model packages in RUDRA)"
    if why:
        status.set(state="failed", message=why)
        return 2
    out, final = folder / "fill.part", folder / "fill"
    shutil.rmtree(out, ignore_errors=True)
    region = req.get("region")
    prompt = req.get("prompt", "")
    cache = prompt_cache_for(folder, Path(a.lora)) if prompt.strip() else None
    if cache is not None:
        wait_for_embedding(cache, status)
    env = fill_env(os.environ, repo)
    free = free_gpu_gb() if a.low_memory == "free" else None
    mode = memory_mode_for(free) if a.low_memory == "free" else a.low_memory
    with open_log(folder / "job.log") as log:
        log.write(f"\n== {time.strftime('%Y-%m-%d %H:%M:%S')} GPU memory free: "
                  f"{'unknown' if free is None else f'{free:.1f} GB'}; memory mode {mode}\n")
        for attempt in [mode] + (["on"] if mode != "on" else []):
            args = recipe(str(folder / "master"), str(out), str(folder / "codes"), a.base, a.lora, req["condition"],
                          int(req.get("seed", 0)), prompt, json.dumps(region) if region else "", steps_of(req),
                          str(cache) if cache else "", strength_of(req), attempt)
            shutil.rmtree(out, ignore_errors=True)
            code, result, cancelled = run_fill((runner or [sys.executable]) + args, repo, env, folder, status, n, log)
            if cancelled:
                shutil.rmtree(out, ignore_errors=True)
                status.set(state="cancelled", message="cancelled")
                return 1
            if code == 0 and result and result.get("ok") and out.is_dir():
                break
            shutil.rmtree(out, ignore_errors=True)
            if out_of_memory(result) and attempt != "on":
                log.write("== not enough GPU memory: once more with --low-memory on\n")
                status.set(done=0, message="not enough GPU memory: drawing again in low-memory mode")
                continue
            detail = (result or {}).get("detail") or f"exit {code}"
            if out_of_memory(result):
                message = (f"not enough GPU memory{', even in low-memory mode' if attempt == 'on' else ''}: close other "
                           f"programs that use the GPU, then press Generate again ({detail}; log: {folder / 'job.log'})")
            else:
                message = f"{(result or {}).get('error', 'the fill')}: {detail} (log: {folder / 'job.log'})"
            status.set(state="failed", message=message)
            return 2
    shutil.rmtree(final, ignore_errors=True)
    os.replace(out, final)
    stats = {k: result.get(k) for k in ("generated_fraction", "fill_flicker_p95_ev", "fallback_frames", "seconds")}
    status.set(state="done", done=n, of=n, message="done", stats=stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
