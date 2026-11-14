"""Set up the AI fill (layer 3) from inside the app (4 Nov 2026).

What scripts/GENFILL.ps1 -Setup and scripts/genfill_setup.sh do by hand, as one
command the app runs and follows: it checks the machine, installs the Python
packages the fill needs into the running interpreter, and downloads
Wan2.1-VACE-1.3B-diffusers (Apache-2.0, about 25 GB) into --base. A download
that stops resumes where it left off (huggingface_hub keeps the parts).

    python -m rudra.genfill.setup --base <folder> [--json] [--check-only] [--skip-packages]

With --json every step is one JSON object per line on stdout:
    {"event": "step", "step": "packages" | "model" | "verify", "text": ...}
    {"event": "log", "text": ...}
    {"event": "check", "device": "cuda", "memory_gb": 16.0, "free_gb": 120.5, "enough": true, ...}
    {"event": "done", "ok": true, "base": ..., "device": ...} or {"event": "done", "ok": false, "error": ...}
Exit code 0 when the fill is ready to run, 1 when a step failed, 2 when the
machine cannot run it (GPU memory or disk).
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ID = 'Wan-AI/Wan2.1-VACE-1.3B-diffusers'
MODEL_GB = 25.0          # the snapshot on disk, rounded up
MIN_MEMORY_GB = 15.5     # a 16 GB GPU (or a Mac with 16 GB) reports a little under 16
# pyproject.toml [project.optional-dependencies] genfill, plus OpenColorIO for the AP0 masters.
PACKAGES = ['diffusers>=0.35', 'transformers>=4.48', 'peft>=0.13', 'accelerate>=1.0', 'OpenEXR>=3.2',
            'safetensors>=0.4', 'sentencepiece', 'ftfy', 'huggingface-hub>=0.25', 'opencolorio>=2.4']
IMPORTS = ['diffusers', 'transformers', 'peft', 'accelerate', 'OpenEXR', 'safetensors', 'sentencepiece',
           'ftfy', 'huggingface_hub', 'PyOpenColorIO']

_JSON = False


def emit(event: str, **fields) -> None:
    if _JSON:
        print(json.dumps({'event': event, **fields}), flush=True)
    else:
        text = fields.get('text') or ', '.join(f'{k}={v}' for k, v in fields.items())
        print(f'[{event}] {text}', flush=True)


def machine(base: Path) -> dict:
    """The device the fill would run on, its memory, and the free disk where the model goes."""
    out = {'device': 'cpu', 'memory_gb': None, 'gpu': None}
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out.update(device='cuda', gpu=p.name, memory_gb=round(p.total_memory / 2**30, 1))
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            out['device'] = 'mps'
            try:   # a Mac's GPU shares its memory
                mem = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout)
                out['memory_gb'] = round(mem / 2**30, 1)
            except (OSError, ValueError):
                pass
        out['torch'] = torch.__version__
    except ImportError:
        out['torch'] = None
    probe = base
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    out['free_gb'] = round(shutil.disk_usage(probe).free / 2**30, 1)
    return out


def no_gpu_reason(m: dict) -> str:
    """Why the fill cannot run here, in words that say what to do (6 Nov 2026: a Python without
    PyTorch was reported as 'no GPU', and the artist went looking for a driver problem)."""
    if m.get('torch') is None:
        return (f'PyTorch is not installed in {sys.executable}. Install the build for your GPU '
                '(https://pytorch.org/get-started/locally/: CUDA on Windows and Linux), then run this again')
    if m['device'] == 'cpu' and '+cpu' in str(m.get('torch')):
        return (f"PyTorch {m['torch']} is the CPU-only build: install the CUDA build instead "
                '(https://pytorch.org/get-started/locally/)')
    if m['device'] == 'cpu':
        return (f"PyTorch {m['torch']} found no CUDA GPU or Apple GPU: the AI fill needs an NVIDIA GPU with 16 GB "
                'and a current driver (Windows, Linux) or Apple silicon with 16 GB (Mac)')
    return f"the GPU has {m['memory_gb']} GB; the AI fill needs 16 GB"


def have_model(base: Path) -> bool:
    return (base / 'transformer' / 'config.json').is_file() and (base / 'model_index.json').is_file()


def missing_packages() -> list[str]:
    missing = []
    for name in IMPORTS:
        try:
            importlib.import_module(name)
        except Exception:   # noqa: BLE001 (a broken install counts as missing)
            missing.append(name)
    return missing


def install_packages() -> bool:
    cmd = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *PACKAGES]
    emit('log', text=' '.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            emit('log', text=line)
    return proc.wait() == 0


def download_model(base: Path) -> None:
    from huggingface_hub import snapshot_download
    base.mkdir(parents=True, exist_ok=True)
    # local_dir: real files in --base (what the app looks for), resumable.
    snapshot_download(REPO_ID, local_dir=str(base))


def main(argv=None) -> int:
    global _JSON
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True, help='where Wan2.1-VACE-1.3B-diffusers goes (or already is)')
    ap.add_argument('--json', action='store_true', help='one JSON object per line')
    ap.add_argument('--check-only', action='store_true', help='report what is there and what is missing; change nothing')
    ap.add_argument('--skip-packages', action='store_true')
    ap.add_argument('--allow-small-gpu', action='store_true', help='set up even under 16 GB (the fill may not run)')
    a = ap.parse_args(argv)
    _JSON = a.json
    base = Path(a.base).expanduser().resolve()

    m = machine(base)
    model_there = have_model(base)
    need_disk = 0.0 if model_there else MODEL_GB
    enough_mem = m['memory_gb'] is not None and m['memory_gb'] >= MIN_MEMORY_GB
    enough_disk = m['free_gb'] >= need_disk
    missing = missing_packages()
    emit('check', **m, model=model_there, base=str(base), missing_packages=missing,
         need_disk_gb=need_disk, enough_memory=enough_mem, enough_disk=enough_disk,
         enough=enough_mem and enough_disk)
    if a.check_only:
        ready = model_there and not missing and m['device'] != 'cpu'
        emit('done', ok=ready, base=str(base), device=m['device'], check_only=True)
        return 0 if ready else 1
    if m['device'] == 'cpu' or (not enough_mem and not a.allow_small_gpu):
        why = no_gpu_reason(m)
        emit('done', ok=False, error=why, base=str(base), device=m['device'])
        return 2
    if not enough_disk:
        emit('done', ok=False, error=f"{m['free_gb']} GB free where the model goes; it needs {need_disk:.0f} GB",
             base=str(base), device=m['device'])
        return 2

    if missing and not a.skip_packages:
        emit('step', step='packages', text='Installing ' + ', '.join(missing))
        if not install_packages():
            emit('done', ok=False, error='pip install failed (see the log)', base=str(base))
            return 1
        importlib.invalidate_caches()
        still = missing_packages()
        if still:
            emit('done', ok=False, error='still missing after the install: ' + ', '.join(still), base=str(base))
            return 1

    if not model_there:
        emit('step', step='model', text=f'Downloading {REPO_ID} (about {MODEL_GB:.0f} GB) into {base}')
        try:
            download_model(base)
        except Exception as e:   # noqa: BLE001 (network, disk, auth: all end the same way)
            emit('done', ok=False, error=f'model download: {e}', base=str(base))
            return 1

    emit('step', step='verify', text='Checking the model')
    try:
        cfg = json.loads((base / 'transformer' / 'config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        emit('done', ok=False, error=f'the model in {base} is incomplete: {e}', base=str(base))
        return 1
    emit('log', text=f"transformer {cfg.get('_class_name')}, {cfg.get('num_layers')} layers, vace {cfg.get('vace_layers')}")
    emit('done', ok=True, base=str(base), device=m['device'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
