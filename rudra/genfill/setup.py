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

1.2.6 (readiness report, checklist item 7): an offline install and pinned, hashed models.
model_manifest.json (beside this file, shipped with the app) holds, for Wan2.1-VACE and for
Florence-2, the Hugging Face revision and every file's size and SHA-256:

    --make-manifest              write it from a known-good --base (and its Florence-2 folder):
                                 the revision is the one huggingface_hub recorded when it downloaded
    --from <folder|zip>          install Wan2.1-VACE from a folder or zip, no network; refused
                                 unless every file matches the manifest
    --florence                   also set up Florence-2 (Suggest a prompt) in <base>/../Florence-2-base,
    --florence-from <folder|zip> downloaded at the pinned revision or installed from a folder or zip
                                 (the app's Set up automatically passes --florence: a download that
                                 fails is a warning, the fill does not need it; a mismatch is an error)
    --florence-only              Florence-2 alone: no GPU check, no packages, no Wan step
    --verify                     check every file's SHA-256 against the manifest (a download or an
                                 install is always checked)
A download asks for the manifest's revision, never "latest". Without a manifest entry nothing can
be checked: --from then needs --allow-unverified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import warnings
import zipfile
from pathlib import Path

REPO_ID = 'Wan-AI/Wan2.1-VACE-1.3B-diffusers'
FLORENCE_ID = 'florence-community/Florence-2-base'   # Suggest a prompt (prompts.py); about 0.5 GB
FLORENCE_DIR = 'Florence-2-base'                     # beside the Wan folder
MANIFEST = Path(__file__).with_name('model_manifest.json')
MODEL_GB = 25.0          # the snapshot on disk, rounded up
MIN_MEMORY_GB = 15.5     # a 16 GB GPU (or a Mac with 16 GB) reports a little under 16
# pyproject.toml [project.optional-dependencies] genfill, plus OpenColorIO for the AP0 masters, as (what the
# fill imports, the package that brings it). 1.3.2: NumPy and OpenCV too (decode.py, masks.py and composite.py
# import cv2): the bootstrap installed them and this list did not, so a Python set up here lacked them.
REQUIREMENTS = [('numpy', 'numpy>=1.26,<3'), ('cv2', 'opencv-python-headless>=4.9'), ('diffusers', 'diffusers>=0.35'),
                ('transformers', 'transformers>=4.48'), ('peft', 'peft>=0.13'), ('accelerate', 'accelerate>=1.0'),
                ('OpenEXR', 'OpenEXR>=3.2'), ('safetensors', 'safetensors>=0.4'), ('sentencepiece', 'sentencepiece'),
                ('ftfy', 'ftfy'), ('huggingface_hub', 'huggingface-hub>=0.25'), ('PyOpenColorIO', 'opencolorio>=2.4')]
PACKAGES = [req for _, req in REQUIREMENTS]
IMPORTS = [name for name, _ in REQUIREMENTS]
# The driver PyTorch's CUDA 12.8 build (the bootstrap's) asks for on Windows and Linux.
DRIVER_FOR_CUDA_12_8 = 570

_JSON = False


def emit(event: str, **fields) -> None:
    if _JSON:
        print(json.dumps({'event': event, **fields}), flush=True)
    else:
        text = fields.get('text') or ', '.join(f'{k}={v}' for k, v in fields.items())
        print(f'[{event}] {text}', flush=True)


def nvidia_smi() -> tuple[list[dict] | None, str | None]:
    """The NVIDIA GPUs nvidia-smi lists, as {'name', 'memory_gb', 'driver'}, and what it said when it failed:
    (None, None) on a computer with no nvidia-smi (no NVIDIA driver), ([], error) when it is there and fails."""
    exe = shutil.which('nvidia-smi')
    if not exe and sys.platform == 'win32':
        for p in (Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'nvidia-smi.exe',
                  Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'NVIDIA Corporation' / 'NVSMI' / 'nvidia-smi.exe'):
            if p.is_file():
                exe = str(p)
                break
    if not exe:
        return None, None
    try:
        r = subprocess.run([exe, '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return [], str(e)
    if r.returncode != 0:
        said = (r.stdout + r.stderr).strip().splitlines()
        return [], said[0] if said else f'exit code {r.returncode}'
    gpus = []
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
        try:
            memory = round(float(parts[1]) / 1024, 1)
        except ValueError:
            memory = None
        gpus.append({'name': parts[0], 'memory_gb': memory, 'driver': parts[2]})
    return gpus, None


def machine(base: Path) -> dict:
    """The device the fill would run on, its memory, and the free disk where the model goes. When PyTorch
    sees no GPU, what nvidia-smi says too (1.3.2), so the reason can name the driver."""
    out = {'device': 'cpu', 'memory_gb': None, 'gpu': None}
    try:
        import torch
        # 1.3.2: PyTorch says why it sees no GPU as a warning (an old driver, most often): kept for the reason.
        with warnings.catch_warnings(record=True) as said:
            warnings.simplefilter('always')
            cuda = torch.cuda.is_available()
        if not cuda and said:
            out['cuda_warning'] = ' '.join(str(said[0].message).split())[:240]
        if cuda:
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
    except Exception as e:   # noqa: BLE001
        out['torch'] = None
        if not (isinstance(e, ModuleNotFoundError) and e.name == 'torch'):
            # 1.3.2: installed but does not load ("DLL load failed", WinError 126, a package it needs); was a traceback
            out['torch_error'] = f'{type(e).__name__}: {e}'[:300]
    if out['device'] == 'cpu' and out.get('torch') and '+cpu' not in str(out['torch']) and sys.platform != 'darwin':
        out['nvidia'], out['nvidia_error'] = nvidia_smi()
    probe = base
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    out['free_gb'] = round(shutil.disk_usage(probe).free / 2**30, 1)
    return out


def no_gpu_reason(m: dict) -> str:
    """Why the fill cannot run here, in words that say what to do (6 Nov 2026: a Python without
    PyTorch was reported as 'no GPU', and the artist went looking for a driver problem)."""
    if m.get('torch_error'):
        # 1.3.2 (end-user setups): PyTorch that is there and does not load. On Windows that is a DLL it needs,
        # the Microsoft Visual C++ runtime most often.
        low = m['torch_error'].lower()
        dll = 'winerror 126' in low or '.dll' in low or 'dll load failed' in low
        return (f"PyTorch is installed in {sys.executable} but does not load ({m['torch_error']})" +
                (': install the Microsoft Visual C++ Redistributable (x64), https://aka.ms/vs/17/release/vc_redist.x64.exe, '
                 'restart the computer, then press Set up automatically again' if dll else
                 ': press Set up automatically again, and if this comes back, send the setup log'))
    if m.get('torch') is None:
        return (f'PyTorch is not installed in {sys.executable}. Install the build for your GPU '
                '(https://pytorch.org/get-started/locally/: CUDA on Windows and Linux), then run this again')
    if m['device'] == 'cpu' and '+cpu' in str(m.get('torch')):
        return (f"PyTorch {m['torch']} is the CPU-only build: install the CUDA build instead "
                '(https://pytorch.org/get-started/locally/)')
    if m['device'] == 'cpu' and m.get('nvidia'):
        # 1.3.2: an NVIDIA GPU is there and PyTorch cannot use it: the driver, almost always.
        g = max(m['nvidia'], key=lambda x: x.get('memory_gb') or 0)
        why = f" (PyTorch: {m['cuda_warning']})" if m.get('cuda_warning') else ''
        return (f"PyTorch {m['torch']} cannot use the {g['name']} with the NVIDIA driver installed ({g['driver']}){why}: "
                f'install the current driver ({DRIVER_FOR_CUDA_12_8} or newer, https://www.nvidia.com/drivers), restart the '
                'computer, then press Set up automatically again')
    if m['device'] == 'cpu' and m.get('nvidia_error'):
        return (f"the NVIDIA driver does not answer (nvidia-smi: {m['nvidia_error']}): reinstall the current driver "
                '(https://www.nvidia.com/drivers), restart the computer, then press Set up automatically again')
    if m['device'] == 'cpu':
        return (f"PyTorch {m['torch']} found no CUDA GPU or Apple GPU: the AI fill needs an NVIDIA GPU with 16 GB "
                'and a current driver (Windows, Linux) or Apple silicon with 16 GB (Mac)')
    return f"the {m.get('gpu') or 'GPU'} has {m['memory_gb']} GB of memory; the AI fill needs 16 GB"


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


def install_packages(missing: list[str] | None = None) -> bool:
    """pip install the packages the fill needs: those of the missing imports (1.3.2: only those, so a Python
    of one's own keeps the versions it has), or all of them."""
    wanted = [req for name, req in REQUIREMENTS if missing is None or name in missing] or PACKAGES
    cmd = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *wanted]
    emit('log', text=' '.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            emit('log', text=line)
    return proc.wait() == 0


def load_manifest(path: Path = MANIFEST) -> dict:
    """The manifest, {} when there is none. One that is there but unreadable is an error
    (code review, 19 Nov 2026): it used to read as {}, and a download then went unchecked."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        raise ValueError(f'{p} cannot be read ({e})') from e


def pinned(entry: dict) -> bool:
    """Whether a manifest entry pins a revision and lists the files to check a download against."""
    return bool(entry.get('revision')) and bool(entry.get('files'))


def model_files(root: Path):
    """(relative posix path, path) of every file of a model folder, huggingface_hub's .cache left out."""
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root)
        if p.is_file() and '.cache' not in rel.parts:
            yield rel.as_posix(), p


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download_revisions(root: Path) -> set[str]:
    """The commits the files of a local_dir download came from: huggingface_hub writes each as the first line
    of .cache/huggingface/download/<file>.metadata once that file is in."""
    d = root / '.cache' / 'huggingface' / 'download'
    revs = set()
    for m in (d.rglob('*.metadata') if d.is_dir() else []):
        try:
            first = m.read_text(encoding='utf-8').splitlines()[0].strip()
        except (OSError, IndexError, UnicodeDecodeError):
            continue
        if re.fullmatch(r'[0-9a-f]{40}', first):
            revs.add(first)
    return revs


def snapshot_revision(root: Path) -> str | None:
    """The commit a local_dir download came from. None when there is none or they disagree."""
    revs = download_revisions(root)
    return revs.pop() if len(revs) == 1 else None


def stopped_download(base: Path, entry: dict) -> list[str]:
    """The manifest's files a download into base has not brought in yet, when base is such a download that
    stopped (1.3.2): huggingface_hub's .cache is in it, every file it has came from the manifest's revision, and
    every file that differs from the manifest is missing, not another size (a file being downloaded waits in
    .cache until it is whole). [] for a whole model, and for a folder that is something else: another revision
    or a model put there by hand is still checked, and refused, as before."""
    if not pinned(entry) or not (base / '.cache' / 'huggingface').is_dir():
        return []
    if not download_revisions(base) <= {entry['revision']}:
        return []
    problems = verify(base, entry, hashes=False)
    return problems if problems and all(p.startswith('missing ') for p in problems) else []


def download_gb(base: Path, entry: dict) -> float:
    """The disk a download into base needs: MODEL_GB for the whole model, its share for what is still to come
    when some of the manifest's files are there already at their size."""
    files = entry.get('files') or {}
    total = sum(int(f.get('size') or 0) for f in files.values())
    if not total:
        return MODEL_GB
    there = sum(int(f.get('size') or 0) for rel, f in files.items()
                if (base / rel).is_file() and (base / rel).stat().st_size == f.get('size'))
    return round(MODEL_GB * (total - there) / total, 1)


def manifest_entry(root: Path, repo_id: str, revision: str | None = None) -> dict:
    return {'repo_id': repo_id, 'revision': revision or snapshot_revision(root),
            'files': {rel: {'size': p.stat().st_size, 'sha256': sha256_file(p)} for rel, p in model_files(root)}}


def verify(root: Path, entry: dict, hashes: bool = True) -> list[str]:
    """What in root does not match the manifest entry: missing files, other sizes, other SHA-256
    (hashes=False: sizes only). Empty when it matches, or when the entry lists no files."""
    problems = []
    for rel, want in sorted((entry.get('files') or {}).items()):
        p = root / rel
        if not p.is_file():
            problems.append(f'missing {rel}')
        elif p.stat().st_size != want.get('size'):
            problems.append(f'{rel}: {p.stat().st_size} bytes, the manifest says {want.get("size")}')
        elif hashes:
            emit('log', text=f'SHA-256 {rel}')
            if sha256_file(p) != want.get('sha256'):
                problems.append(f'{rel}: SHA-256 does not match the manifest')
    return problems


def install_from(src: Path, dest: Path, entry: dict) -> list[str]:
    """Install a model from a folder or a zip (one top folder inside is fine) into dest, offline.
    It is unpacked beside dest first and only moved into place when every file matches the entry;
    a dest already there is kept as <dest>.before_install. The problems, or [] when installed."""
    staging = dest.with_name(dest.name + '.installing')
    if staging.exists():
        shutil.rmtree(staging)
    if src.is_file() and zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                parts = Path(n).parts
                if n.startswith(('/', '\\')) or '..' in parts or (parts and ':' in parts[0]):
                    return [f'{src}: unsafe path in the zip: {n}']
            emit('log', text=f'Unpacking {src}')
            z.extractall(staging)
        kids = list(staging.iterdir())
        root = kids[0] if len(kids) == 1 and kids[0].is_dir() else staging
    elif src.is_dir():
        emit('log', text=f'Copying {src}')
        shutil.copytree(src, staging)
        root = staging
    else:
        return [f'{src} is neither a folder nor a zip']
    problems = verify(root, entry)
    if problems:
        shutil.rmtree(staging, ignore_errors=True)
        return problems
    if dest.exists():
        old = dest.with_name(dest.name + '.before_install')
        if old.exists():
            shutil.rmtree(staging, ignore_errors=True)
            return [f'{dest} is there and so is {old}: remove one of them first']
        dest.rename(old)
        emit('log', text=f'The model that was there is now {old}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    root.rename(dest)
    if root != staging:
        shutil.rmtree(staging, ignore_errors=True)
    return []


def setup_florence(florence: Path, entry: dict, src: str | None,
                   allow_unverified: bool = False) -> tuple[list[str], str | None]:
    """Florence-2 into its folder: from src (a folder or zip), or downloaded at the manifest's
    revision when the folder does not hold every file yet (a download that stopped is resumed);
    then checked against the manifest entry. A download needs a pinned entry, unless allow_unverified.
    (problems, warning): problems end the setup (a model that does not match); a download that
    fails is only a warning, as Suggest a prompt falls back to its presets without it."""
    if src:
        emit('step', step='florence', text=f'Installing Florence-2 from {src} into {florence}')
        return install_from(Path(src).expanduser(), florence, entry), None
    fresh = False
    there = (florence / 'config.json').is_file()
    complete = there and (not entry.get('files') or not verify(florence, entry, hashes=False))
    if not complete:
        if not pinned(entry) and not allow_unverified:
            return [], ('Florence-2 is not pinned in the model manifest, so it was not downloaded (--allow-unverified '
                        'takes the latest unchecked); Suggest a prompt uses its presets until it is')
        rev = entry.get('revision')
        emit('step', step='florence', text=f'Downloading {FLORENCE_ID}{"@" + rev[:12] if rev else ""} '
             f'(MIT, about 0.5 GB) into {florence}')
        try:
            download_model(florence, FLORENCE_ID, rev)
        except Exception as e:   # noqa: BLE001 (network, disk, auth)
            return [], f'Florence-2 was not downloaded ({e}); Suggest a prompt uses its presets until it is'
        fresh = True
    problems = verify(florence, entry, hashes=fresh)
    if problems and fresh:   # never left where the app would load it
        shutil.rmtree(florence, ignore_errors=True)
    return problems, None


def bytes_under(root: Path) -> int:
    """Every byte under root, what is still being downloaded included (huggingface_hub keeps an unfinished file in
    root/.cache/huggingface/download)."""
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


DOWNLOAD_HEARTBEAT_S = 15.0


def download_model(base: Path, repo_id: str = REPO_ID, revision: str | None = None, total_gb: float = 0.0) -> None:
    from huggingface_hub import snapshot_download
    base.mkdir(parents=True, exist_ok=True)
    # 1.3.2 (28 Nov 2026, end-user report: the setup window sat at "Setting up the AI fill..." with nothing in its
    # log): the download says how far it got every DOWNLOAD_HEARTBEAT_S seconds, whether or not huggingface_hub's
    # own progress bars reach the window (they do not when it downloads through hf_xet, or stdout is not a console).
    import threading
    import time
    stop = threading.Event()
    start, t0 = bytes_under(base), time.monotonic()

    def heartbeat() -> None:
        while not stop.wait(DOWNLOAD_HEARTBEAT_S):
            got = bytes_under(base)
            rate = (got - start) / max(1e-6, time.monotonic() - t0) / 1e6
            of = f' of about {total_gb:.0f} GB' if total_gb else ''
            emit('step', step='model', text=f'Downloading {repo_id}: {got / 1e9:.1f} GB{of} in {base} ({rate:.0f} MB/s)')

    beat = threading.Thread(target=heartbeat, daemon=True)
    beat.start()
    try:
        # local_dir: real files in --base (what the app looks for), resumable; the pinned revision.
        snapshot_download(repo_id, local_dir=str(base), revision=revision)
    finally:
        stop.set()
        beat.join(timeout=2)


def main(argv=None) -> int:
    try:
        return _main(argv)
    except Exception as e:   # noqa: BLE001 (1.3.2: whatever stops it, the app hears why; the traceback goes to the log)
        traceback.print_exc()
        emit('done', ok=False, error=f'the setup stopped: {type(e).__name__}: {e}'[:400])
        return 1


def _main(argv=None) -> int:
    global _JSON
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True, help='where Wan2.1-VACE-1.3B-diffusers goes (or already is)')
    ap.add_argument('--json', action='store_true', help='one JSON object per line')
    ap.add_argument('--check-only', action='store_true', help='report what is there and what is missing; change nothing')
    ap.add_argument('--skip-packages', action='store_true')
    ap.add_argument('--allow-small-gpu', action='store_true', help='set up even under 16 GB (the fill may not run)')
    ap.add_argument('--from', dest='from_', metavar='FOLDER_OR_ZIP', help='install Wan2.1-VACE from here, offline')
    ap.add_argument('--florence', action='store_true', help='also set up Florence-2 for Suggest a prompt')
    ap.add_argument('--florence-from', metavar='FOLDER_OR_ZIP', help='install Florence-2 from here, offline')
    ap.add_argument('--florence-only', action='store_true', help='set up Florence-2 alone (no GPU check, packages or Wan)')
    ap.add_argument('--verify', action='store_true', help="check every file's SHA-256 against the manifest")
    ap.add_argument('--allow-unverified', action='store_true', help='install or download a model the manifest does not pin (unchecked)')
    ap.add_argument('--manifest', default=str(MANIFEST), help='the manifest to read or write')
    ap.add_argument('--make-manifest', action='store_true',
                    help='write the manifest from the models in --base and its Florence-2 folder, then stop')
    a = ap.parse_args(argv)
    _JSON = a.json
    base = Path(a.base).expanduser().resolve()
    florence = base.parent / FLORENCE_DIR
    try:
        manifest = load_manifest(Path(a.manifest))
    except ValueError as e:
        # Unreadable: an error, unless the run writes a new one or asked for unchecked models.
        if not a.make_manifest and not a.allow_unverified:
            emit('done', ok=False, error=str(e) + ' (--allow-unverified to go on without it)', base=str(base))
            return 1
        manifest = {}

    if a.make_manifest:
        if not have_model(base):
            emit('done', ok=False, error=f'no Wan2.1-VACE model in {base}', base=str(base))
            return 1
        emit('step', step='manifest', text=f'Hashing {base}')
        out = {'schema': 'rudra_model_manifest_v1', 'wan': manifest_entry(base, REPO_ID)}
        if (florence / 'config.json').is_file():
            emit('step', step='manifest', text=f'Hashing {florence}')
            out['florence'] = manifest_entry(florence, FLORENCE_ID)
        Path(a.manifest).write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
        missing_rev = [k for k in ('wan', 'florence') if k in out and not out[k]['revision']]
        emit('done', ok=True, manifest=a.manifest, files=sum(len(out[k]['files']) for k in ('wan', 'florence') if k in out),
             no_revision=missing_rev)
        return 0

    for key, src in (('wan', a.from_), ('florence', a.florence_from)):
        if src and not (manifest.get(key) or {}).get('files') and not a.allow_unverified:
            emit('done', ok=False, error=f'{a.manifest} has no hashes for {key}: nothing to check the install against '
                 '(run --make-manifest on a known-good model, or pass --allow-unverified)', base=str(base))
            return 1

    if a.florence_only:
        problems, warning = setup_florence(florence, manifest.get('florence') or {}, a.florence_from, a.allow_unverified)
        if problems:
            emit('done', ok=False, error='Florence-2 does not match the manifest: ' + '; '.join(problems[:5]), base=str(base))
            return 1
        if warning:
            emit('done', ok=False, error=warning, base=str(base))
            return 1
        emit('done', ok=True, base=str(base), florence=str(florence))
        return 0

    # 1.3.2: said before the checks, which take a while (PyTorch loads, every package is imported) and print nothing.
    emit('step', step='check', text='Checking the GPU, the disk and the AI fill\'s packages')
    m = machine(base)
    wan = manifest.get('wan') or {}
    model_there = have_model(base)
    # 1.3.2 (end-user report, 27 Nov 2026): a download that stopped (18 GB of it; the window closed, the network
    # dropped) has its small files in first, transformer/config.json among them, and that was taken for the
    # model: every run after it skipped the download and then refused the folder for the files it lacked. Such a
    # folder is downloaded again now, which fetches only what is still missing (huggingface_hub keeps the rest).
    resume = stopped_download(base, wan) if model_there and not a.from_ else []
    if resume:
        model_there = False
        emit('log', text=f'The download into {base} stopped before it finished ({len(resume)} files to come): '
                         'it carries on from there')
    need_disk = 0.0 if model_there else download_gb(base, wan)
    enough_mem = m['memory_gb'] is not None and m['memory_gb'] >= MIN_MEMORY_GB
    enough_disk = m['free_gb'] >= need_disk
    missing = missing_packages()
    emit('check', **m, model=model_there, base=str(base), missing_packages=missing,
         florence=(florence / 'config.json').is_file(), pinned=bool((manifest.get('wan') or {}).get('revision')),
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
        if not install_packages(missing):
            emit('done', ok=False, error='pip install failed (see the log)', base=str(base))
            return 1
        importlib.invalidate_caches()
        still = missing_packages()
        if still:
            emit('done', ok=False, error='still missing after the install: ' + ', '.join(still), base=str(base))
            return 1

    fresh = False
    if a.from_:
        emit('step', step='model', text=f'Installing {REPO_ID} from {a.from_} into {base}')
        problems = install_from(Path(a.from_).expanduser(), base, wan)
        if problems:
            emit('done', ok=False, error='the model was not installed: ' + '; '.join(problems[:5]), base=str(base))
            return 1
    elif not model_there:
        if not pinned(wan) and not a.allow_unverified:
            emit('done', ok=False, error=f'{a.manifest} does not pin {REPO_ID} (a revision and its files\' SHA-256): '
                 'not downloaded unchecked (run --make-manifest on a known-good model, or pass --allow-unverified)',
                 base=str(base))
            return 1
        rev = wan.get('revision')
        emit('step', step='model', text=f'Downloading {REPO_ID}{"@" + rev[:12] if rev else ""} '
             f'(about {max(1.0, need_disk):.0f} GB{" still to come" if resume else ""}) into {base}')
        try:
            download_model(base, REPO_ID, rev, total_gb=sum(int(f.get('size') or 0) for f in (wan.get('files') or {}).values()) / 1e9 or MODEL_GB)
        except Exception as e:   # noqa: BLE001 (network, disk, auth: all end the same way)
            traceback.print_exc()
            emit('done', ok=False, error=f'the model download stopped ({str(e)[:300]}). Check the internet connection '
                 '(huggingface.co), then press Set up automatically again: it carries on where it stopped',
                 base=str(base))
            return 1
        fresh = True
    if wan.get('files') and not a.from_:
        emit('step', step='verify', text='Checking the model against the manifest' + (' (SHA-256)' if fresh or a.verify else ''))
        problems = verify(base, wan, hashes=fresh or a.verify)
        if problems:
            emit('done', ok=False, error='the model does not match the manifest: ' + '; '.join(problems[:5]) +
                 ('' if fresh else f'. Move {base} aside (or delete it), then press Set up automatically again: it '
                                   'downloads the model RUDRA was tested with'), base=str(base))
            return 1

    florence_warning = None
    if a.florence or a.florence_from:
        problems, florence_warning = setup_florence(florence, manifest.get('florence') or {}, a.florence_from,
                                                    a.allow_unverified)
        if problems:
            emit('done', ok=False, error='Florence-2 does not match the manifest: ' + '; '.join(problems[:5]), base=str(base))
            return 1
        if florence_warning:
            emit('log', text=florence_warning)

    emit('step', step='verify', text='Checking the model')
    try:
        cfg = json.loads((base / 'transformer' / 'config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        emit('done', ok=False, error=f'the model in {base} is incomplete: {e}', base=str(base))
        return 1
    emit('log', text=f"transformer {cfg.get('_class_name')}, {cfg.get('num_layers')} layers, vace {cfg.get('vace_layers')}")
    emit('done', ok=True, base=str(base), device=m['device'],
         florence=str(florence) if (florence / 'config.json').is_file() else None, florence_warning=florence_warning)
    return 0


if __name__ == '__main__':
    sys.exit(main())
