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
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ID = 'Wan-AI/Wan2.1-VACE-1.3B-diffusers'
FLORENCE_ID = 'florence-community/Florence-2-base'   # Suggest a prompt (prompts.py); about 0.5 GB
FLORENCE_DIR = 'Florence-2-base'                     # beside the Wan folder
MANIFEST = Path(__file__).with_name('model_manifest.json')
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


def load_manifest(path: Path = MANIFEST) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


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


def snapshot_revision(root: Path) -> str | None:
    """The commit a local_dir download came from: huggingface_hub writes it as the first line of
    .cache/huggingface/download/<file>.metadata. None when there is none or they disagree."""
    d = root / '.cache' / 'huggingface' / 'download'
    revs = set()
    for m in (d.rglob('*.metadata') if d.is_dir() else []):
        try:
            first = m.read_text(encoding='utf-8').splitlines()[0].strip()
        except (OSError, IndexError, UnicodeDecodeError):
            continue
        if re.fullmatch(r'[0-9a-f]{40}', first):
            revs.add(first)
    return revs.pop() if len(revs) == 1 else None


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


def setup_florence(florence: Path, entry: dict, src: str | None) -> tuple[list[str], str | None]:
    """Florence-2 into its folder: from src (a folder or zip), or downloaded at the manifest's
    revision when the folder has no model yet; then checked against the manifest entry.
    (problems, warning): problems end the setup (a model that does not match); a download that
    fails is only a warning, as Suggest a prompt falls back to its presets without it."""
    if src:
        emit('step', step='florence', text=f'Installing Florence-2 from {src} into {florence}')
        return install_from(Path(src).expanduser(), florence, entry), None
    fresh = False
    if not (florence / 'config.json').is_file():
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


def download_model(base: Path, repo_id: str = REPO_ID, revision: str | None = None) -> None:
    from huggingface_hub import snapshot_download
    base.mkdir(parents=True, exist_ok=True)
    # local_dir: real files in --base (what the app looks for), resumable; the pinned revision.
    snapshot_download(repo_id, local_dir=str(base), revision=revision)


def main(argv=None) -> int:
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
    ap.add_argument('--allow-unverified', action='store_true', help='install from a folder or zip with no manifest entry')
    ap.add_argument('--manifest', default=str(MANIFEST), help='the manifest to read or write')
    ap.add_argument('--make-manifest', action='store_true',
                    help='write the manifest from the models in --base and its Florence-2 folder, then stop')
    a = ap.parse_args(argv)
    _JSON = a.json
    base = Path(a.base).expanduser().resolve()
    florence = base.parent / FLORENCE_DIR
    manifest = load_manifest(Path(a.manifest))

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
        problems, warning = setup_florence(florence, manifest.get('florence') or {}, a.florence_from)
        if problems:
            emit('done', ok=False, error='Florence-2 does not match the manifest: ' + '; '.join(problems[:5]), base=str(base))
            return 1
        if warning:
            emit('done', ok=False, error=warning, base=str(base))
            return 1
        emit('done', ok=True, base=str(base), florence=str(florence))
        return 0

    m = machine(base)
    model_there = have_model(base)
    need_disk = 0.0 if model_there else MODEL_GB
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
        if not install_packages():
            emit('done', ok=False, error='pip install failed (see the log)', base=str(base))
            return 1
        importlib.invalidate_caches()
        still = missing_packages()
        if still:
            emit('done', ok=False, error='still missing after the install: ' + ', '.join(still), base=str(base))
            return 1

    wan = manifest.get('wan') or {}
    fresh = False
    if a.from_:
        emit('step', step='model', text=f'Installing {REPO_ID} from {a.from_} into {base}')
        problems = install_from(Path(a.from_).expanduser(), base, wan)
        if problems:
            emit('done', ok=False, error='the model was not installed: ' + '; '.join(problems[:5]), base=str(base))
            return 1
    elif not model_there:
        rev = wan.get('revision')
        emit('step', step='model', text=f'Downloading {REPO_ID}{"@" + rev[:12] if rev else ""} '
             f'(about {MODEL_GB:.0f} GB) into {base}')
        try:
            download_model(base, REPO_ID, rev)
        except Exception as e:   # noqa: BLE001 (network, disk, auth: all end the same way)
            emit('done', ok=False, error=f'model download: {e}', base=str(base))
            return 1
        fresh = True
    if wan.get('files') and not a.from_:
        emit('step', step='verify', text='Checking the model against the manifest' + (' (SHA-256)' if fresh or a.verify else ''))
        problems = verify(base, wan, hashes=fresh or a.verify)
        if problems:
            emit('done', ok=False, error='the model does not match the manifest: ' + '; '.join(problems[:5]), base=str(base))
            return 1

    florence_warning = None
    if a.florence or a.florence_from:
        problems, florence_warning = setup_florence(florence, manifest.get('florence') or {}, a.florence_from)
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
