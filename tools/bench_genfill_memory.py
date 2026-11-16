"""How much GPU memory AI fill needs, in each memory mode (16 Nov 2026, RUDRA 1.2.5).

Runs the fill (python -m rudra.genfill.run, the same command the app runs) on the first frames of a
master, once as it was gated (mode off, no cap) and then in each mode under each memory cap, and
compares every output frame with the gated one. A cap makes PyTorch on this card stop where a smaller
card would (RUDRA_GENFILL_MEMORY_CAP_GB, run.py), so a 16 GB or larger card can stand in for 10 and 12 GB.

    python tools/bench_genfill_memory.py --master D:/shots/09V_master --caps 12,10 --modes swap,tile,on
    python tools/bench_genfill_memory.py --master ... --base ... --lora ...   # parts not found by themselves

Writes reports/genfill_memory_<stamp>/report.json and report.md, and keeps the fills beside them.

The model must actually run: a fill only generates where the SDR clipped (code 250 and up by default).
When the gated fill generates nothing in these frames, it is run again with --tau-high and --blend-high
lowered (200 by default, --fallback-tau), and every run uses that: the memory and the seams are what is
measured here, not the look. The blend has to come down too: with --blend-high 255 the fill is written
only over fully clipped pixels, and a shot without any keeps the master unchanged, so every mode would
compare as identical (the 16 Nov 2026 run did). A run in which the model did not run, or wrote no pixel,
is marked so and counts for nothing.

Reading it: 'swap' must match the gated fill exactly (max difference 0). 'tile' and 'on' differ a little
at the VAE's tile seams; the difference is given in ACEScct code values over the whole frame, and the
frames are kept so the seams can be looked at. A card of N GB is offered a mode only when that mode
finished under a cap of N - 1 GB (the 1 GB is the CUDA context and the desktop, which the cap does not
count) and its difference is within the limit agreed for AI fill's gates.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
V3_SHA256 = '9fff7eb8492ae62619a126d302d9c940403b2e4aa03e02daf79d98142008238a'   # native/app/genfill_job.hpp


def models_root() -> Path:
    """The app's model download folder: <AppLocalData>/models (QSettings models/downloadRoot aside)."""
    if sys.platform == 'win32':
        return Path(os.environ.get('LOCALAPPDATA', '')) / 'FXTD Studios' / 'RUDRA' / 'models'
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'FXTD Studios' / 'RUDRA' / 'models'
    return Path.home() / '.local' / 'share' / 'FXTD Studios' / 'RUDRA' / 'models'


def find_base() -> Path | None:
    for p in [models_root() / 'Wan2.1-VACE-1.3B-diffusers', REPO / 'Wan2.1-VACE-1.3B-diffusers',
              Path('G:/datasets/models/Wan2.1-VACE-1.3B-diffusers')]:
        if (p / 'transformer' / 'config.json').exists():
            return p
    hub = Path(os.environ.get('HF_HUB_CACHE') or Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')) / 'hub')
    snaps = sorted((hub / 'models--Wan-AI--Wan2.1-VACE-1.3B-diffusers' / 'snapshots').glob('*/transformer/config.json'))
    return snaps[-1].parent.parent if snaps else None


def find_v3_lora() -> Path | None:
    from rudra.genfill.model import sha256_file
    for p in sorted(models_root().rglob('*.safetensors')):
        if p.stat().st_size > 10_000_000 and sha256_file(p) == V3_SHA256:
            return p
    return None


def clip_of(master: Path, frames: int, into: Path, glob: str) -> Path:
    """The master's first `frames` frames, copied (the fill writes a new folder; the master is not touched)."""
    src = sorted(master.glob(glob))[:frames]
    if not src:
        raise SystemExit(f'no {glob} frames in {master}')
    into.mkdir(parents=True, exist_ok=True)
    for f in src:
        shutil.copy2(f, into / f.name)
    return into


def run_fill(python: str, base: Path, lora: Path, clip: Path, out: Path, mode: str, cap: float, extra: list[str]) -> dict:
    env = dict(os.environ)
    env.pop('RUDRA_GENFILL_MEMORY_CAP_GB', None)
    if cap > 0:
        env['RUDRA_GENFILL_MEMORY_CAP_GB'] = str(cap)
    env['PYTHONUNBUFFERED'] = '1'
    cmd = [python, '-m', 'rudra.genfill.run', '--base', str(base), '--lora', str(lora), '--master', str(clip),
           '--out', str(out), '--low-memory', mode, '--json'] + extra
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    seconds = time.time() - t0
    events = []
    for line in p.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    done = next((e for e in events if e.get('event') == 'done'), None)
    err = next((e for e in events if e.get('event') == 'error'), None)
    ready = next((e for e in events if e.get('event') == 'ready'), {})
    text = (err or {}).get('error', '') + ' ' + (err or {}).get('detail', '') + ' ' + p.stderr[-2000:]
    oom = 'OutOfMemory' in text or 'out of memory' in text.lower()
    r = {'mode': mode, 'cap_gb': cap, 'ok': bool(done), 'out_of_memory': oom and not done,
         'seconds': round(seconds, 1), 'load_seconds': ready.get('load_seconds')}
    if done:
        r['memory'] = done.get('memory', {})
        r['windows'] = done.get('windows')
        r['windows_generated'] = done.get('windows_generated', 0)
        r['generated_fraction'] = done.get('generated_fraction')
        r['generated_pixels'] = done.get('generated_pixels', 0)
        r['model_ran'] = bool(r['windows_generated'])
        if r['windows']:
            r['seconds_per_window'] = round((seconds - (ready.get('load_seconds') or 0)) / r['windows'], 1)
    else:
        r['error'] = ((err or {}).get('error', '') + ': ' + (err or {}).get('detail', '')).strip(': ') or p.stderr[-600:]
    return r


def compare(gated: Path, other: Path) -> dict:
    """Every EXR frame of `other` against the gated fill's, in ACEScct code values (0..1 over the
    working range): the largest difference anywhere, the mean, and the share of pixels past 1/1023."""
    from rudra.genfill import encoding
    from rudra.genfill.run import read_frame
    worst, means, past = 0.0, [], []
    names = sorted(p.name for p in gated.glob('*.exr'))
    if not names:
        return {'frames': 0}
    for n in names:
        b = other / n
        if not b.exists():
            return {'frames': 0, 'missing': n}
        x = encoding.acescct(np.clip(read_frame(gated / n)['ap1'], 0, None))
        y = encoding.acescct(np.clip(read_frame(b)['ap1'], 0, None))
        d = np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))
        worst = max(worst, float(d.max()))
        means.append(float(d.mean()))
        past.append(float((d.max(-1) > 1.0 / 1023).mean()))
    return {'frames': len(names), 'max_diff_cct': round(worst, 6), 'mean_diff_cct': round(float(np.mean(means)), 8),
            'pixels_past_one_10bit_code': round(float(np.mean(past)), 6), 'identical': worst == 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--master', required=True, help='a master folder of EXR frames (the fill reads these)')
    ap.add_argument('--glob', default='*.exr')
    ap.add_argument('--frames', type=int, default=33, help='how many frames (33: one window; 58: two)')
    ap.add_argument('--base', default='', help='Wan2.1-VACE-1.3B-diffusers (default: where the app keeps it)')
    ap.add_argument('--lora', default='', help='the AI fill v3 LoRA (default: the verified one in the model folder)')
    ap.add_argument('--caps', default='12,10', help='card sizes to stand in for, GB (each run under size - 1)')
    ap.add_argument('--modes', default='swap,tile,on')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--out', default='', help='default: reports/genfill_memory_<stamp>')
    ap.add_argument('--extra', default='', help="more arguments for rudra.genfill.run, e.g. '--focus region'")
    ap.add_argument('--fallback-tau', type=int, default=200,
                    help='when the gated fill generates nothing, run everything again with --tau-high this (0: stop instead)')
    a = ap.parse_args(argv)

    base = Path(a.base) if a.base else find_base()
    if not base or not (base / 'transformer' / 'config.json').exists():
        raise SystemExit('no Wan2.1-VACE-1.3B-diffusers found: give --base')
    lora = Path(a.lora) if a.lora else find_v3_lora()
    if not lora or not lora.exists():
        raise SystemExit('no AI fill v3 LoRA found: give --lora')
    out = Path(a.out) if a.out else REPO / 'reports' / f"genfill_memory_{datetime.now():%Y-%m-%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    clip = clip_of(Path(a.master), a.frames, out / 'clip', a.glob)
    extra = a.extra.split() if a.extra else []
    cards = [float(c) for c in a.caps.split(',') if c.strip()]
    modes = [m.strip() for m in a.modes.split(',') if m.strip()]

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU'
    total = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0
    print(f'{gpu}, {total:.1f} GB · base {base} · LoRA {lora} · {a.frames} frames', flush=True)

    runs = []
    print('gated fill (off, no cap)…', flush=True)
    gated = run_fill(a.python, base, lora, clip, out / 'off_nocap', 'off', 0, extra)
    gated['card_gb'] = None
    if gated['ok'] and not gated['model_ran']:
        # Nothing clipped at the fill's threshold in these frames: the model never ran, so nothing was measured.
        if not a.fallback_tau:
            raise SystemExit('the model did not run: no clipped highlights in these frames. Choose another shot, '
                             'or give --fallback-tau 200')
        print(f'  nothing to fill at --tau-high 250 in these frames: again with --tau-high and --blend-high {a.fallback_tau}',
              flush=True)
        extra = extra + ['--tau-high', str(a.fallback_tau), '--blend-high', str(a.fallback_tau)]
        gated = run_fill(a.python, base, lora, clip, out / f'off_nocap_tau{a.fallback_tau}', 'off', 0, extra)
        gated['card_gb'] = None
        if gated['ok'] and (not gated['model_ran'] or not gated['generated_pixels']):
            raise SystemExit(f'the fill wrote nothing even at {a.fallback_tau}: choose a shot with bright highlights')
    elif gated['ok'] and not gated['generated_pixels']:
        raise SystemExit('the model ran but the fill wrote no pixel (nothing at --blend-high): the modes would all compare '
                         "as identical. Give --extra '--blend-high 200' or choose a shot with clipped highlights")
    gated_dir = out / ('off_nocap' if '--tau-high' not in extra else f'off_nocap_tau{a.fallback_tau}')
    runs.append(gated)
    if not gated['ok']:
        print(f"  the gated fill failed: {gated.get('error')}", flush=True)
    else:
        print(f"  ok · model ran in {gated['windows_generated']} window(s), wrote {gated['generated_pixels']} pixels "
              f"({100 * (gated['generated_fraction'] or 0):.3f} %) · peak {gated['memory'].get('peak_reserved_gb')} GB reserved", flush=True)
    for card in cards:
        for mode in ['off'] + modes:
            cap = card - 1.0
            print(f'{mode} as a {card:g} GB card (cap {cap:g} GB)…', flush=True)
            r = run_fill(a.python, base, lora, clip, out / f'{mode}_{card:g}gb', mode, cap, extra)
            r['card_gb'] = card
            if r['ok'] and gated['ok']:
                r['vs_gated'] = compare(gated_dir, out / f'{mode}_{card:g}gb')
            runs.append(r)
            print('  ' + ('ok' if r['ok'] else 'out of memory' if r['out_of_memory'] else 'failed') +
                  (f" · peak {r['memory'].get('peak_reserved_gb')} GB reserved · {r['seconds']} s" if r['ok'] else ''), flush=True)

    report = {'gpu': gpu, 'gpu_memory_gb': round(total, 1), 'frames': a.frames, 'base': str(base), 'lora': str(lora),
              'master': str(a.master), 'extra': extra, 'gated_dir': str(gated_dir), 'runs': runs}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = [f'# AI fill GPU memory · {gpu} ({total:.1f} GB) · {a.frames} frames', '',
             f"Extra arguments: {' '.join(extra) or 'none'}", '',
             '| Card | Mode | Result | Model ran | Peak reserved GB | Seconds | s / window | Max diff (CCT) | Mean diff | Identical |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in runs:
        v = r.get('vs_gated', {})
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            'gated' if r['card_gb'] is None else f"{r['card_gb']:g} GB", r['mode'],
            'ok' if r['ok'] else 'out of memory' if r['out_of_memory'] else 'failed',
            ('yes' if r.get('model_ran') else 'NO') if r['ok'] else '',
            r.get('memory', {}).get('peak_reserved_gb', ''), r['seconds'], r.get('seconds_per_window', ''),
            v.get('max_diff_cct', ''), v.get('mean_diff_cct', ''), v.get('identical', '')))
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nreport: {out / "report.md"}', flush=True)
    return 0 if gated['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
