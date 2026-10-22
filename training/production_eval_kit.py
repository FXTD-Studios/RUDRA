"""Freeze validation inputs and run strict native-resolution paired evaluation.

Preparation is CPU/IO only. Execution refuses to compete with a live trainer.
This gate measures validation PU-PSNR, never product release readiness.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def select_validation(rows):
    if len({r['asset_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate asset IDs')
    selected = [r for r in rows if r.get('split') == 'val']
    if not selected or any(not r.get('scene_id') for r in selected):
        raise ValueError('Missing validation scenes')
    scenes = {r['scene_id'] for r in selected}
    if scenes & {r.get('scene_id') for r in rows if r.get('split') != 'val'}:
        raise ValueError('Validation scene overlaps another split')
    if len(scenes) < 2:
        raise ValueError('Need multiple independent scene groups')
    return selected


def prepare(manifest, out):
    rows = select_validation([json.loads(l) for l in manifest.read_text().splitlines() if l.strip()])
    if out.exists():
        raise ValueError('Use a new kit directory; frozen kits are immutable')
    for r in rows:
        for kind in ('sdr', 'hdr'):
            if digest(r[kind + '_path']) != r[kind + '_sha256']:
                raise ValueError(f"Media hash mismatch: {r['asset_id']} {kind}")
    out.mkdir(parents=True)
    frozen = out / 'validation.jsonl'
    frozen.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    contract = dict(source_manifest=str(manifest), source_manifest_sha256=digest(manifest),
        validation_sha256=digest(frozen), frames=len(rows), scenes=len({r['scene_id'] for r in rows}),
        source_counts=dict(Counter(r.get('source_id', 'unknown') for r in rows)),
        sdr_kind_counts=dict(Counter(r.get('sdr_kind', 'unknown') for r in rows)),
        category_labels_missing=sum(not r.get('category') for r in rows),
        dimensions=dict(Counter(f"{r.get('width')}x{r.get('height')}" for r in rows)),
        conditions=['clean', 'hard'], nits_scale=203, max_side=0, tile_size=512,
        baseline='bt1886', precision='fp32', gate=dict(clean_lower_ci_min=-0.1, hard_lower_ci_min=0),
        release_authorized=False, purpose='Validation selection only; final test untouched',
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in [
            Path(__file__), ROOT/'training/export_bench_pairs.py', ROOT/'training/paired_gate.py',
            ROOT/'rudra/delivery/bench.py', ROOT/'rudra/inference.py', ROOT/'rudra/sdr2hdr.py']})
    (out/'contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    return contract


def strict_scores(results, expected):
    scores = {}
    for r in results:
        key = (r['clip'], r['frame'])
        if key in scores or key not in expected:
            raise ValueError('Duplicate or unexpected evaluated frame')
        value = r.get('pu_psnr_db')
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Missing/nonfinite PU metric')
        scores[key] = {'pu_psnr_db': value}
    if scores.keys() != expected.keys():
        raise ValueError('Incomplete evaluation coverage')
    return scores


def evaluate(kit, checkpoint, out):
    from training.run_next_hdr_commercial import active_trainers
    if active_trainers():
        raise RuntimeError('Training is active; evaluation stays deferred')
    from training.export_bench_pairs import slug, UNSAFE
    from training.paired_gate import compare
    contract = json.loads((kit/'contract.json').read_text())
    manifest = kit/'validation.jsonl'
    if digest(manifest) != contract['validation_sha256']:
        raise ValueError('Frozen validation manifest changed')
    for name, value in contract['code_sha256'].items():
        if digest(ROOT/name) != value:
            raise ValueError(f'Pinned evaluation code changed: {name}; prepare a new kit')
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    for r in rows:
        for kind in ('sdr', 'hdr'):
            if digest(r[kind+'_path']) != r[kind+'_sha256']:
                raise ValueError('Evaluation input changed')
    expected = {(slug(r['scene_id']), UNSAFE.sub('_', r['asset_id'])): r['scene_id'] for r in rows}
    if len(expected) != len(rows):
        raise ValueError('Export filename collision')
    if out.exists():
        raise ValueError('Use an empty output path; never reuse model outputs')
    out.mkdir(parents=True)
    before = digest(checkpoint)
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=before,
                  contract_sha256=digest(kit/'contract.json'), conditions={}, release_authorized=False)
    for condition in contract['conditions']:
        dest = out/condition
        command = [sys.executable, str(ROOT/'training/export_bench_pairs.py'), '--checkpoint', str(checkpoint),
            '--manifest', str(manifest), '--split', 'val', '--out', str(dest), '--condition', condition,
            '--max-side', '0', '--tile-size', '512', '--tile-overlap', '64', '--precision', 'fp32',
            '--baseline-curve', 'bt1886', '--picker-baseline', 'bt1886', '--no-baseline']
        subprocess.run(command, cwd=ROOT, check=True)
        export = json.loads((dest/'export.json').read_text())
        if export['skipped'] or export['frames'] != len(rows):
            raise ValueError('Exporter skipped evaluation frames')
        for name in ('test', 'baseline_bt1886'):
            subprocess.run([sys.executable, '-m', 'rudra.delivery.cli', 'bench', str(dest),
                '--test-dir', name, '--nits-scale', '203', '--output', str(dest/f'{name}.json')], cwd=ROOT, check=True)
        a, b = [strict_scores(json.loads((dest/f'{name}.json').read_text())['results'], expected)
                for name in ('test', 'baseline_bt1886')]
        metrics = compare(a, b, metrics=('pu_psnr_db',), require_same_frames=True, scenes=expected)
        lower = metrics['pu_psnr_db'].get('ci95', [float('-inf')])[0]
        passed = lower >= -.1 if condition == 'clean' else lower > 0
        report['conditions'][condition] = dict(metrics=metrics, passed=passed)
    if before != digest(checkpoint):
        raise ValueError('Checkpoint changed during evaluation')
    report['validation_pu_gate_passed'] = all(r['passed'] for r in report['conditions'].values())
    report['not_measured'] = ['Temporal stability', 'Independent real SDR', 'Category coverage',
                            'Commercial release clearance', 'HDR display review']
    (out/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--manifest', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    run = sub.add_parser('evaluate')
    run.add_argument('--kit', type=Path, required=True)
    run.add_argument('--checkpoint', type=Path, required=True)
    run.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = prepare(args.manifest, args.out) if args.action == 'prepare' else evaluate(args.kit, args.checkpoint, args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
