"""Read-only candidate screening; never promotes weights or authorizes release."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def screen(metrics):
    keys = ('clean_pu_gain_db', 'hard_pu_gain_db')
    values = [metrics.get(k) for k in keys]
    valid = all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
    # A point-estimate screen, not statistical or perceptual proof.
    return valid and all(v > 0 for v in values)


def read_evaluations(raw):
    records = []
    lines = raw.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A live writer may leave one incomplete final line. Interior
            # corruption and complete malformed lines must not be hidden.
            if i == len(lines) - 1 and not line.endswith(b'\n'):
                break
            raise
        if 'eval' in row:
            records.append(row)
    return records


def audit(run):
    import torch
    path = run / 'checkpoints' / 'train.jsonl'
    raw = path.read_bytes()
    evaluations = read_evaluations(raw)
    by_step = {r['step']: r['eval'] for r in evaluations}
    candidates = []
    for checkpoint in sorted((run / 'checkpoints').glob('*.pt')):
        # Only local training artifacts, CPU-only restricted unpickler.
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        step = payload.get('step')
        metrics = by_step.get(step, {})
        candidates.append(dict(path=str(checkpoint), step=step,
            clean_pu_gain_db=metrics.get('clean_pu_gain_db'),
            hard_pu_gain_db=metrics.get('hard_pu_gain_db'),
            passes_point_estimate_screen=screen(metrics),
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        del payload
    passing = sorted((c for c in candidates if c['passes_point_estimate_screen']),
                     key=lambda c: min(c['clean_pu_gain_db'], c['hard_pu_gain_db']), reverse=True)
    return dict(run=str(run), log_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        completed_evaluations=len(evaluations),
        latest_evaluation=evaluations[-1] if evaluations else None,
        candidates=candidates, shortlist=passing,
        screening_rule='Both clean and hard PU-PSNR gain point estimates strictly positive; rank by the smaller gain.',
        release_authorized=False,
        remaining=['Scene-grouped confidence intervals and category regressions',
                   'Independent real-SDR and HDR reference review',
                   'Full-resolution and temporal evaluation',
                   'Checkpoint lineage, dependency and source rights review',
                   'Calibrated HDR display and delivery verification'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        parser.error('Write the audit outside the training run to keep it immutable.')
    report = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(report=str(args.output), saved_candidates=len(report['candidates']),
                          passing_candidates=len(report['shortlist']), release_authorized=False)))


if __name__ == '__main__':
    main()
