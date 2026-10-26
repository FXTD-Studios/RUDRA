"""Full validation contract and reference-gradient audit, with saved predictions."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.banding_diagnostics import reference_gradient_errors
from training.contract_checks import Runner, evaluate_pair, load_reference
from training.sdr2hdr_dataset import load_rgb


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def summarize(rows):
    """Equal-pair means, with explicit missing counts and per-push strata."""
    result = {}
    for group in ('all', 'none', 'codec'):
        selected = [r for r in rows if group == 'all' or
                    (r['codec'] == 'none') == (group == 'none')]
        methods = {}
        for label in rows[0]['methods']:
            contracts = [r['methods'][label]['contract'] for r in selected]
            contract = {}
            for key in ('r_within_1_code', 'h_abs_log2', 's_visible_lift',
                        'f_log2_l1', 'banding_excess_worst', 'clip_at_ceiling'):
                vals = [c[key] for c in contracts if c.get(key) is not None]
                contract[key] = {'mean': float(np.mean(vals)) if vals else None,
                                 'pairs': len(vals)}
            pushes = {}
            for push in (-3, -2, -1, 0, 1, 2, 3):
                items = [p for r in selected for p in
                         r['methods'][label]['reference']['pushes'] if p['push'] == push]
                scores = {}
                for key in ('code_rmse', 'gradient_rmse', 'curvature_rmse'):
                    vals = [p[key] for p in items if p[key] is not None]
                    scores[key] = {'mean': float(np.mean(vals)) if vals else None,
                                   'pairs': len(vals)}
                pushes[str(push)] = scores
            methods[label] = {'contract': contract, 'pushes': pushes}
        result[group] = {'pairs': len(selected),
                         'scenes': len({r['scene_id'] for r in selected}), 'methods': methods}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.out.exists():
        raise ValueError('use a new output directory')
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r['split'] == 'val' and not r.get('excluded', False)]
    if not rows or len({r['asset_id'] for r in rows}) != len(rows):
        raise ValueError('validation must be nonempty and unique')
    if any(r.get('declared_view') != 'bt1886' or
           r.get('reconstruction_policy') != 'observed_clip_v1' for r in rows):
        raise ValueError('this diagnostic requires corrected BT1886 validation')
    if args.label == 'baseline_bt1886':
        raise ValueError('reserved baseline label')
    # Verify every saved media identity before starting expensive inference.
    for row in rows:
        for kind in ('sdr', 'hdr', 'mask'):
            if sha(row[kind + '_path']) != row[kind + '_sha256']:
                raise ValueError('media identity mismatch: ' + row['asset_id'])
    identity = {'manifest': str(args.manifest), 'manifest_sha256': sha(args.manifest),
                'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
                'label': args.label, 'view': 'bt1886', 'policy': 'observed_clip_v1',
                'device': args.device, 'release_authorized': False}
    runner = Runner({args.label: args.checkpoint}, args.device, 'bt1886', policy='observed_clip_v1')
    args.out.mkdir(parents=True)
    (args.out / 'identity.json').write_text(json.dumps(identity, indent=2))
    per = []
    for index, row in enumerate(rows):
        source = load_rgb(row['sdr_path'], hdr=False)
        target = load_reference(row['hdr_path'])
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        outputs = runner.run(source)
        contracts = evaluate_pair(outputs, source, target, mask, 'bt1886', 'observed_clip_v1')
        item = {key: row[key] for key in ('asset_id', 'scene_id')}
        item['codec'] = row['sdr_codec']
        item['methods'] = {}
        for label, output in outputs.items():
            item['methods'][label] = {'contract': contracts[label],
                                      'reference': reference_gradient_errors(output, target)}
        item['prediction'] = f'{index:03d}_prediction.npy'
        np.save(args.out / item['prediction'], outputs[args.label])
        (args.out / f'{index:03d}_metrics.json').write_text(json.dumps(item, indent=2, allow_nan=False))
        per.append(item)
        print(f'[dequant audit] {index + 1}/{len(rows)} {row["asset_id"]}', flush=True)
    report = {**identity, 'pairs': len(per), 'summary': summarize(per), 'per_pair': per}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
