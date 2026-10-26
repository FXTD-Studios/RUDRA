"""Independent source-defined shadow colour diagnostics on saved audit outputs.

Supplementary evidence only; never replaces round-trip or release criteria.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rudra.delivery.colorspace import rgb_to_rgb_matrix
from training.contract_checks import canonical_to_source, view_forward, load_reference
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha


def shadow_metrics(output, target, source):
    if output.shape != target.shape or source.shape != output.shape:
        raise ValueError('matching HWC images required')
    if not all(np.isfinite(x).all() for x in (output, target, source)):
        raise ValueError('finite images required')
    source_code = canonical_to_source(source)
    matrix = np.asarray(rgb_to_rgb_matrix('rec2020', 'rec709'), np.float32)
    pred, truth = output @ matrix.T, target @ matrix.T
    shown = canonical_to_source(view_forward(output))
    reference = canonical_to_source(view_forward(target))
    signed = (np.arcsinh(pred / 1e-5) - np.arcsinh(truth / 1e-5)).astype(np.float64)
    rows = {}
    for code in (5, 16):
        mask = source_code.max(-1) <= code/255
        count = int(mask.sum())
        row = {'pixels': count}
        if count:
            error = (shown - source_code)[mask] * 255
            target_error = (shown-reference)[mask] * 255
            row.update(signed_reference_rmse=float(np.sqrt(np.mean(signed[mask]**2))),
                       reference_code_rmse=float(np.sqrt(np.mean(target_error**2))),
                       source_increase_gt10_fraction=float(np.mean(error.max(-1)>10)),
                       reference_abs_error_gt10_fraction=float(np.mean(np.abs(target_error).max(-1)>10)),
                       negative_rec709_fraction=float(np.mean(pred[mask].min(-1)<-1e-8)),
                       target_negative_rec709_fraction=float(np.mean(truth[mask].min(-1)<-1e-8)))
        rows[str(code)] = row
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(2)
    if args.out.exists():
        raise ValueError('new report required')
    report = json.loads(args.report.read_text())
    if report['view'] != 'bt1886' or sha(report['manifest']) != report['manifest_sha256']:
        raise ValueError('unsupported view or manifest mismatch')
    manifest = {r['asset_id']: r for r in map(json.loads, Path(report['manifest']).read_text().splitlines())}
    rows = []
    for item in report['per_pair']:
        row = manifest[item['asset_id']]
        for kind in ('sdr', 'hdr'):
            if sha(row[kind+'_path']) != row[kind+'_sha256']:
                raise ValueError('media hash mismatch')
        path = args.report.parent / item['prediction']
        rows.append({'asset_id': row['asset_id'], 'prediction_sha256': sha(path),
                     'metrics': shadow_metrics(np.load(path), load_reference(row['hdr_path']),
                                               load_rgb(row['sdr_path'], hdr=False))})
    summary = {}
    for code in ('5', '16'):
        measured = [r['metrics'][code] for r in rows if r['metrics'][code]['pixels']]
        summary[code] = {'pairs': len(measured), 'pixels': sum(x['pixels'] for x in measured)}
        if measured:
            for key in measured[0]:
                if key != 'pixels':
                    summary[code][key] = float(np.mean([x[key] for x in measured]))
    out = {'report_sha256': sha(args.report), 'checkpoint_sha256': report['checkpoint_sha256'],
           'manifest_sha256': report['manifest_sha256'], 'summary': summary, 'per_pair': rows,
           'aggregation': 'equal-pair means over pairs with source-defined shadow support',
           'scope': 'diagnostic only; negative Rec709 is not alone proof of invalid HDR gamut',
           'release_authorized': False}
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
