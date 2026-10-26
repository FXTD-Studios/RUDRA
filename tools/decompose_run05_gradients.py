"""Diagnostic-only zero-EV error attribution; leaves registered trial gates unchanged."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.contract_checks import canonical_to_source, view_forward, load_reference
from training.banding_diagnostics import LUMA709
from training.sdr2hdr_dataset import load_rgb
from pipeline.run05_corpus import declared_inverse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('use a new output')
    report = json.loads(args.report.read_text())
    source_rows = {r['asset_id']: r for r in map(json.loads, Path(report['manifest']).read_text().splitlines())}
    rows = []
    for item in report['per_pair']:
        row = source_rows[item['asset_id']]
        source = load_rgb(row['sdr_path'], hdr=False)
        target = load_reference(row['hdr_path'])
        truth = canonical_to_source(view_forward(target)) * 1023
        luma = truth @ LUMA709
        smooth = cv2.GaussianBlur(luma, (0, 0), 4)
        sy, sx = np.gradient(smooth)
        gentle = (np.hypot(sx, sy) > .03) & (np.hypot(sx, sy) < 1.5) & (luma > 2) & (luma < 1020)
        gentle[[0,-1],:] = False; gentle[:,[0,-1]] = False
        top = canonical_to_source(source).max(axis=2)
        ordinary = (top >= 5/255) & (top <= 254/255)
        outputs = {'baseline': declared_inverse(source, 'bt1886'),
                   report['label']: np.load(args.report.parent / item['prediction'])}
        result = {'asset_id': item['asset_id'], 'codec': item['codec'], 'methods': {}}
        for label, output in outputs.items():
            residual = (canonical_to_source(view_forward(output))*1023-truth).astype(np.float64)
            groups = {'ordinary_edges': [], 'reconstruction_or_boundary_edges': []}
            for axis in (0,1):
                ordinary_edge = ordinary & np.roll(ordinary, -1, axis=axis)
                error = np.roll(residual, -1, axis=axis)-residual
                groups['ordinary_edges'].append(error[gentle & ordinary_edge])
                groups['reconstruction_or_boundary_edges'].append(error[gentle & ~ordinary_edge])
            result['methods'][label] = {}
            for group, values in groups.items():
                values = np.concatenate(values)
                result['methods'][label][group] = {'edge_channels': int(values.size),
                    'squared_error_sum': float(np.square(values).sum()),
                    'rmse': float(np.sqrt(np.mean(values**2))) if values.size else None}
        rows.append(result)
    args.out.write_text(json.dumps({'scope': 'zeroEV diagnostic with same reference gentle-region rule; ordinary edges exclude reconstructed neighbors',
        'report': str(args.report), 'rows': rows}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
