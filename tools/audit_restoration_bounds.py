"""Measure unavoidable clean-code error under source-preservation budgets.

Synthetic diagnostic only. The oracle projection is not a model and cannot
establish image quality. No corpus, target policy or acceptance gate is changed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.run05_corpus import degrade


def bound_metrics(clean, observed, budget):
    """Exact pointwise lower bound for squared code error in a box interval."""
    error = np.maximum(np.abs(observed - clean) - budget, 0)
    return {'oracle_rmse': float(np.sqrt(np.mean(error ** 2))),
            'channels_outside_budget_fraction': float(np.mean(error > 1e-6)),
            'oracle_max_error': float(error.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('use a new report')
    y, x = np.mgrid[:256, :512].astype(np.float32)
    clean = np.stack((30 + 190*x/511, 40 + 170*y/255,
                      60 + 110*(x+y)/766), axis=-1)
    # Coloured edges exercise chroma coding; all values stay away from clipping.
    clean[64:192, 160:320] = [180, 45, 95]
    source = np.rint(clean).astype(np.uint8)
    results = []
    for codec, quality in [('none', 0), ('jpeg', 70), ('libx264', 26),
                           ('libx265', 26), ('libsvtav1', 26), ('libaom-av1', 26)]:
        observed = degrade(source, {'codec': codec, 'quality': quality}).astype(np.float32)
        results.append({'codec': codec, 'quality': quality,
                        'input_rmse': float(np.sqrt(np.mean((observed-clean)**2))),
                        'budgets': {str(b): bound_metrics(clean, observed, b) for b in (.49, 1., 2.)}})
    report = {'units': '8-bit source RGB codes', 'synthetic_shape': list(clean.shape),
              'results': results, 'release_authorized': False,
              'scope': 'Pointwise clean-target feasibility; not perceptual quality, temporal evidence or representative data coverage.'}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
