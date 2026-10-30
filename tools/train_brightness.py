"""Train the v3 brightness estimator on tools/build_brightness_set.py's set (CPU, minutes).

    python tools/train_brightness.py --set <dir> --out <file.json> [--over-weight 4] [--epochs 300]

Reports, on the held-out dev scenes: mean |b_est - b_true| against the master's |b_true| (b = 0),
and the overshoot rate (b_est - b_true > 1 stop). The saved estimator carries those numbers.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def metrics(est, X, y, w):
    if not len(y):
        return {}
    p = est.predict(X)
    d = p - y
    return {'n': int(len(y)), 'mae_est': float(np.average(np.abs(d), weights=w)), 'mae_master': float(np.average(np.abs(y), weights=w)),
            'bias': float(np.average(d, weights=w)), 'overshoot_rate_1stop': float(np.average(d > 1.0, weights=w)),
            'undershoot_rate_1stop': float(np.average(d < -1.0, weights=w))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--set', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--over-weight', type=float, default=4.0)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--seed', type=int, default=20261030)
    a = ap.parse_args(argv)
    from rudra.genfill.brightness import BrightnessEstimator
    d = np.load(Path(a.set) / 'set.npz')
    X, y, w, split = d['X'], d['y'], d['w'], d['split']
    tr, dv = split == 'train', split == 'dev'
    est = BrightnessEstimator().fit(X[tr], y[tr], w[tr], over_weight=a.over_weight, epochs=a.epochs, seed=a.seed)
    rep = {'train': metrics(est, X[tr], y[tr], w[tr]), 'dev': metrics(est, X[dv], y[dv], w[dv]),
           'set_summary': json.loads((Path(a.set) / 'summary.json').read_text())}
    est.meta.update({'report': rep})
    est.save(a.out)
    print(json.dumps({k: rep[k] for k in ('train', 'dev')}, indent=1))


if __name__ == '__main__':
    main()
