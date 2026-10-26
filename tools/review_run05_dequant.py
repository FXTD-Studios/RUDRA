"""Render fixed diagnostic selections from saved full-frame trial predictions."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.contract_checks import canonical_to_source, view_forward, load_reference
from training.sdr2hdr_dataset import load_rgb
from tools.compare_run05_dequant import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--auxiliary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('use a new review directory')
    control, auxiliary = [json.loads(p.read_text()) for p in (args.control, args.auxiliary)]
    verdict = compare(control, auxiliary)
    cl, al = control['label'], auxiliary['label']
    crows = {r['asset_id']: r for r in control['per_pair']}
    arows = {r['asset_id']: r for r in auxiliary['per_pair']}
    manifest = {r['asset_id']: r for r in map(json.loads, Path(control['manifest']).read_text().splitlines())}

    def gradient(row, label):
        vals = [p['gradient_rmse'] for p in row['methods'][label]['reference']['pushes']
                if p['gradient_rmse'] is not None]
        return float(np.mean(vals)) if vals else None

    measurable = [k for k in crows if gradient(crows[k], cl) is not None]
    chosen = {}
    def select(key, reason):
        chosen.setdefault(key, []).append(reason)
    if measurable:
        delta = lambda k: gradient(arows[k], al) - gradient(crows[k], cl)
        select(max(measurable, key=delta), 'largest gradient regression / least improvement')
        select(min(measurable, key=delta), 'largest gradient improvement / least regression')
        clean = [k for k in measurable if arows[k]['codec'] == 'none']
        if clean:
            select(max(clean, key=delta), 'worst uncompressed gradient change')
    for metric, reason in [('h_abs_log2', 'worst auxiliary highlight error'),
                           ('s_visible_lift', 'worst auxiliary shadow lift')]:
        measured = [k for k in arows if arows[k]['methods'][al]['contract'].get(metric) is not None]
        if measured:
            select(max(measured, key=lambda k: arows[k]['methods'][al]['contract'][metric]), reason)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.out.mkdir(parents=True)
    reviews = []
    for i, (key, reasons) in enumerate(chosen.items()):
        row = manifest[key]
        source = canonical_to_source(load_rgb(row['sdr_path'], hdr=False))
        target = load_reference(row['hdr_path'])
        c = np.load(args.control.parent / crows[key]['prediction'])
        a = np.load(args.auxiliary.parent / arows[key]['prediction'])
        fig, axes = plt.subplots(2, 3, figsize=(18, 8), layout='constrained')
        panels = [('Source', source), ('Control, source view', canonical_to_source(view_forward(c))),
                  ('Auxiliary, source view', canonical_to_source(view_forward(a))),
                  ('Reference +3 EV', canonical_to_source(view_forward(target * 8))),
                  ('Control +3 EV', canonical_to_source(view_forward(c * 8))),
                  ('Auxiliary +3 EV', canonical_to_source(view_forward(a * 8)))]
        for axis, (title, pixels) in zip(axes.flat, panels):
            axis.imshow(np.clip(pixels, 0, 1)); axis.set_title(title); axis.axis('off')
        fig.suptitle(key + '\n' + '; '.join(reasons), fontsize=10)
        filename = f'{i:02d}_comparison.png'
        fig.savefig(args.out / filename, dpi=120)
        plt.close(fig)
        reviews.append({'asset_id': key, 'selection': reasons, 'image': filename})
    (args.out / 'selection.json').write_text(json.dumps({'reviews': reviews, 'trial': verdict}, indent=2))


if __name__ == '__main__':
    main()
