"""Save source-view comparisons and reusable predictions for diagnostic review."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.contract_checks import Runner, load_reference, view_forward, canonical_to_source
from training.banding_diagnostics import reference_gradient_errors
from training.sdr2hdr_dataset import load_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('use a new review directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(2)
    report = json.loads(args.report.read_text())
    label, checkpoint = next(iter(report['checkpoints'].items()))
    chosen = [r for r in report['per_pair'] if r['methods'][label].get('s_visible_lift') is not None]
    highlights = [r for r in report['per_pair'] if r['methods'][label].get('h_abs_log2') is not None]
    if highlights:
        worst = max(highlights, key=lambda r:r['methods'][label]['h_abs_log2'])
        if worst['asset_id'] not in {r['asset_id'] for r in chosen}:
            chosen.append(worst)
    rows = {r['asset_id']:r for r in map(json.loads, Path(report['manifest']).read_text().splitlines())}
    runner = Runner({label:Path(checkpoint)}, 'cpu', report['view'], policy=report.get('reconstruction_policy', 'legacy_nearwhite'))
    args.out.mkdir(parents=True)
    summaries = []
    for i, item in enumerate(chosen):
        row = rows[item['asset_id']]
        source = load_rgb(row['sdr_path'], hdr=False)
        target = load_reference(row['hdr_path'])
        outputs = runner.run(source)
        prediction = outputs[label]
        np.save(args.out / f'{i:02d}_prediction.npy', prediction)
        metrics = {name:reference_gradient_errors(value, target, report['view']) for name,value in outputs.items()}
        summaries.append(dict(index=i, asset_id=item['asset_id'], source=row['sdr_path'], target=row['hdr_path'], metrics=metrics))
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout='constrained')
        views = [('Source', canonical_to_source(source)),
                 ('Reference +3 EV', canonical_to_source(view_forward(target * 8, report['view']))),
                 ('Model, declared view', canonical_to_source(view_forward(prediction, report['view']))),
                 ('Model +3 EV', canonical_to_source(view_forward(prediction * 8, report['view'])))]
        for axis, (title, pixels) in zip(axes.flat, views):
            axis.imshow(np.clip(pixels, 0, 1)); axis.set_title(title); axis.axis('off')
        fig.suptitle(item['asset_id'], fontsize=10)
        fig.savefig(args.out / f'{i:02d}_review.png', dpi=110)
        plt.close(fig)
        print(f'review {i+1}/{len(chosen)}', flush=True)
    (args.out/'review.json').write_text(json.dumps(dict(checkpoint=checkpoint, view=report['view'], rows=summaries), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
