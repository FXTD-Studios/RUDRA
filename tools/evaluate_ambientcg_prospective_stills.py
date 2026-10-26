"""Evaluate frozen prospective cases only after the original DEV gates pass."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prospective_checkpoint_gate import verify_candidate, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-root', type=Path, required=True)
    ap.add_argument('--steps', type=int, choices=(1600, 6400), required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    a = ap.parse_args()
    selection = verify_candidate(a.candidate_root, a.steps)
    if a.preflight_only:
        print(json.dumps(selection, indent=2)); return
    if a.out is None or a.out.exists():
        raise ValueError('A new --out directory is required')
    corpus = Path('G:/datasets/corpora/ambientcg_prospective_stills_20261025')
    registration = Path('configs/experiments/ambientcg_prospective_stills_20261025.json')
    dataset = json.loads((corpus / 'report.json').read_text())
    reg = json.loads(registration.read_text())
    if dataset['registration_sha256'] != sha(registration) or sha(corpus / 'registration.json') != sha(registration):
        raise ValueError('Prospective registration changed')
    expected = {(s['asset_id'], yaw, pitch, ev, bits) for s in reg['sources']
                for yaw in reg['projection']['yaw_degrees'] for pitch in reg['projection']['pitch_degrees']
                for ev in reg['exposure']['ev_offsets'] for bits in reg['source_bits']}
    actual = [(r['scene_id'], r['yaw'], r['pitch'], r['ev'], r['source_bits']) for r in dataset['rows']]
    if len(actual) != 48 or len(set(actual)) != 48 or set(actual) != expected:
        raise ValueError('Missing, duplicate or altered prospective cases')
    for row in dataset['rows']:
        for key in ('source', 'reference'):
            if sha(row[key + '_path']) != row[key + '_sha256']:
                raise ValueError('Changed prospective media')
    # Only now may a model be loaded or these images evaluated.
    import numpy as np
    import torch
    from tools.rudra_aces_export import load_model, read_still, infer
    from tools.prospective_metrics import measure, aggregate
    from training.aces_reference import reference_view
    from training.aces_model import reconstruction_weight
    from rudra.delivery.exr import write_exr, read_exr
    from rudra.delivery.colorspace import AP1_CHROMATICITIES
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    view = reference_view(); view.set_threads(2)
    selection.update(registration_sha256=sha(registration), corpus_report_sha256=sha(corpus / 'report.json'),
                     evaluator_sha256=sha(__file__),
                     metrics_sha256=sha(Path(__file__).with_name('prospective_metrics.py')),
                     gate_sha256=sha(Path(__file__).with_name('prospective_checkpoint_gate.py')),
                     output=a.out.resolve().as_posix())
    freeze = Path('reports/ambientcg_holdout_candidates_20261025/frozen_model_selection.json')
    # One selected checkpoint: a later output path must not silently turn these
    # prospective cases into another model-selection loop. Preserve failed runs.
    with freeze.open('x') as f:
        json.dump(selection, f, indent=2)
    a.out.mkdir(); (a.out / 'masters').mkdir()
    (a.out / 'selection.json').write_text(json.dumps(selection, indent=2))
    model, _ = load_model(selection['checkpoint'], a.device)
    rows = []
    for index, case in enumerate(dataset['rows']):
        source = read_still(case['source_path']); truth = np.load(case['reference_path'])
        candidate, mask, stats, _ = infer(model, source, a.device, include_fully_clipped=True,
                                         clipped_smoothing='hybrid64')
        analytic = view.inverse(source)
        w = reconstruction_weight(torch.from_numpy(source).permute(2, 0, 1)[None])[0, 0].numpy()
        high = (w > 0) & (source.max(-1) > .5)
        full = high & (source.min(-1) >= 254.5 / 255)
        regions = {'highlight': high, 'partial': high & ~full, 'fully_clipped': full}
        methods = {'analytic': analytic, 'candidate': candidate}
        metrics = measure(methods, truth, regions, view)
        path = a.out / 'masters' / f'{index:03d}.exr'
        write_exr(path, candidate, half=False, chromaticities=AP1_CHROMATICITIES, reconstruction_mask=mask)
        reread, _ = read_exr(path)
        assert np.array_equal(reread, candidate)
        rows.append({'case': case, 'coverage_pixels': {k: int(v.sum()) for k, v in regions.items()},
                     'metrics': metrics, 'appearance': stats, 'master': path.as_posix(),
                     'master_sha256': sha(path), 'float_master_exact_readback': True})
        (a.out / 'partial.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
        print(f'{index+1}/48 evaluated', flush=True)
    groups = aggregate(rows, [s['asset_id'] for s in reg['sources']], reg['source_bits'])
    report = {'selection': selection, 'rows': rows, 'groups': groups,
              'aggregation': 'Equal mean of supported per-frame metrics; empty supports explicitly counted, never scored as zero.',
              'scope': reg['scope'], 'release_authorized': False}
    (a.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
