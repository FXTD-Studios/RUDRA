"""TRAIN-only context-use diagnosis; no fitting or checkpoint selection."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import minimum_filter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def regional(pred, target, mask):
    count = int(mask.sum())
    if not count:
        return None
    a, b = np.arcsinh(pred), np.arcsinh(target)
    residual = a - b
    sums, edges = 0., 0
    for axis in (0, 1):
        left = [slice(None)] * 2
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        valid = mask[tuple(left)] & mask[tuple(right)]
        sums += float(np.square(np.diff(residual, axis=axis))[valid].mean(-1).sum(dtype=np.float64))
        edges += int(valid.sum())
    return {'pixels': count, 'edges': edges,
            'photo_asinh_mse': float(np.square(residual[mask]).mean(dtype=np.float64)),
            'gradient_asinh_mse': sums / edges if edges else None,
            'pred_asinh_rgb_std': a[mask].std(0, dtype=np.float64).tolist(),
            'target_asinh_rgb_std': b[mask].std(0, dtype=np.float64).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    for path, digest in reg['input_hashes'].items():
        if sha(path) != digest:
            raise ValueError('input identity')
    args.out.mkdir(exist_ok=False)
    (args.out / 'registration.json').write_bytes(args.registration.read_bytes())
    (args.out / 'auditor.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    view = reference_view(); view.set_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = {name: model_from_checkpoint(torch.load(path, map_location='cpu', weights_only=True)).eval().cuda()
              for name, path in reg['checkpoints'].items()}
    audit = json.loads(Path(reg['context_audit']).read_text())
    selected = {r['asset_id']: r for r in audit['rows'] if r['domain'] == 'replay' and r['blind_pixels']}
    pilots = {r['asset_id']: r for r in json.loads(Path(reg['pilot']).read_text())['rows']}
    schedule = json.loads(Path(reg['schedule']).read_text())
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].cuda()
    array = lambda a: a[0].permute(1, 2, 0).cpu().numpy()
    rows, crops, probes = [], [], []
    for asset, entry in selected.items():
        row = pilots[asset]
        if row['split'] != 'train': raise ValueError('TRAIN only')
        for key in ('sdr', 'target'):
            if sha(row[key + '_path']) != row[key + '_sha256']: raise ValueError('content identity')
        source = canonical_to_source(load_rgb(row['sdr_path'], hdr=False))
        target = np.load(row['target_path'])
        radius = entry['radius_bound']
        blind = minimum_filter((source == 1).all(-1), size=2*radius+1, mode='constant', cval=0)
        assert int(blind.sum()) == entry['blind_pixels']
        full = source.min(-1) >= 254.5/255
        with torch.inference_mode():
            outputs = {name: array(model(tensor(source)).acescg) for name, model in models.items()}
        masks = {'blind': blind, 'full_other': full & ~blind}
        rows.append({'asset_id': asset, 'scene': row['scene_id'], 'regions': {
            region: {name: regional(pred, target, mask) for name, pred in outputs.items()}
            for region, mask in masks.items()}})
        for index, sample in enumerate(schedule):
            if sample['domain'] != 'replay' or asset not in sample['assets']: continue
            y, x, h, w = sample['crop']; t, b, l, r = sample['borders']
            valid = np.zeros_like(blind); valid[y+t:y+h-b, x+l:x+w-r] = True
            if not (blind & valid).any(): continue
            crops.append({'schedule_index': index, 'asset_id': asset,
                          'metrics': {name: regional(pred, target, blind & valid) for name, pred in outputs.items()}})
        # Median blind coordinate is deterministic. Preserve its entire old
        # receptive square and change only the remote image context.
        points = np.argwhere(blind); y, x = points[len(points)//2]
        changed = source.copy(); changed[:] = .18
        changed[y-radius:y+radius+1, x-radius:x+radius+1] = source[y-radius:y+radius+1, x-radius:x+radius+1]
        values = {}
        with torch.inference_mode():
            for name, model in models.items():
                if name == 'parent':
                    field = lambda s: model.full_context(tensor(s), tensor(view.inverse(s)))[:, 3:]
                else:
                    field = lambda s: model.context_fields(model.conditioning(tensor(s), tensor(view.inverse(s))), source.shape[:2])
                original, altered = field(source), field(changed)
                values[name] = float((original[0, :, y, x] - altered[0, :, y, x]).abs().max())
        assert values['parent'] == 0 and values['local'] == 0
        probes.append({'asset_id': asset, 'point_yx': [int(y), int(x)], 'remote_field_change_max': values})
        print('audited', asset, flush=True)
    summary = {}
    for name in models:
        metrics = [r['regions']['blind'][name] for r in rows]
        summary[name] = {
            'blind_pixel_weighted_photo': sum(m['photo_asinh_mse']*m['pixels'] for m in metrics)/sum(m['pixels'] for m in metrics),
            'blind_edge_weighted_gradient': sum(m['gradient_asinh_mse']*m['edges'] for m in metrics)/sum(m['edges'] for m in metrics),
            'scheduled_blind_photo': sum(r['metrics'][name]['photo_asinh_mse']*r['metrics'][name]['pixels'] for r in crops)/sum(r['metrics'][name]['pixels'] for r in crops)}
    result = {'summary': summary, 'rows': rows, 'scheduled_crops': crops, 'probes': probes,
              'blind_pixels': sum(r['regions']['blind']['parent']['pixels'] for r in rows),
              'scheduled_blind_pixels': sum(r['metrics']['parent']['pixels'] for r in crops),
              'registration_sha256': sha(args.registration), 'tool_sha256': sha(__file__),
              'scope': reg['scope'], 'release_authorized': False}
    (args.out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'summary': summary, 'probes': probes}, indent=2))


if __name__ == '__main__': main()
