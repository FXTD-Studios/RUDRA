"""Training-only diagnosis of bounded-head saturation, gradients and target range."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.dequantization import DequantizationLoss
from training.export_bench_pairs import load_model
from training.sdr2hdr_dataset import load_rgb
from training.run05_objective import restoration_total
from training.dark_experiment import near_black_log_loss
from rudra.sdr2hdr import sdr2hdr_loss
from rudra.source_baseline import baseline_curve_index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('use a new output')
    torch.set_num_threads(2)
    model = load_model(args.checkpoint, torch.device('cpu')).eval()
    if not model.bounded_source or model.reconstruction_policy != 'observed_clip_v1':
        raise ValueError('requires corrected bounded-source model')
    objective = DequantizationLoss(policy=model.reconstruction_policy)
    selected, counts = [], {}
    for row in map(json.loads, args.manifest.read_text().splitlines()):
        if row['split'] != 'train' or row.get('excluded', False):
            continue
        codec = row['sdr_codec']
        if counts.get(codec, 0) < 2:
            selected.append(row)
            counts[codec] = counts.get(codec, 0) + 1
    captured = {}
    hook = model.dequant.register_forward_hook(lambda module, inp, out: captured.update(raw=out))
    reports = []
    for row in selected:
        tensors = []
        for kind in ('sdr', 'hdr'):
            path = Path(row[kind + '_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row[kind + '_sha256']:
                raise ValueError('media hash mismatch')
            image = load_rgb(path, hdr=kind == 'hdr')
            h, w = image.shape[:2]
            image = image[(h-256)//2:(h-256)//2+256, (w-256)//2:(w-256)//2+256].copy()
            tensors.append(torch.from_numpy(image).permute(2, 0, 1)[None])
        source, target = tensors
        output = model(source, baseline_curve=baseline_curve_index('bt1886'))
        raw = captured['raw']
        losses = sdr2hdr_loss(output, source, target)
        main_loss = restoration_total(losses) + .05 * near_black_log_loss(output.hdr, target, source, 16/255)
        dq = objective(output.hdr, target, source)
        params = (model.dequant.weight, raw)
        old = torch.autograd.grad(main_loss, params, retain_graph=True)
        new = torch.autograd.grad(dq['total'], params, retain_graph=True)
        code_gradient = torch.autograd.grad(dq['code'], model.dequant.weight, retain_graph=True)[0]
        spatial_gradient = torch.autograd.grad(dq['gradient'], model.dequant.weight)[0]
        with torch.no_grad():
            codes = objective.source_code(source) * 255
            truth = objective.source_code(objective.forward_view(target)) * 255
            pred = objective.source_code(objective.forward_view(output.hdr)) * 255
            top = codes.amax(1, keepdim=True)
            keep = ((top >= 5) & (top <= 254)).expand_as(codes)
            values = raw[keep]
            delta = (truth - codes)[keep]
            def norm(x):
                return float(x.norm())
            def cosine(a, b):
                return float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()).clamp_min(1e-30))
            reports.append({'asset_id': row['asset_id'], 'codec': row['sdr_codec'],
                'ordinary_values': int(keep.sum()),
                'raw_abs_p50_p95_p99': torch.quantile(values.abs(), torch.tensor([.5,.95,.99])).tolist() if values.numel() else None,
                'raw_abs_gt3_fraction': float((values.abs() > 3).float().mean()) if values.numel() else None,
                'tanh_derivative_mean': float((1-values.tanh().square()).mean()) if values.numel() else None,
                'target_outside_bound_fraction': float((delta.abs() > .49).float().mean()) if delta.numel() else None,
                'target_outside_bound_excess_p95_max': [float(torch.quantile((delta.abs()-.49).clamp_min(0), .95)),
                                                       float((delta.abs()-.49).clamp_min(0).max())] if delta.numel() else None,
                'target_subcode_mean': float(delta.mean()) if delta.numel() else None,
                'predicted_subcode_mean': float((pred-codes)[keep].mean()) if delta.numel() else None,
                'target_code_mae_from_source': float(delta.abs().mean()) if delta.numel() else None,
                'prediction_target_code_mae': float((pred-truth)[keep].abs().mean()) if delta.numel() else None,
                'loss': {'primary': float(main_loss), 'auxiliary': float(dq['total'])},
                'parameter_gradient_norm': {'primary': norm(old[0]), 'auxiliary': norm(new[0]),
                    'weighted_auxiliary': .003*norm(new[0]), 'cosine': cosine(old[0],new[0])},
                'auxiliary_components': {'code_norm': norm(code_gradient), 'spatial_norm': norm(spatial_gradient),
                    'primary_code_cosine': cosine(old[0],code_gradient),
                    'primary_spatial_cosine': cosine(old[0],spatial_gradient)},
                'ordinary_raw_gradient_norm': {'primary': norm(old[1][keep]), 'auxiliary': norm(new[1][keep])}})
        print(f'[branch audit] {len(reports)}/{len(selected)}', flush=True)
    hook.remove()
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'scope': 'FP32 CPU, center256 training crops, first2 per codec; no optimization or validation tuning',
              'rows': reports}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
