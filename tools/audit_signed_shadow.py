"""TRAIN-only dark-crop gradient diagnostic; no optimizer updates."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.signed_shadow import SignedShadowLoss
from training.reconstruction_gradient import ReconstructionGradientLoss
from training.dark_experiment import near_black_log_loss
from training.contract_checks import canonical_to_source, load_reference
from training.export_bench_pairs import load_model
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha
from training.run05_objective import restoration_total
from rudra.sdr2hdr import sdr2hdr_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--full-context', action='store_true', help='infer full frame before taking the same diagnostic crop')
    p.add_argument('--parameter-gradients', action='store_true')
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('new report required')
    if args.full_context and args.parameter_gradients:
        raise ValueError('full context parameter audit is not supported')
    torch.set_num_threads(2)
    model = load_model(args.checkpoint, torch.device('cpu')).eval()
    signed, spatial = SignedShadowLoss(), ReconstructionGradientLoss()
    rows, counts = [], {}
    for row in map(json.loads, args.manifest.read_text().splitlines()):
        family = row['sdr_curve']
        if row['split'] != 'train' or row.get('excluded') or counts.get(family, 0) >= 2:
            continue
        source = load_rgb(row['sdr_path'], hdr=False)
        dark = (canonical_to_source(source).max(-1) <= 16/255).astype(np.float32)
        if dark.mean() < .01:
            continue
        for kind in ('sdr', 'hdr'):
            if sha(row[kind+'_path']) != row[kind+'_sha256']:
                raise ValueError('media identity mismatch')
        counts[family] = counts.get(family, 0) + 1
        density = cv2.boxFilter(dark, -1, (256, 256), normalize=False)
        cy, cx = np.unravel_index(density.argmax(), density.shape)
        y, x = int(np.clip(cy-128, 0, source.shape[0]-256)), int(np.clip(cx-128, 0, source.shape[1]-256))
        def tensor(a):
            return torch.from_numpy(a[y:y+256, x:x+256].copy()).permute(2, 0, 1)[None]
        s, t = tensor(source), tensor(load_reference(row['hdr_path']))
        with torch.set_grad_enabled(args.parameter_gradients):
            prediction = model(s, baseline_curve=1)
            output = prediction.hdr
            context = {}
            if args.full_context:
                full = torch.from_numpy(source.copy()).permute(2, 0, 1)[None]
                full_output = model(full, baseline_curve=1).hdr[..., y:y+256, x:x+256].clone()
                context = {'crop_signed_loss': float(signed(output, t, s)),
                           'full_context_signed_loss': float(signed(full_output, t, s)),
                           'max_radiance_difference': float((full_output-output).abs().max())}
                output = full_output
        if not args.parameter_gradients:
            output = output.detach().requires_grad_()
        losses = {'signed': signed(output, t, s),
                  'near_black': near_black_log_loss(output, t, s),
                  'spatial': spatial(output, t, s)['total']}
        parameter_norms = {}
        if args.parameter_gradients:
            primary = restoration_total(sdr2hdr_loss(prediction, s, t)) + .05*losses['near_black']
            for name, value in {**losses, 'primary': primary}.items():
                pg = torch.autograd.grad(value, tuple(model.parameters()), retain_graph=True, allow_unused=True)
                parameter_norms[name] = sum(float(g.square().sum()) for g in pg if g is not None)**.5
        grads = {k: torch.autograd.grad(v, output, retain_graph=True)[0] for k, v in losses.items()}
        base = grads['signed']; evidence = {}
        for name, grad in grads.items():
            if not torch.isfinite(grad).all():
                raise ValueError('nonfinite gradient')
            cosine = (base*grad).sum() / (base.norm()*grad.norm()).clamp_min(1e-30)
            # A radiance-space diagnostic step; not model training or an optimum.
            step = grad * (1e-7 / grad.abs().max().clamp_min(1e-30))
            updated = output.detach() - step
            evidence[name] = {'loss': float(losses[name].detach()), 'output_gradient_norm': float(grad.norm()),
                              'cosine_with_signed': float(cosine),
                              'signed_loss_after_output_step': float(signed(updated, t, s).detach())}
        rows.append({'asset_id': row['asset_id'], 'family': family, 'crop_xy': [x, y],
                     'dark_fraction': float(dark[y:y+256,x:x+256].mean()), 'context': context,
                     'parameter_gradient_norms': parameter_norms, 'objectives': evidence})
        print(f'[signed shadow] {len(rows)}', flush=True)
    report = {'checkpoint_sha256': sha(args.checkpoint), 'manifest_sha256': sha(args.manifest),
              'scope': 'first2 TRAIN rows per look with >=1% source-dark pixels; input-selected darkest256 crop; no parameter updates',
              'full_context': args.full_context,
              'parameter_gradients': args.parameter_gradients,
              'rows': rows, 'release_authorized': False}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
