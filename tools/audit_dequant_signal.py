"""CPU-only training-crop diagnostic for the experimental dequantization loss."""
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
from rudra.sdr2hdr import sdr2hdr_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('refusing to replace an audit')
    torch.set_num_threads(2)
    model = load_model(args.checkpoint, torch.device('cpu')).eval()
    if not getattr(model, 'bounded_source', False):
        raise ValueError('this diagnostic requires a bounded-source checkpoint')
    objective = DequantizationLoss()
    selected, counts = [], {}
    for line in args.manifest.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row.get('split') != 'train':
            continue
        codec = row.get('sdr_codec', 'none')
        if counts.get(codec, 0) < 2:
            selected.append(row)
            counts[codec] = counts.get(codec, 0) + 1
    reports = []
    for row in selected:
        def read(key, hdr):
            a = load_rgb(row[key], hdr=hdr)
            h, w = a.shape[:2]
            a = a[(h-256)//2:(h-256)//2+256, (w-256)//2:(w-256)//2+256].copy()
            return torch.from_numpy(a).permute(2, 0, 1)[None]
        source, target = read('sdr_path', False), read('hdr_path', True)
        output = model(source)
        legacy = restoration_total(sdr2hdr_loss(output, source, target))
        candidate = objective(output.hdr, target, source)
        parameter = model.dequant.weight
        old_grad = torch.autograd.grad(legacy, parameter, retain_graph=True)[0].norm()
        new_grad = torch.autograd.grad(candidate['total'], parameter)[0].norm()
        with torch.no_grad():
            src_code = objective.source_code(source) * 255
            truth = objective.source_code(objective.forward_view(target)) * 255
            pred = objective.source_code(objective.forward_view(output.hdr)) * 255
            top = src_code.amax(1, keepdim=True)
            keep = ((top >= 5) & (top <= .94 * 255)).expand_as(src_code)
            delta = (truth - src_code)[keep]
            report = dict(asset_id=row['asset_id'], codec=row.get('sdr_codec'),
                ordinary_values=int(keep.sum()),
                target_subcode_mae=float(delta.abs().mean()) if delta.numel() else None,
                prediction_subcode_mae=float((pred-src_code)[keep].abs().mean()) if delta.numel() else None,
                baseline_target_code_mae=float(delta.abs().mean()) if delta.numel() else None,
                prediction_target_code_mae=float((pred-truth)[keep].abs().mean()) if delta.numel() else None,
                candidate_loss={k: float(v.detach()) for k,v in candidate.items()},
                legacy_dequant_gradient_norm=float(old_grad), candidate_dequant_gradient_norm=float(new_grad))
            reports.append(report)
        print(f"audited {len(reports)}/{len(selected)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        manifest=str(args.manifest), crop='center256 native; train only; first2 per codec',
        note='Gradient comparison excludes the separate near-black term; not a quality or generalization gate.',
        rows=reports), indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
