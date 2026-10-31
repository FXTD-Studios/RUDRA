"""Short camera-motion smoke: fixed-cell exposure change beyond analytic baseline.

Not optical-flow aligned and not a temporal release test. Scene content changes
can affect the statistic. Uses the same512/64 FP32 tiled Runner as image audits.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.contract_checks import Runner, LUMA2020
from training.sdr2hdr_dataset import load_rgb
from rudra.video import exposure_cells
from tools.evaluate_run05_dequant import sha


def excess(previous, current, baseline_previous, baseline_current):
    delta = (current-previous) - (baseline_current-baseline_previous)
    return {'global_excess_ev': abs(float(delta.mean())),
            'cell_excess_ev': float(np.abs(delta).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', action='append', required=True, help='label=path')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('new output required')
    torch.set_num_threads(2)
    checkpoints = {label: Path(path) for label, path in (s.split('=',1) for s in args.checkpoint)}
    manifest = json.loads(args.manifest.read_text())
    for clip in manifest['clips']:
        for frame in clip['frames']:
            if sha(frame['sdr_path']) != frame['sdr_sha256']:
                raise ValueError('source identity mismatch')
    runner = Runner(checkpoints, args.device, 'bt1886', policy='observed_clip_v1')
    clips = []
    for clip in manifest['clips']:
        previous = None
        rows = []
        for frame in clip['frames']:
            outputs = runner.run(load_rgb(frame['sdr_path'], hdr=False))
            if not all(np.isfinite(x).all() for x in outputs.values()):
                raise ValueError('nonfinite prediction')
            # Same canonical linear Rec2020 luminance for baseline and models.
            cells = {name: exposure_cells((value @ LUMA2020) * (10000/203)) for name, value in outputs.items()}
            if previous is not None:
                rows.append({'frame': frame['frame'], 'methods': {
                    name: excess(previous[name], cells[name], previous['baseline_bt1886'], cells['baseline_bt1886'])
                    for name in checkpoints}})
            previous = cells
        summary = {name: {key: {'mean': float(np.mean([r['methods'][name][key] for r in rows])),
                                'max': max(r['methods'][name][key] for r in rows)}
                           for key in ('global_excess_ev','cell_excess_ev')} for name in checkpoints}
        clips.append({'scene_id': clip['scene_id'], 'transitions': rows, 'summary': summary})
        print(json.dumps({'scene': clip['scene_id'], 'summary': summary}), flush=True)
    result = {'manifest_sha256': sha(args.manifest),
              'checkpoints': {label: {'path': str(path), 'sha256': sha(path)} for label,path in checkpoints.items()},
              'device': args.device, 'inference': 'FP32 tiled512 overlap64, no temporal smoothing',
              'exposure_units': 'linear luminance relative to203nit diffuse white; exposure_cells floor applied in those units',
              'clips': clips, 'release_authorized': False,
              'scope': '2short9-frame camera-motion snippets; fixed-cell analytic-baseline-relative diagnostic, not flow aligned; no object motion, compressed video or generator coverage'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
