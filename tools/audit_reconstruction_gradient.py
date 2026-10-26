"""TRAIN-only input-boundary crop signal audit; no parameter updates."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.reconstruction_gradient import ReconstructionGradientLoss
from training.contract_checks import canonical_to_source, load_reference
from training.export_bench_pairs import load_model
from training.sdr2hdr_dataset import load_rgb
from training.run05_objective import restoration_total
from training.dark_experiment import near_black_log_loss
from rudra.sdr2hdr import sdr2hdr_loss
from rudra.source_baseline import baseline_curve_index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise ValueError('new output required')
    torch.set_num_threads(2)
    model=load_model(args.checkpoint,torch.device('cpu')).eval()
    if model.reconstruction_policy!='observed_clip_v1': raise ValueError('wrong model policy')
    objective=ReconstructionGradientLoss()
    params=tuple(model.parameters())
    rows=[];counts={}
    for row in map(json.loads,args.manifest.read_text().splitlines()):
        family=row['sdr_curve']
        if row['split']!='train' or row.get('excluded') or row['hi_fraction']<.001 or counts.get(family,0)>=2: continue
        counts[family]=counts.get(family,0)+1
        source=load_rgb(row['sdr_path'],hdr=False)
        target=load_reference(row['hdr_path'])
        top=canonical_to_source(source).max(axis=2)
        missing=(top>=254.5/255).astype(np.float32)
        boundary=np.abs(np.diff(missing,axis=0,append=missing[-1:]))+np.abs(np.diff(missing,axis=1,append=missing[:,-1:]))
        density=cv2.boxFilter(boundary,-1,(256,256),normalize=False)
        cy,cx=np.unravel_index(density.argmax(),density.shape)
        y=int(np.clip(cy-128,0,source.shape[0]-256));x=int(np.clip(cx-128,0,source.shape[1]-256))
        def tensor(a):return torch.from_numpy(a[y:y+256,x:x+256].copy()).permute(2,0,1)[None]
        s,t=tensor(source),tensor(target)
        output=model(s,baseline_curve=baseline_curve_index('bt1886'))
        primary=restoration_total(sdr2hdr_loss(output,s,t))+.05*near_black_log_loss(output.hdr,t,s,16/255)
        candidate=objective(output.hdr,t,s)
        a=torch.autograd.grad(primary,params,allow_unused=True,retain_graph=True)
        b=torch.autograd.grad(candidate['total'],params,allow_unused=True)
        norm=lambda grads:sum(float(g.square().sum()) for g in grads if g is not None)**.5
        an,bn=norm(a),norm(b)
        dot=sum(float((ga*gb).sum()) for ga,gb in zip(a,b) if ga is not None and gb is not None)
        rows.append(dict(asset_id=row['asset_id'],family=family,codec=row['sdr_codec'],crop_xy=[x,y],
            support_fraction=float(candidate['support_fraction']),primary_loss=float(primary.detach()),
            candidate_loss=float(candidate['total'].detach()),primary_norm=an,candidate_norm=bn,
            cosine=dot/max(an*bn,1e-30),candidate_over_primary=bn/max(an,1e-30)))
        print(f'[reconstruction gradient] {len(rows)}',flush=True)
    args.out.write_text(json.dumps(dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        scope='CPU FP32 TRAIN first2 high-containing rows per look; input-only maximum-boundary256 crop; no optimizer updates',rows=rows),indent=2,allow_nan=False))


if __name__=='__main__': main()
