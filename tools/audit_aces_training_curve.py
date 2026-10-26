"""Fixed TRAIN crop diagnostics across snapshots; never selects a release model."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.train_aces_spatial import load_training_rows
from tools.evaluate_run05_dequant import sha
from training.aces_model import ACESModel,reconstruction_weight
from training.aces_reference import inverse_aces2_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    torch.set_num_threads(2)
    data=load_training_rows(json.loads(a.pilot.read_text()))
    rng=np.random.default_rng(20261024);crops=[];locations=[]
    for i,(source,target,hi,lo) in enumerate(data):
        h,w=source.shape[:2]
        for mode,points in (('highlight',hi),('shadow',lo),('uniform',())):
            if len(points):
                py,px=points[int(rng.integers(len(points)))]
                y=int(np.clip(py-64,0,h-128));x=int(np.clip(px-64,0,w-128))
            else:y=int(rng.integers(h-127));x=int(rng.integers(w-127))
            crops.append((torch.from_numpy(source[y:y+128,x:x+128].copy()).permute(2,0,1),
                          torch.from_numpy(target[y:y+128,x:x+128].copy()).permute(2,0,1)))
            locations.append({'image':i,'mode':mode,'x':x,'y':y,'region_present':bool(len(points)) if mode!='uniform' else None})
    source=torch.stack([c[0] for c in crops]);target=torch.stack([c[1] for c in crops])
    baseline=inverse_aces2_reference(source);weight=reconstruction_weight(source)
    masks={'all':torch.ones_like(weight,dtype=torch.bool),'highlight':(weight>0)&(source.amax(1,keepdim=True)>.5),
           'shadow':(weight>0)&(source.amax(1,keepdim=True)<=.5)}
    def metrics(pred):
        error=(torch.asinh(pred)-torch.asinh(target)).square().mean(1,keepdim=True)
        return {name:{'pixel_count':int(mask.sum()),'pooled_mse':float(error[mask].mean()) if mask.any() else None,
                      'equal_crop_mse':float(torch.stack([e[m].mean() for e,m in zip(error,mask) if m.any()]).mean()) if mask.any() else None}
                for name,mask in masks.items()}
    result={'pilot_sha256':sha(a.pilot),'seed':20261024,'crop':128,'locations':locations,
            'scope':'fixed TRAIN probes only, region-centred with uniform fallback; no optimizer or checkpoint selection; not full-frame quality',
            'analytic':metrics(baseline),'checkpoints':[]}
    for checkpoint in a.checkpoints:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['pilot_sha256']!=result['pilot_sha256']:raise ValueError('training identity mismatch')
        model=ACESModel(state['model']['predictor.stem.weight'].shape[0],state['contract'].get('normalization','group')).eval()
        if model.contract()!=state['contract']:raise ValueError('contract mismatch')
        model.load_state_dict(state['model'],strict=True)
        with torch.inference_mode():
            pred=torch.cat([model(batch).acescg for batch in source.split(4)])
        if not torch.isfinite(pred).all():raise ValueError('nonfinite prediction')
        row={'path':str(checkpoint),'sha256':sha(checkpoint),'step':state['step'],'metrics':metrics(pred)}
        result['checkpoints'].append(row);print(json.dumps(row),flush=True)
    a.out.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
