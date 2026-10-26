"""Fixed TRAIN-only shadow crops; separate brightness/chroma errors, no optimization."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint,reconstruction_weight
from training.aces_shadow import luminance
from training.aces_losses import shadow_reference_scale
from tools.train_aces_spatial import load_training_rows
from tools.evaluate_run05_dequant import sha


def error_maps(pred,target):
    scale=shadow_reference_scale();py=luminance(pred);ty=luminance(target)
    pc=pred[:,(0,2)]-py;tc=target[:,(0,2)]-ty
    return {'rgb_scaled':(torch.asinh(pred/scale)-torch.asinh(target/scale)).square().mean(1,keepdim=True),
            'rgb_absolute':(torch.asinh(pred)-torch.asinh(target)).square().mean(1,keepdim=True),
            'luma_scaled':(torch.asinh(py/scale)-torch.asinh(ty/scale)).square(),
            'chroma_scaled':(torch.asinh(pc/scale)-torch.asinh(tc/scale)).square().mean(1,keepdim=True)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pilot','checkpoint','control','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new report required')
    torch.set_num_threads(2);rng=np.random.default_rng(20261029)
    pilot=json.loads(a.pilot.read_text());data=load_training_rows(pilot)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if state['recipe']['training_pilot_sha256']!=sha(a.pilot):raise ValueError('TRAIN identity mismatch')
    models={'candidate':model_from_checkpoint(state).eval(),
            'control':model_from_checkpoint(torch.load(a.control,map_location='cpu',weights_only=True)).eval()}
    if state['recipe']['base_checkpoint_sha256']!=sha(a.control):raise ValueError('frozen control mismatch')
    rows=[];locations=[]
    for index,(source,target,_,points) in enumerate(data):
        if not len(points):continue
        sums={};den=0.;pixels=0
        for _ in range(8):
            h,w=source.shape[:2];py,px=points[int(rng.integers(len(points)))];y=int(np.clip(py-rng.integers(8,120),0,h-128));x=int(np.clip(px-rng.integers(8,120),0,w-128))
            locations.append([index,x,y]);src=torch.from_numpy(source[y:y+128,x:x+128].copy()).permute(2,0,1)[None]
            truth=torch.from_numpy(target[y:y+128,x:x+128].copy()).permute(2,0,1)[None]
            weight=(reconstruction_weight(src)*(src.amax(1,keepdim=True)<=.5))[:,:,8:-8,8:-8]
            den+=float(weight.sum());pixels+=int((weight>0).sum())
            with torch.inference_mode():
                outputs={name:model(src) for name,model in models.items()}
                predictions={name:out.acescg for name,out in outputs.items()};predictions['analytic']=outputs['candidate'].baseline
                for name,pred in predictions.items():
                    for metric,values in error_maps(pred,truth).items():
                        key=name+'/'+metric;sums[key]=sums.get(key,0.)+float((values[:,:,8:-8,8:-8]*weight).sum())
        rows.append({'asset_id':pilot['rows'][index]['asset_id'],'weighted_support':den,'pixel_observations':pixels,
                     'metrics':{k:v/den if den else None for k,v in sums.items()}})
        print('TRAIN image',index,'pixels',pixels,flush=True)
    valid=[r for r in rows if r['weighted_support']>0]
    means={k:float(np.mean([r['metrics'][k] for r in valid])) for k in valid[0]['metrics']}
    result={'rows':rows,'equal_image_mean':means,'locations':locations,'seed':20261029,'crop':128,'excluded_border':8,
            'source_sha256':{k:sha(v) for k,v in {'pilot':a.pilot,'checkpoint':a.checkpoint,'control':a.control,'script':Path(__file__)}.items()},
            'scope':'fixed eight targeted TRAIN crops per shadow-eligible image; overlapping observations, weighted losses; not full-frame development or release'}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(means,indent=2))


if __name__=='__main__':main()
