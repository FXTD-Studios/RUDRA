"""TRAIN-only gradient scale probe before any shadow-objective experiment."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import ACESModel
from training.aces_losses import absolute_reconstruction_loss,shadow_relative_loss,shadow_reference_scale
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','pilot','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new report required')
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert state['pilot_sha256']==sha(a.pilot)
    model=ACESModel(8,normalization='pixel_rms').to(device).eval();model.load_state_dict(state['model'])
    results=[];counts={'highlight':0,'shadow':0}
    for row in json.loads(a.pilot.read_text())['rows']:
        assert row['split']=='train'
        if sha(row['sdr_path'])!=row['sdr_sha256'] or sha(row['target_path'])!=row['target_sha256']:raise ValueError('identity mismatch')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
        top=source.max(-1)
        for region,support in (('highlight',top>=254.5/255),('shadow',top<3/255)):
            if counts[region]>=8 or support.sum()<200:continue
            pooled=torch.nn.functional.avg_pool2d(torch.from_numpy(support.astype(np.float32))[None,None],128,stride=32)[0,0]
            yy,xx=np.unravel_index(int(pooled.argmax()),pooled.shape);y=int(yy)*32;x=int(xx)*32
            s=torch.from_numpy(source[y:y+128,x:x+128].copy()).permute(2,0,1)[None].to(device)
            t=torch.from_numpy(target[y:y+128,x:x+128].copy()).permute(2,0,1)[None].to(device)
            out=model(s);error=(torch.asinh(out.acescg)-torch.asinh(t)).square().mean(1,keepdim=True)
            w=out.reconstruction_weight;shadow=w*(s.amax(1,keepdim=True)<=.5);highlight=w-shadow
            losses={'absolute':absolute_reconstruction_loss(out,t),
                    'absolute_shadow_contribution':(error*shadow).sum()/w.sum().clamp_min(1),
                    'absolute_highlight_contribution':(error*highlight).sum()/w.sum().clamp_min(1),
                    'view_scaled_shadow':shadow_relative_loss(out,t,s)}
            measured={}
            for name,loss in losses.items():
                gradients=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True,allow_unused=True)
                norm=sum(float(g.detach().square().sum()) for g in gradients if g is not None)**.5
                measured[name]={'loss':float(loss.detach()),'parameter_gradient_norm':norm}
            results.append({'asset_id':row['asset_id'],'region':region,'xy':[x,y],'measurements':measured})
            counts[region]+=1;print('[loss probe]',counts,flush=True)
        if min(counts.values())>=8:break
    a.out.write_text(json.dumps({'checkpoint_sha256':sha(a.checkpoint),'pilot_sha256':sha(a.pilot),
        'shadow_scale_ap1':shadow_reference_scale(),'rows':results,'scope':'first8 eligible TRAIN images per region; parameter gradient diagnostic, no optimizer update'},indent=2,allow_nan=False))


if __name__=='__main__':main()
