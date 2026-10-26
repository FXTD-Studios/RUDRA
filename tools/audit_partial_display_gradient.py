"""Frozen TRAIN directional-derivative and recorded normalization diagnosis."""
import json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.train_partial_display import _Forward,forward_partial
from tools.summarize_aces_generated import sha
from training.aces_model import model_from_checkpoint
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def main():
    out=Path('G:/rudra_training/partial_display_gradient_20261025');out.mkdir(exist_ok=False)
    root=Path('G:/rudra_training/partial_temporal1_400_20261025')
    checkpoints={'initial':'G:/rudra_training/aces_guarded_projected600_20261024/step600.pt','final':str(root/'step400.pt')}
    pilots={'still':'G:/datasets/corpora/aces_consistent_exposures_v2_20261024/train/report.json',
            'motion':'G:/datasets/corpora/aces_temporal_context_pairs_20261024/train/report.json'}
    pools={'still':root/'crop_pool.json','motion':root/'motion_pool.json'}
    reg={'checkpoints':{p:sha(p) for p in checkpoints.values()},'pilots':{p:sha(p) for p in pilots.values()},
         'pools':{str(p):sha(p) for p in pools.values()},'history_sha256':sha(root/'report.json'),
         'selection':'First crop from first four distinct scenes per domain; up to64 evenly spaced partial pixels inside64px excluded border; initial and final weights.',
         'steps':[.01,.001,.0001,.00001],'seed':20261025,
         'scope':'Output-radiance directional derivatives, not parameter-space optimization. Raw code-squared MSE. EMA history described separately. No tuning or training.'}
    (out/'registration.json').write_text(json.dumps(reg,indent=2));(out/'auditor.py').write_bytes(Path(__file__).read_bytes())
    view=reference_view();view.set_threads(2);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models={k:model_from_checkpoint(torch.load(p,map_location='cpu',weights_only=True)).eval().cuda() for k,p in checkpoints.items()}
    rng=np.random.default_rng(reg['seed']);records=[]
    for domain,pilot in pilots.items():
        rows={r['asset_id']:r for r in json.loads(Path(pilot).read_text())['rows']};seen=set()
        for sample in json.loads(pools[domain].read_text()):
            asset=sample['asset_id'] if domain=='still' else sample['assets'][0];row=rows[asset]
            if row['scene_id'] in seen:continue
            if row['split']!='train':raise ValueError('TRAIN only')
            for key in ('sdr','target'):
                if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('input identity')
            source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
            y,x=sample['origin'];source=source[y:y+192,x:x+192];target=target[y:y+192,x:x+192]
            tensor=torch.from_numpy(source.copy()).permute(2,0,1)[None].cuda()
            for name,model in models.items():
                with torch.no_grad():pred,mask,_=forward_partial(model,tensor)
                pred=pred[0,:,64:-64,64:-64].permute(1,2,0).cpu().numpy();valid=mask[0,0,64:-64,64:-64].cpu().numpy()
                indices=np.flatnonzero(valid);indices=indices[np.linspace(0,len(indices)-1,min(64,len(indices)),dtype=int)]
                a=pred.reshape(-1,3)[indices];truth=target[64:-64,64:-64].reshape(-1,3)[indices]
                random=rng.normal(size=a.shape).astype(np.float32)
                for ev in (-3,0,3):
                    scale=2.**ev;z=torch.from_numpy(a.T.copy())[None,:,None,:].requires_grad_()
                    desired=view.forward(truth*scale)
                    shown=_Forward.apply(z*scale);desired_t=torch.from_numpy(desired.T.copy())[None,:,None,:]
                    loss=((shown-desired_t)*255).square().mean();loss.backward();g=z.grad[0,:,0].numpy().T.copy()
                    def objective(value):
                        residual=(view.forward(np.ascontiguousarray(value*scale))-desired).astype(np.float64)*255
                        return float(np.square(residual).mean())
                    directions={'random':random,'descent':-g}
                    tests=[]
                    for direction,d in directions.items():
                        d=d/max(float(np.max(np.abs(d))),1e-20)*np.maximum(np.abs(a),.01)
                        predicted=float(np.sum(g.astype(np.float64)*d))
                        for step in reg['steps']:
                            fd=(objective(a+step*d)-objective(a-step*d))/(2*step)
                            tests.append({'direction':direction,'step':step,'predicted':predicted,'measured':fd,
                                          'relative_disagreement':abs(fd-predicted)/max(abs(fd),abs(predicted),1e-8)})
                    records.append({'domain':domain,'asset_id':asset,'state':name,'ev':ev,'pixels':len(indices),'loss':float(loss.detach()),
                                    'gradient_l2':float(np.linalg.norm(g)),'nonfinite_gradient_values':int((~np.isfinite(g)).sum()),'directional_tests':tests})
            seen.add(row['scene_id']);print('audited',domain,len(seen),flush=True)
            if len(seen)==4:break
        if len(seen)!=4:raise ValueError('coverage')
    hist=json.loads((root/'report.json').read_text())['history'];normalization={}
    for ev in ('-3.0','0.0','3.0'):
        values=np.array([r['analytic_ema'][ev] for r in hist]);effective=.5/3/np.maximum(values,1e-4)
        normalization[ev]={'ema_min':float(values.min()),'ema_max':float(values.max()),'effective_coefficient_min':float(effective.min()),'effective_coefficient_max':float(effective.max()),'ema_floor_count':int((values<1e-4).sum())}
    summary={}
    for step in reg['steps']:
        values=[t['relative_disagreement'] for r in records for t in r['directional_tests'] if t['step']==step]
        summary[str(step)]={'median_relative_disagreement':float(np.median(values)),'p95':float(np.percentile(values,95)),
                            'sign_disagreements':sum(t['predicted']*t['measured']<0 for r in records for t in r['directional_tests'] if t['step']==step)}
    result={'rows':records,'summary':summary,'normalization':normalization,'registration_sha256':sha(out/'registration.json'),
            'trainer_sha256':sha('tools/train_partial_display.py'),'scope':reg['scope'],'release_authorized':False}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'summary':summary,'normalization':normalization},indent=2))


if __name__=='__main__':main()
