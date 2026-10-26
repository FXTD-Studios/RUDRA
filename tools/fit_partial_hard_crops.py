"""TRAIN-only diagnostic: finite parameter probes and bounded fixed-crop fitting."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_partial_objective_conflict import objective_terms, symmetric_parameter_probe
from tools.summarize_aces_generated import sha
from training.aces_model import model_from_checkpoint
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def main():
    base=Path('G:/rudra_training'); out=base/'partial_hard_crop_fit_20261025'
    out.mkdir(exist_ok=False)
    tail=base/'partial_training_fit_20261025/tail.json'
    checkpoint=base/'partial_pcgrad_deterministic_control400_20261025/step400.pt'
    scale=base/'partial_display_fixed_scale_20261025/report.json'
    selection=json.loads(tail.read_text())['uncapped']['top10'][:4]
    pilots=[Path('G:/datasets/corpora')/p/'train/report.json' for p in
            ('aces_consistent_exposures_v2_20261024','aces_temporal_context_pairs_20261024')]
    reg={'inputs':{str(p):sha(p) for p in [tail,checkpoint,scale,*pilots]},
         'selection':[{'asset_id':r['asset_id'],'origin':r['origin']} for r in selection],
         'steps':100,'lr':1e-4,'gradient_clip':1.,'seed':20261025,
         'arms':['zero_ev_only','photo_three_ev_fixed_scale'],
         'policy':'Top4 TRAIN contribution crops, fixed batch at uncapped checkpoint. Independent reset per arm, Adam no inherited moments. Diagnostic only; no temporal objective, DEV selection or release. No checkpoint qualifies from this test.'}
    (out/'registration.json').write_text(json.dumps(reg,indent=2))
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    rows={r['asset_id']:r for p in pilots for r in json.loads(p.read_text())['rows']}
    sources=[];targets=[]
    for crop in selection:
        r=rows[crop['asset_id']];assert r['split']=='train'
        for key in ('sdr','target'): assert sha(r[key+'_path'])==r[key+'_sha256']
        y,x=crop['origin'];s=canonical_to_source(load_rgb(r['sdr_path'],hdr=False));t=np.load(r['target_path'])
        sources.append(s[y:y+192,x:x+192].copy());targets.append(t[y:y+192,x:x+192].copy())
    torch.set_num_threads(2);reference_view().set_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x=torch.from_numpy(np.stack(sources)).permute(0,3,1,2).cuda()
    t=torch.from_numpy(np.stack(targets)).permute(0,3,1,2).cuda()
    denominators=json.loads(scale.read_text())['denominators']
    coeff={'photo':1.,**{f'ev{ev}':.5/3/denominators[str(ev)] for ev in (-3,0,3)}}
    result={'registration_sha256':sha(out/'registration.json'),'arms':{},'probes':[]}
    for arm in reg['arms']:
        torch.manual_seed(reg['seed'])
        model=model_from_checkpoint(torch.load(checkpoint,map_location='cpu',weights_only=True)).eval().cuda()
        for name,p in model.named_parameters():p.requires_grad_(name.startswith('highlight_context.'))
        params=[p for p in model.parameters() if p.requires_grad]
        frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
        def evaluate():
            with torch.no_grad():return {k:float(v) for k,v in objective_terms(model,x,t,False)[0].items()}
        initial=evaluate()
        if arm==reg['arms'][0]:
            for name in initial:
                terms,_=objective_terms(model,x,t,False)
                gradients=torch.autograd.grad(terms[name],params,allow_unused=True)
                g=torch.cat([(a if a is not None else torch.zeros_like(p)).flatten() for a,p in zip(gradients,params)]).detach().cpu().numpy().astype(np.float64)
                norm=float(np.linalg.norm(g)); direction=-g/max(norm,1e-30)
                for step in (1e-3,1e-4):
                    values=symmetric_parameter_probe(params,direction,step,evaluate)
                    measured=(values[1][name]-values[-1][name])/(2*step)
                    result['probes'].append({'term':name,'step':step,'predicted':-norm,'measured':measured,
                        'relative_error':abs(measured+norm)/max(norm,abs(measured),1e-8),
                        'positive_step_changes':{k:values[1][k]-initial[k] for k in initial}})
            assert evaluate()==initial
            (out/'probes.json').write_text(json.dumps(result['probes'],indent=2,allow_nan=False))
        optimizer=torch.optim.Adam(params,lr=reg['lr']);history=[]
        for step in range(reg['steps']):
            optimizer.zero_grad(set_to_none=True);terms,n=objective_terms(model,x,t,False)
            loss=terms['ev0'] if arm=='zero_ev_only' else sum(coeff[k]*v for k,v in terms.items())
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,reg['gradient_clip'],error_if_nonfinite=True);optimizer.step()
            history.append({'step':step+1,'before':{k:float(v.detach()) for k,v in terms.items()},'objective':float(loss.detach())})
            if (step+1)%20==0:print(arm,step+1,history[-1]['before'],flush=True)
        final=evaluate()
        assert all(torch.equal(p.detach().cpu(),frozen[n]) for n,p in model.named_parameters() if n in frozen)
        result['arms'][arm]={'initial':initial,'final':final,'history':history,'pixels':n,'frozen_parameters_exact':True}
        torch.save({'model_state':model.state_dict(),'diagnostic_only':True},out/(arm+'.pt'))
        (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        del model,optimizer
    print(json.dumps({k:{'initial':v['initial'],'final':v['final']} for k,v in result['arms'].items()},indent=2))


if __name__=='__main__':main()
