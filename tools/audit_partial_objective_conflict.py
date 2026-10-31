"""Frozen TRAIN parameter-gradient conflict, not an optimizer trajectory replay."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.train_partial_display import _Forward, forward_partial
from tools.summarize_aces_generated import sha
from training.aces_model import model_from_checkpoint, reconstruction_weight
from training.aces_reference import reference_view
from training.aces_active_temporal import active_temporal_loss
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def gradient_summary(gradients, coefficients):
    names=list(gradients); g=np.stack([gradients[n] for n in names]).astype(np.float64)
    if not np.isfinite(g).all(): raise ValueError('nonfinite parameter gradient')
    gram=g@g.T; norms=np.sqrt(np.maximum(np.diag(gram),0))
    cosine=np.divide(gram,norms[:,None]*norms[None,:],out=np.zeros_like(gram),where=(norms[:,None]*norms[None,:])>0)
    weights=np.array([coefficients[n] for n in names]); combined=weights@g; norm=float(np.linalg.norm(combined))
    directional=-(g@combined)/max(norm,1e-30)
    combined_cosine=np.divide(-directional,norms,out=np.zeros_like(norms),where=norms>0)
    return {'norms':dict(zip(names,norms.tolist())),
            'pair_cosines':{a+'|'+b:float(cosine[i,j]) for i,a in enumerate(names) for j,b in enumerate(names) if j>i},
            'weighted_sum_norm':norm,
            'unit_descent_directional_derivative':dict(zip(names,directional.tolist())),
            'adverse_terms':[n for n,c in zip(names,combined_cosine) if c < -1e-6]}


def objective_terms(model,x,t,motion):
    pred,mask,_=forward_partial(model,x);terms={}
    if motion:
        high=(reconstruction_weight(x)>0)&(x.amax(1,keepdim=True)>.5)
        terms['temporal']=torch.stack([active_temporal_loss(pred[i:i+2],t[i:i+2],high[i:i+2],mask[i:i+2],border=64)[0] for i in (0,2)]).mean()
    pred=pred[...,64:-64,64:-64];target=t[...,64:-64,64:-64];valid=mask[...,64:-64,64:-64].float();n=valid.sum()
    assert n>0
    terms['photo']=((torch.asinh(pred)-torch.asinh(target)).square().mean(1,keepdim=True)*valid).sum()/n
    for ev in (-3,0,3):
        with torch.no_grad():desired=_Forward.apply(target*2.**ev)
        terms[f'ev{ev}']=(((_Forward.apply(pred*2.**ev)-desired)*255).square().mean(1,keepdim=True)*valid).sum()/n
    return terms,int(n)


def symmetric_parameter_probe(params,direction,step,evaluate):
    """Always restore exact original tensor values, including on evaluator failure."""
    originals=[p.detach().clone() for p in params]
    chunks=[];offset=0
    for p in params:
        chunks.append(torch.as_tensor(direction[offset:offset+p.numel()],device=p.device,dtype=p.dtype).reshape_as(p))
        offset+=p.numel()
    if offset!=len(direction):raise ValueError('direction size')
    values={}
    try:
        with torch.no_grad():
            for sign in (1,-1):
                for p,original,d in zip(params,originals,chunks):p.copy_(original+sign*step*d)
                values[sign]=evaluate()
    finally:
        with torch.no_grad():
            for p,original in zip(params,originals):p.copy_(original)
        assert all(torch.equal(p,original) for p,original in zip(params,originals))
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parameter-probes',action='store_true');args=parser.parse_args()
    root=Path('G:/rudra_training/partial_temporal1_400_20261025')
    out=Path('G:/rudra_training/partial_parameter_probes_20261025' if args.parameter_probes else 'G:/rudra_training/partial_objective_conflict_20261025');out.mkdir(exist_ok=False)
    ckpts={'initial':Path('G:/rudra_training/aces_guarded_projected600_20261024/step600.pt'),
           'control':root/'step400.pt',
           'coverage':Path('G:/rudra_training/partial_coverage400_20261025/step400.pt'),
           'group_coverage':Path('G:/rudra_training/partial_group_coverage400_20261025/step400.pt')}
    pilots=[Path('G:/datasets/corpora')/p/'train/report.json' for p in ('aces_consistent_exposures_v2_20261024','aces_temporal_context_pairs_20261024')]
    scale_path=Path('G:/rudra_training/partial_display_fixed_scale_20261025/report.json')
    inputs=pilots+[root/n for n in ('crop_pool.json','motion_pool.json','schedule.json','report.json')]+[scale_path]
    steps=[0,1,100,101,200,201,398,399]
    reg={'checkpoints':{str(p):sha(p) for p in ckpts.values()},'inputs':{str(p):sha(p) for p in inputs},
         'zero_based_control_batches':steps,
         'policy':'Same eight saved TRAIN batches at four frozen weights; photo, individual display EV and temporal parameter gradients. Raw cosines; common fixed TRAIN normalization and control historical EMA at selected index. Neither normalization represents other checkpoints actual Adam steps. No updates or DEV inputs.',
         'release_authorized':False}
    if args.parameter_probes:
        reg['parameter_steps']=[.01,.001,.0001,.00001]
        reg['probe_policy']='Absolute parameter-L2 steps along negative unit weighted gradient. Symmetric derivatives and actual positive-step loss changes; exact restore after every probe. Same32cases and both normalizations; no Adam interpretation.'
    (out/'registration.json').write_text(json.dumps(reg,indent=2));(out/'auditor.py').write_bytes(Path(__file__).read_bytes())
    rows={r['asset_id']:r for p in pilots for r in json.loads(p.read_text())['rows']}
    pools=[json.loads((root/n).read_text()) for n in ('crop_pool.json','motion_pool.json')]
    schedule=json.loads((root/'schedule.json').read_text());hist=json.loads((root/'report.json').read_text())['history']
    fixed={float(k):v for k,v in json.loads(scale_path.read_text())['denominators'].items()}
    torch.set_num_threads(2);reference_view().set_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    batches=[]
    for index in steps:
        motion=index%2==0;sources=[];targets=[];selection=[]
        for crop_index in schedule[index]:
            crop=pools[int(motion)][crop_index];y,x=crop['origin']
            for asset in crop['assets'] if motion else [crop['asset_id']]:
                row=rows[asset];assert row['split']=='train'
                for key in ('sdr','target'):
                    assert sha(row[key+'_path'])==row[key+'_sha256']
                s=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));t=np.load(row['target_path'])
                sources.append(s[y:y+192,x:x+192].copy());targets.append(t[y:y+192,x:x+192].copy())
                selection.append({'asset_id':asset,'origin':[y,x]})
        batches.append((index,motion,torch.from_numpy(np.stack(sources)).permute(0,3,1,2).cuda(),
                        torch.from_numpy(np.stack(targets)).permute(0,3,1,2).cuda(),selection))
    records=[]
    for state,path in ckpts.items():
        model=model_from_checkpoint(torch.load(path,map_location='cpu',weights_only=True)).eval().cuda()
        for name,p in model.named_parameters():p.requires_grad_(name.startswith('highlight_context.'))
        params=[p for p in model.parameters() if p.requires_grad]
        before=[p.detach().clone() for p in model.parameters()]
        for index,motion,x,t,selection in batches:
            terms,n=objective_terms(model,x,t,motion)
            gradients={}
            for name,value in terms.items():
                grad=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
                gradients[name]=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for g,p in zip(grad,params)]).cpu().numpy()
            coeff={'photo':1.,'temporal':1.,**{f'ev{ev}':.5/3/fixed[ev] for ev in (-3,0,3)}}
            ema_coeff={'photo':1.,'temporal':1.,**{f'ev{ev}':.5/3/max(hist[index]['analytic_ema'][str(float(ev))],1e-4) for ev in (-3,0,3)}}
            records.append({'state':state,'batch_index':index,'domain':'motion' if motion else 'still','selection':selection,
                            'pixels':int(n),'losses':{k:float(v.detach()) for k,v in terms.items()},
                            'fixed_coefficients':coeff,'control_ema_coefficients':ema_coeff,
                            'fixed':gradient_summary(gradients,coeff),'control_ema':gradient_summary(gradients,ema_coeff)})
            if args.parameter_probes:
                probes=[]
                for norm,weights in (('fixed',coeff),('control_ema',ema_coeff)):
                    combined=sum(weights[k]*g.astype(np.float64) for k,g in gradients.items())
                    direction=-combined/max(float(np.linalg.norm(combined)),1e-30)
                    for step in reg['parameter_steps']:
                        def evaluate():
                            return {k:float(v) for k,v in objective_terms(model,x,t,motion)[0].items()}
                        values=symmetric_parameter_probe(params,direction,step,evaluate)
                        measurements={}
                        for name,g in gradients.items():
                            predicted=float(g.astype(np.float64)@direction)
                            measured=(values[1][name]-values[-1][name])/(2*step)
                            measurements[name]={'predicted':predicted,'measured':measured,
                                'relative_disagreement':abs(measured-predicted)/max(abs(measured),abs(predicted),1e-8),
                                'sign_disagreement':bool(measured*predicted<0),
                                'positive_step_loss_delta':values[1][name]-records[-1]['losses'][name]}
                        probes.append({'normalization':norm,'step':step,'terms':measurements})
                records[-1]['parameter_probes']=probes
                print('probed',state,index,flush=True)
        assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
        del model; print('audited',state,flush=True)
    summary={}
    for state in ckpts:
        selected=[r for r in records if r['state']==state]
        summary[state]={}
        for norm in ('fixed','control_ema'):
            names=sorted({n for r in selected for n in r[norm]['norms']})
            summary[state][norm]={'adverse_counts':{n:sum(n in r[norm]['adverse_terms'] for r in selected) for n in names},
                                 'evaluated_counts':{n:sum(n in r[norm]['norms'] for r in selected) for n in names}}
        pairs=sorted({n for r in selected for n in r['fixed']['pair_cosines']})
        summary[state]['pair_cosines']={n:{'negative_count':sum(r['fixed']['pair_cosines'][n]<-1e-6 for r in selected if n in r['fixed']['pair_cosines']),
                                         'median':float(np.median([r['fixed']['pair_cosines'][n] for r in selected if n in r['fixed']['pair_cosines']]))} for n in pairs}
    result={'summary':summary,'rows':records,'weights_unchanged':True,'registration_sha256':sha(out/'registration.json'),'auditor_sha256':sha(__file__)}
    if args.parameter_probes:
        result['probe_summary']={}
        for step in reg['parameter_steps']:
            measurements=[v for r in records for p in r['parameter_probes'] if p['step']==step for v in p['terms'].values()]
            errors=[v['relative_disagreement'] for v in measurements]
            result['probe_summary'][str(step)]={'count':len(measurements),'median_relative_disagreement':float(np.median(errors)),
                'p95_relative_disagreement':float(np.percentile(errors,95)),
                'sign_disagreements':sum(v['sign_disagreement'] for v in measurements),
                'positive_step_increases':sum(v['positive_step_loss_delta']>0 for v in measurements)}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(summary,indent=2))
    if args.parameter_probes:print(json.dumps(result['probe_summary'],indent=2))


if __name__=='__main__':main()
