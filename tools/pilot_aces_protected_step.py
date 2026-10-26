"""Independent TRAIN-only fresh-Adam and projected-update probes; no saved model."""
import argparse,itertools,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint
from training.aces_detail_loss import highlight_gradient_loss
from training.aces_active_temporal import active_temporal_loss
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def project_direction(direction,gradients):
    """Nearest direction satisfying three nonnegative gradient dot products."""
    d=np.asarray(direction,dtype=np.float64);g=np.asarray(gradients,dtype=np.float64)
    norms=np.linalg.norm(g,axis=1);g=g[norms>0]/norms[norms>0,None]
    candidates=[np.zeros_like(d)];tol=1e-10*max(np.linalg.norm(d),1e-12)
    for count in range(len(g)+1):
        for indices in itertools.combinations(range(len(g)),count):
            a=g[list(indices)]
            candidate=d-a.T@np.linalg.lstsq(a@a.T,a@d,rcond=1e-12)[0] if count else d.copy()
            if np.all(g@candidate>=-tol):candidates.append(candidate)
    return min(candidates,key=lambda v:float(np.square(v-d).sum()))


def acceptable(before,after,rtol,atol):
    tolerance=atol+rtol*np.abs(before)
    return bool(np.all(after<=before+tolerance) and after[2]<before[2]-tolerance[2])


def objectives(model,x,target):
    o=model(x);pred=o.acescg;full=(o.reconstruction_weight>0)&(x.amin(1,keepdim=True)>=254.5/255)
    high=(o.reconstruction_weight>0)&(x.amax(1,keepdim=True)>.5);valid=full.clone()
    valid[:,:,:4]=False;valid[:,:,-4:]=False;valid[:,:,:,:4]=False;valid[:,:,:,-4:]=False
    if not valid.flatten(1).any(1).all():raise ValueError('empty supervision')
    error=(torch.asinh(pred)-torch.asinh(target)).square().mean(1,keepdim=True)
    photo=((error*valid).flatten(1).sum(1)/valid.flatten(1).sum(1)).mean()
    detail=highlight_gradient_loss(pred,target,valid);temporal,count,_=active_temporal_loss(pred,target,high,full)
    if not count:raise ValueError('empty temporal')
    return (photo,detail,temporal)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();reg=json.loads(args.registration.read_text())
    for key in ('checkpoint','pilot'):
        if sha(reg[key])!=reg[key+'_sha256']:raise ValueError('input identity')
    args.out.mkdir(exist_ok=False);(args.out/'pilot.py').write_bytes(Path(__file__).read_bytes());(args.out/'registration.json').write_bytes(args.registration.read_bytes())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=torch.load(reg['checkpoint'],map_location='cpu',weights_only=True);model=model_from_checkpoint(state).eval().to('cuda');params=list(model.full_context.parameters())
    original=torch.nn.utils.parameters_to_vector(params).detach().clone();frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('full_context.')}
    groups={}
    for r in json.loads(Path(reg['pilot']).read_text())['rows']:
        if r['split']!='train':raise ValueError('TRAIN only')
        groups.setdefault(r['group_id'],[]).append(r)
    rows=[]
    def set_vector(v):
        with torch.no_grad():torch.nn.utils.vector_to_parameters(v.clone(),params)
    def values(x,y):
        with torch.no_grad():return np.array([float(v) for v in objectives(model,x,y)])
    for group in reg['groups']:
        items=sorted(groups[group],key=lambda r:r['frame'])
        if [r['translation_xy'] for r in items]!=[[0,0],[.5,.25],[1,.5]]:raise ValueError('motion')
        xs=[];ys=[]
        for r in items:
            for k in ('sdr','target'):
                if sha(r[k+'_path'])!=r[k+'_sha256']:raise ValueError('source identity')
            xs.append(canonical_to_source(load_rgb(r['sdr_path'],hdr=False)));ys.append(np.load(r['target_path']))
        x=torch.from_numpy(np.stack(xs)).permute(0,3,1,2).to('cuda');y=torch.from_numpy(np.stack(ys)).permute(0,3,1,2).to('cuda')
        for first in (0,1):
            set_vector(original);xx=x[first:first+2];yy=y[first:first+2];other=1-first
            before_other=values(x[other:other+2],y[other:other+2]);losses=objectives(model,xx,yy);before=np.array([float(v.detach()) for v in losses]);grads=[]
            for i,loss in enumerate(losses):
                grads.append(torch.cat([g.flatten() for g in torch.autograd.grad(loss,params,retain_graph=i<2)]))
            total=sum(grads);optimizer=torch.optim.Adam(params,lr=reg['learning_rate']);offset=0
            for p in params:p.grad=total[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
            optimizer.step();adam=torch.nn.utils.parameters_to_vector(params).detach().clone();direction=(original-adam).double().cpu().numpy()
            adam_after=values(xx,yy);adam_other=values(x[other:other+2],y[other:other+2])
            adam_trials=[];adam_accepted=None
            if reg.get('matched_adam_search',False):
                for scale in reg['scales']:
                    set_vector(adam if scale==1 else original+(adam-original)*scale)
                    after=values(xx,yy);passed=acceptable(before,after,reg['relative_tolerance'],reg['absolute_tolerance'])
                    adam_trials.append({'scale':scale,'after':after.tolist(),'pass':passed})
                    if passed:
                        adam_accepted={'scale':scale,'after':after.tolist(),'other_after':values(x[other:other+2],y[other:other+2]).tolist()};break
            projected=project_direction(direction,torch.stack(grads).double().cpu().numpy());trials=[];accepted=None
            for scale in reg['scales']:
                set_vector(original-torch.from_numpy(projected*scale).to(original));after=values(xx,yy)
                passed=acceptable(before,after,reg['relative_tolerance'],reg['absolute_tolerance'])
                trials.append({'scale':scale,'after':after.tolist(),'pass':passed})
                if passed:
                    accepted={'scale':scale,'after':after.tolist(),'other_after':values(x[other:other+2],y[other:other+2]).tolist()};break
            rows.append({'group':group,'desired_fraction':items[0]['desired_fully_clipped_fraction'],'first':first,'before':before.tolist(),'other_before':before_other.tolist(),'adam_after':adam_after.tolist(),'adam_other_after':adam_other.tolist(),'adam_pass':acceptable(before,adam_after,reg['relative_tolerance'],reg['absolute_tolerance']),'projected_norm_ratio':float(np.linalg.norm(projected)/max(np.linalg.norm(direction),1e-30)),'projected_gradient_dots':(torch.stack(grads).double().cpu().numpy()@projected).tolist(),'trials':trials,'accepted':accepted})
            rows[-1].update(adam_trials=adam_trials,adam_accepted=adam_accepted)
            set_vector(original)
        print('groups',len(rows)//2,flush=True)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise ValueError('frozen parent changed')
    if not torch.equal(original,torch.nn.utils.parameters_to_vector(params)):raise ValueError('branch not restored')
    def passed_other(r,key):
        after=r['adam_other_after'] if key=='adam' else r['accepted']['other_after']
        return acceptable(np.array(r['other_before']),np.array(after),reg['relative_tolerance'],reg['absolute_tolerance'])
    accepted=[r for r in rows if r['accepted'] is not None]
    summary={'probes':len(rows),'adam_all_three_pass':sum(r['adam_pass'] for r in rows),'projected_all_three_pass':len(accepted),'projected_stalled':len(rows)-len(accepted),'projected_accepted_other_pair_pass':sum(passed_other(r,'projected') for r in accepted),'adam_other_pair_pass':sum(passed_other(r,'adam') for r in rows),'median_retained_direction_norm':float(np.median([r['projected_norm_ratio'] for r in rows])),'frozen_parent_exact':True,'original_branch_restored':True}
    if reg.get('matched_adam_search',False):
        aa=[r for r in rows if r['adam_accepted'] is not None]
        summary.update(matched_adam_search_pass=len(aa),matched_adam_search_other_pass=sum(acceptable(np.array(r['other_before']),np.array(r['adam_accepted']['other_after']),reg['relative_tolerance'],reg['absolute_tolerance']) for r in aa),projection_only_pass=sum(r['accepted'] is not None and r['adam_accepted'] is None for r in rows),adam_search_only_pass=sum(r['accepted'] is None and r['adam_accepted'] is not None for r in rows))
    (args.out/'report.json').write_text(json.dumps({'summary':summary,'rows':rows,'registration_sha256':sha(args.registration),'pilot_sha256':sha(__file__),'scope':reg['scope'],'release_authorized':False},indent=2,allow_nan=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
