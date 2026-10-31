"""Bounded persistent-Adam TRAIN trajectories with transactional rejected steps."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.pilot_aces_protected_step import objectives,acceptable,project_direction
from training.aces_model import model_from_checkpoint
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def equal_state(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


def restore(params,optimizer,weights,state):
    with torch.no_grad():torch.nn.utils.vector_to_parameters(weights.clone(),params)
    optimizer.load_state_dict(copy.deepcopy(state));optimizer.zero_grad(set_to_none=True)
    if not torch.equal(torch.nn.utils.parameters_to_vector(params),weights) or not equal_state(optimizer.state_dict(),state):
        raise ValueError('rollback failed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();reg=json.loads(a.registration.read_text())
    for k in ('checkpoint','pilot'):
        if sha(reg[k])!=reg[k+'_sha256']:raise ValueError('input identity')
    a.out.mkdir(exist_ok=False)
    for name,path in [('pilot.py',__file__),('step_helper.py','tools/pilot_aces_protected_step.py')]:
        (a.out/name).write_bytes(Path(path).read_bytes())
    (a.out/'registration.json').write_bytes(a.registration.read_bytes())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=torch.load(reg['checkpoint'],map_location='cpu',weights_only=True);model=model_from_checkpoint(state).eval().to('cuda');params=list(model.full_context.parameters())
    original=torch.nn.utils.parameters_to_vector(params).detach().clone();frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('full_context.')}
    groups={}
    for r in json.loads(Path(reg['pilot']).read_text())['rows']:
        if r['split']!='train':raise ValueError('TRAIN only')
        groups.setdefault(r['group_id'],[]).append(r)
    def set_vector(v):
        with torch.no_grad():torch.nn.utils.vector_to_parameters(v.clone(),params)
    def values(x,y):
        with torch.no_grad():return np.array([float(v) for v in objectives(model,x,y)])
    rows=[]
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
            xx=x[first:first+2];yy=y[first:first+2];other=1-first
            for method in ('adam','projected'):
                set_vector(original);optimizer=torch.optim.Adam(params,lr=reg['learning_rate']);initial=values(xx,yy);other_initial=values(x[other:other+2],y[other:other+2]);steps=[]
                for step in range(1,reg['max_steps']+1):
                    weights=torch.nn.utils.parameters_to_vector(params).detach().clone();saved=copy.deepcopy(optimizer.state_dict());losses=objectives(model,xx,yy);before=np.array([float(v.detach()) for v in losses]);grads=[]
                    for i,loss in enumerate(losses):grads.append(torch.cat([g.flatten() for g in torch.autograd.grad(loss,params,retain_graph=i<2)]))
                    total=sum(grads)
                    if not torch.isfinite(total).all():raise ValueError('nonfinite gradient')
                    offset=0
                    for param in params:param.grad=total[offset:offset+param.numel()].reshape_as(param).clone();offset+=param.numel()
                    optimizer.step();proposal=torch.nn.utils.parameters_to_vector(params).detach().clone();direction=(weights-proposal).double().cpu().numpy()
                    if method=='projected':direction=project_direction(direction,torch.stack(grads).double().cpu().numpy())
                    trials=[];accepted=False
                    for scale in reg['scales']:
                        if method=='adam':set_vector(proposal if scale==1 else weights+(proposal-weights)*scale)
                        else:set_vector(weights-torch.from_numpy(direction*scale).to(weights))
                        after=values(xx,yy);passed=acceptable(before,after,reg['relative_tolerance'],reg['absolute_tolerance']);trials.append({'scale':scale,'after':after.tolist(),'pass':passed})
                        if passed:accepted=True;break
                    if not accepted:restore(params,optimizer,weights,saved)
                    steps.append({'step':step,'before':before.tolist(),'trials':trials,'accepted':accepted,'rollback_verified':not accepted})
                    if not accepted:break
                final=values(xx,yy);other_final=values(x[other:other+2],y[other:other+2]);accepted_count=sum(s['accepted'] for s in steps)
                state_steps=[int(v['step']) for v in optimizer.state.values()]
                if state_steps and any(n!=accepted_count for n in state_steps):raise ValueError('optimizer step counter')
                rows.append({'group':group,'first':first,'method':method,'desired_fraction':items[0]['desired_fully_clipped_fraction'],'initial':initial.tolist(),'final':final.tolist(),'other_initial':other_initial.tolist(),'other_final':other_final.tolist(),'accepted_steps':accepted_count,'attempts':len(steps),'steps':steps,'final_pass':acceptable(initial,final,reg['relative_tolerance'],reg['absolute_tolerance']),'other_pass':acceptable(other_initial,other_final,reg['relative_tolerance'],reg['absolute_tolerance']),'optimizer_step_count_verified':True})
                set_vector(original)
        print('groups',len(rows)//4,flush=True)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise ValueError('parent changed')
    if not torch.equal(original,torch.nn.utils.parameters_to_vector(params)):raise ValueError('branch not restored')
    summary={}
    for method in ('adam','projected'):
        rr=[r for r in rows if r['method']==method]
        summary[method]={'trajectories':len(rr),'accepted_steps':sum(r['accepted_steps'] for r in rr),'attempts':sum(r['attempts'] for r in rr),'complete_budget':sum(r['accepted_steps']==reg['max_steps'] for r in rr),'zero_updates':sum(r['accepted_steps']==0 for r in rr),'final_pass':sum(r['final_pass'] for r in rr),'other_pass':sum(r['other_pass'] for r in rr),'median_temporal_relative_change':float(np.median([(r['final'][2]-r['initial'][2])/r['initial'][2] for r in rr]))}
    (a.out/'report.json').write_text(json.dumps({'summary':summary,'rows':rows,'frozen_parent_exact':True,'original_branch_restored':True,'registration_sha256':sha(a.registration),'pilot_sha256':sha(__file__),'helper_sha256':sha('tools/pilot_aces_protected_step.py'),'scope':reg['scope'],'release_authorized':False},indent=2,allow_nan=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
