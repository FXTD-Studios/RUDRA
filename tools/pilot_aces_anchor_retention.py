"""TRAIN-only matched projected updates with an optional fixed retention panel."""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.pilot_aces_protected_step import project_direction
from tools.pilot_aces_protected_trajectory import restore
from tools.audit_aces_highlight_strata import strata, metrics
from tools.audit_aces_reference_detail import gradient_scores
from tools.summarize_aces_generated import sha
from training.aces_model import model_from_checkpoint
from training.aces_detail_loss import highlight_gradient_loss
from training.aces_active_temporal import active_temporal_loss
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from training.aces_panel_direction import project_panel, log_detail_loss


def retained(before, after, initial, rtol, atol):
    """Per-member protection, including fixed-baseline protection against drift."""
    if set(before) != set(after) or set(before) != set(initial):
        return False
    for key in before:
        b, a, i = before[key], after[key], initial[key]
        if a['log_edges'] != b['log_edges'] or a['log_edges'] != i['log_edges']:
            return False
        for name in ('photo', 'detail', 'temporal', 'log_detail'):
            if a[name] is None or b[name] is None or i[name] is None:
                if not (a[name] is b[name] is i[name] is None): return False
                continue
            if not np.isfinite(a[name]): return False
            if any(a[name] > ref[name] + atol + rtol * abs(ref[name]) for ref in (b, i)):
                return False
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();reg=json.loads(a.registration.read_text())
    if sha(reg['checkpoint'])!=reg['checkpoint_sha256']:raise ValueError('checkpoint identity')
    pilots={}
    for domain,path in reg['pilots'].items():
        if sha(path)!=reg['pilot_hashes'][domain]:raise ValueError('pilot identity')
        pilots[domain]={r['asset_id']:r for r in json.loads(Path(path).read_text())['rows']}
    if len({r['scene'] for r in reg['selection']})!=len(reg['selection']):raise ValueError('scene overlap')
    a.out.mkdir(exist_ok=False);(a.out/'registration.json').write_bytes(a.registration.read_bytes())
    dependencies=[__file__,'training/aces_panel_direction.py','tools/pilot_aces_protected_step.py','tools/pilot_aces_protected_trajectory.py','tools/audit_aces_highlight_strata.py','tools/audit_aces_reference_detail.py','training/aces_detail_loss.py','training/aces_active_temporal.py','training/aces_full_context_model.py','training/aces_model.py']
    hashes={str(path):sha(path) for path in dependencies}
    for path in dependencies:(a.out/Path(path).name).write_bytes(Path(path).read_bytes())
    torch.set_num_threads(2);torch.manual_seed(reg['seed']);reference_view().set_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model_from_checkpoint(torch.load(reg['checkpoint'],map_location='cpu',weights_only=True)).eval().to('cuda')
    params=list(model.full_context.parameters());original=torch.nn.utils.parameters_to_vector(params).detach().clone()
    frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('full_context.')}
    batches=[]
    for selection in reg['selection']:
        xs=[];ys=[]
        if selection['domain']=='motion':
            motion=[pilots['motion'][asset] for asset in selection['assets']]
            if [r['frame'] for r in motion]!=[0,1] or [r['translation_xy'] for r in motion]!=[[0,0],[.5,.25]]:raise ValueError('known motion required')
        for asset in selection['assets']:
            row=pilots[selection['domain']][asset]
            if row['split']!='train' or row['scene_id']!=selection['scene']:raise ValueError('TRAIN identity')
            for key in ('sdr','target'):
                if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('content identity')
            xs.append(canonical_to_source(load_rgb(row['sdr_path'],hdr=False)));ys.append(np.load(row['target_path']))
        x=np.stack(xs);y=np.stack(ys)
        if selection['crop']:
            top,left,h,w=selection['crop'];x=x[:,top:top+h,left:left+w];y=y[:,top:top+h,left:left+w]
        batches.append((selection,torch.from_numpy(x.copy()).permute(0,3,1,2),torch.from_numpy(y.copy()).permute(0,3,1,2)))
    def terms(batch, report=False, panel_gradients=False):
        spec,x,y=batch;x=x.to('cuda');y=y.to('cuda');o=model(x);pred=o.acescg
        full=(o.reconstruction_weight>0)&(x.amin(1,keepdim=True)>=254.5/255);valid=full.clone()
        t,b,l,r=spec['borders'];valid[:,:,:t]=False;valid[:,:,-b:]=False;valid[:,:,:,:l]=False;valid[:,:,:,-r:]=False
        if not valid.flatten(1).any(1).all():raise ValueError('empty supervision')
        error=(torch.asinh(pred)-torch.asinh(y)).square().mean(1,keepdim=True)
        photo=((error*valid).flatten(1).sum(1)/valid.flatten(1).sum(1)).mean();detail=highlight_gradient_loss(pred,y,valid)
        values=[photo,detail]
        if spec['domain']=='motion':
            temporal,count,_=active_temporal_loss(pred,y,(o.reconstruction_weight>0)&(x.amax(1,keepdim=True)>.5),full)
            if not count:raise ValueError('empty temporal')
            values.append(temporal)
        if panel_gradients:
            log=log_detail_loss(pred,y,valid)
            if log is not None:values.append(log)
            return values
        if not report:return values
        pp=pred.detach().permute(0,2,3,1).cpu().numpy();tt=y.permute(0,2,3,1).cpu().numpy();xx=x.permute(0,2,3,1).cpu().numpy()
        mm=valid[:,0].cpu().numpy();scores=[gradient_scores(p,t,m) for p,t,m in zip(pp,tt,mm)]
        vv=[s['rmse_ev'] for s in scores if s.get('edges',0)]
        result={'photo':float(photo),'detail':float(detail),'temporal':float(values[2]) if len(values)==3 else None,'log_detail':float(np.mean(vv)) if vv else None,'log_edges':[s['edges'] for s in scores]}
        result['regions']={}
        for region in ('full_boundary','full_near','full_deep'):
            rr=[metrics(p,t,strata(s,m)[region]&m) for p,t,s,m in zip(pp,tt,xx,mm)]
            result['regions'][region]={k:float(np.mean([v[k] for v in rr if v[k] is not None])) if any(v[k] is not None for v in rr) else None for k in ('asinh_mse','gradient_mse')}
        return result
    evaluations=0
    def evaluate(indices):
        nonlocal evaluations
        with torch.no_grad():
            result={str(i):terms(batches[i],True) for i in indices};evaluations+=len(indices);return result
    anchors=[i for i,b in enumerate(batches) if b[0]['role']=='anchor'];updates=[i for i,b in enumerate(batches) if b[0]['role']=='update']
    if len(reg['order'])!=reg['attempts'] or len(anchors)!=6 or len(updates)!=6:raise ValueError('registered budget/panel')
    runs={}
    methods=reg.get('methods',['current_only','anchor_guard'])
    if any(m not in ('current_only','anchor_guard','anchor_direction') for m in methods):raise ValueError('method')
    for method in methods:
        with torch.no_grad():torch.nn.utils.vector_to_parameters(original.clone(),params)
        optimizer=torch.optim.Adam(params,lr=reg['learning_rate']);start=time.perf_counter();evaluations=0
        initial=evaluate(range(len(batches)));panel={str(i):initial[str(i)] for i in anchors};initial_panel=copy.deepcopy(panel);steps=[]
        for step,index in enumerate(reg['order'],1):
            batch=batches[updates[index]];weights=torch.nn.utils.parameters_to_vector(params).detach().clone();state=copy.deepcopy(optimizer.state_dict())
            losses=terms(batch);before=np.array([float(v.detach()) for v in losses]);grads=[]
            for j,loss in enumerate(losses):grads.append(torch.cat([v.flatten() for v in torch.autograd.grad(loss,params,retain_graph=j<len(losses)-1)]))
            if any(not torch.isfinite(v).all() for v in grads):raise ValueError('nonfinite gradient')
            total=sum(grads);offset=0
            for param in params:param.grad=total[offset:offset+param.numel()].reshape_as(param).clone();offset+=param.numel()
            optimizer.step();proposal=torch.nn.utils.parameters_to_vector(params).detach().clone()
            direction=project_direction((weights-proposal).double().cpu().numpy(),torch.stack(grads).double().cpu().numpy());trials=[];accepted=False;projection=None
            if method=='anchor_direction':
                with torch.no_grad():torch.nn.utils.vector_to_parameters(weights.clone(),params)
                panel_grads=list(grads)
                for anchor in anchors:
                    terms_panel=terms(batches[anchor],panel_gradients=True)
                    for j,loss in enumerate(terms_panel):
                        gg=torch.autograd.grad(loss,params,retain_graph=j<len(terms_panel)-1)
                        panel_grads.append(torch.cat([v.flatten() for v in gg]).detach())
                direction,projection=project_panel((weights-proposal).double().cpu().numpy(),torch.stack(panel_grads).double().cpu().numpy())
            for scale in reg['scales']:
                with torch.no_grad():
                    torch.nn.utils.vector_to_parameters((weights-torch.from_numpy(direction*scale).to(weights)).clone(),params)
                    after=np.array([float(v) for v in terms(batch)]);evaluations+=1
                tol=reg['atol']+reg['rtol']*np.abs(before);strict=2 if len(losses)==3 else 0
                local=bool(np.isfinite(after).all() and np.all(after<=before+tol) and after[strict]<before[strict]-tol[strict]);panel_ok=None
                failures=[]
                if local and method!='current_only':
                    candidate_panel=evaluate(anchors);panel_ok=retained(panel,candidate_panel,initial_panel,reg['rtol'],reg['atol'])
                    for member,value in candidate_panel.items():
                        if value['log_edges']!=panel[member]['log_edges']:failures.append(member+':log_edges')
                        for key in ('photo','detail','temporal','log_detail'):
                            if value[key] is None:continue
                            if any(value[key]>ref[member][key]+reg['atol']+reg['rtol']*abs(ref[member][key]) for ref in (panel,initial_panel)):
                                failures.append(member+':'+key)
                accepted=local and (method=='current_only' or panel_ok)
                trials.append({'scale':scale,'after':after.tolist(),'current_pass':local,'panel_pass':panel_ok,'panel_failures':failures,'accepted':bool(accepted)})
                if accepted:
                    if method!='current_only':panel=candidate_panel
                    break
            if not accepted:restore(params,optimizer,weights,state)
            steps.append({'step':step,'batch':updates[index],'before':before.tolist(),'accepted':bool(accepted),'trials':trials,'rollback_verified':not accepted,'projection':projection})
            print(method,step,'accepted',accepted,flush=True)
        final=evaluate(range(len(batches)));accepted_count=sum(s['accepted'] for s in steps)
        if any(int(v['step'])!=accepted_count for v in optimizer.state.values()):raise ValueError('optimizer counters')
        runs[method]={'initial':initial,'final':final,'steps':steps,'accepted':accepted_count,'attempts':len(steps),'evaluation_batches':evaluations,'anchor_gradient_batches':len(anchors)*len(steps) if method=='anchor_direction' else 0,'seconds':time.perf_counter()-start}
    if any(run['initial']!=runs[methods[0]]['initial'] for run in runs.values()):raise ValueError('initial mismatch')
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise ValueError('parent changed')
    with torch.no_grad():torch.nn.utils.vector_to_parameters(original.clone(),params)
    if not torch.equal(original,torch.nn.utils.parameters_to_vector(params)):raise ValueError('restore')
    result={'runs':runs,'registration_sha256':sha(a.registration),'source_hashes':hashes,'frozen_parent_exact':True,'original_branch_restored':True,'scope':reg['scope'],'release_authorized':False}
    (a.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False));print({k:{q:v[q] for q in ('accepted','attempts','evaluation_batches','seconds')} for k,v in runs.items()})


if __name__=='__main__':main()
