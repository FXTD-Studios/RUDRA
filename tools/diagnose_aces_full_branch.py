"""Registered three-arm TRAIN fitting pilot for isolated full-highlight routing."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint
from training.aces_full_context_model import ACESFullContextModel,fully_clipped_support
from training.aces_context_model import aligned_pair_residual_loss
from training.aces_detail_loss import highlight_gradient_loss
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def boundary_error(pred,target,high,full):
    residual=torch.asinh(pred)-torch.asinh(target);total=pred.new_zeros(());count=0
    for dim in (2,3):
        a=[slice(None)]*4;b=a.copy();a[dim]=slice(None,-1);b[dim]=slice(1,None)
        mask=high[tuple(a)]&high[tuple(b)]&(full[tuple(a)]!=full[tuple(b)])
        total+=(torch.diff(residual,dim=dim).square().mean(1,keepdim=True)*mask).sum();count+=int(mask.sum())
    return float(total/count) if count else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();reg=json.loads(a.registration.read_text())
    for key in ('parent','pilot'):
        if sha(reg[key])!=reg[key+'_sha256']:raise ValueError('registered identity')
    a.out.mkdir(exist_ok=False)
    for file in (__file__,'training/aces_full_context_model.py','training/aces_context_model.py','training/aces_detail_loss.py','training/aces_model.py'):(a.out/Path(file).name).write_bytes(Path(file).read_bytes())
    (a.out/'registration.json').write_bytes(a.registration.read_bytes())
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True);torch.set_num_threads(2);reference_view().set_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest={r['asset_id']:r for r in json.loads(Path(reg['pilot']).read_text())['rows']};rows=[manifest[k] for k in reg['asset_ids']]
    if any(r['split']!='train' or r['group_id']!=reg['group_id'] for r in rows):raise ValueError('TRAIN group')
    if [r['frame'] for r in rows]!=[0,1,2] or [r['translation_xy'] for r in rows]!=[[0,0],[.5,.25],[1,.5]]:raise ValueError('pair identity')
    sources=[];targets=[]
    for r in rows:
        for k in ('sdr','target'):
            if sha(r[k+'_path'])!=r[k+'_sha256']:raise ValueError('input identity')
        sources.append(canonical_to_source(load_rgb(r['sdr_path'],hdr=False)));targets.append(np.load(r['target_path']))
    source=torch.from_numpy(np.stack(sources)).permute(0,3,1,2).to('cuda');target=torch.from_numpy(np.stack(targets)).permute(0,3,1,2).to('cuda')
    state=torch.load(reg['parent'],map_location='cpu',weights_only=True);parent=model_from_checkpoint(state).eval().to('cuda')
    with torch.no_grad():cached=parent(source)
    base,restored,weight=cached.baseline,cached.restored_acescg,cached.reconstruction_weight
    high=(weight>0)&(source.amax(1,keepdim=True)>.5);full=fully_clipped_support(source,weight);valid=high.clone()
    valid[:,:,:4]=False;valid[:,:,-4:]=False;valid[:,:,:,:4]=False;valid[:,:,:,-4:]=False
    def measure(pred):
        error=(torch.asinh(pred)-torch.asinh(target)).square().mean(1,keepdim=True);result={}
        for name,mask in (('high',valid),('full',valid&full),('partial',valid&~full)):
            values=(error*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)
            result[name+'_fit']=float(values[:2].mean());result[name+'_phase_probe']=float(values[2])
        result.update(boundary_mse=boundary_error(pred[:2],target[:2],valid[:2],full[:2]),
            high_temporal=float(aligned_pair_residual_loss(pred[:2],target[:2],high[:2])),
            full_gradient=float(highlight_gradient_loss(pred[:2],target[:2],(valid&full)[:2])))
        return result
    initial=measure(cached.acescg);results={};predictions={}
    for arm in reg['arms']:
        torch.manual_seed(reg['seed']);isolated=arm=='isolated_full';model=(ACESFullContextModel.from_parent(parent) if isolated else model_from_checkpoint(state)).to('cuda').train()
        with torch.no_grad():assert torch.equal(model(source).acescg,cached.acescg)
        branch=model.full_context if isolated else model.highlight_context;optimizer=torch.optim.Adam(branch.parameters(),lr=reg['learning_rate']);history=[];start=time.time()
        mask=valid if arm=='shared_balanced' else valid&full;temporal_mask=high if arm=='shared_balanced' else full
        if not mask[:2].flatten(1).any(1).all():raise ValueError('empty fit support')
        for step in range(1,reg['steps_per_arm']+1):
            pred,_=model.predict_highlights(source[:2],base[:2],restored[:2],weight[:2])
            error=(torch.asinh(pred)-torch.asinh(target[:2])).square().mean(1,keepdim=True)
            photo=((error*mask[:2]).flatten(1).sum(1)/mask[:2].flatten(1).sum(1)).mean()
            gradient=highlight_gradient_loss(pred,target[:2],mask[:2]);temporal=aligned_pair_residual_loss(pred,target[:2],temporal_mask[:2]);loss=photo+gradient+temporal
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in branch.parameters()):raise ValueError('nonfinite gradient')
            optimizer.step()
            if step==1 or step%100==0:
                item={'step':step,'photo':float(photo.detach()),'gradient':float(gradient.detach()),'temporal':float(temporal.detach())};history.append(item);print(arm,item,flush=True)
        with torch.no_grad():final=model(source).acescg
        frozen=all(torch.equal(v.cpu(),state['model'][k]) for k,v in model.state_dict().items() if k in state['model'] and (isolated or not k.startswith('highlight_context.')))
        support=full if isolated else high;outside=torch.equal(final[~support.expand_as(final)],cached.acescg[~support.expand_as(final)])
        if not frozen or not outside:raise ValueError('frozen/outside invariant')
        metrics=measure(final);predictions[arm]=final.detach().cpu();np.save(a.out/(arm+'_prediction.npy'),final.permute(0,2,3,1).cpu().numpy())
        results[arm]={'final':metrics,'history':history,'seconds':time.time()-start,'frozen_parent':frozen,'outside_support_exact':outside,
            'full_fit_halved':metrics['full_fit']<=.5*initial['full_fit'],'boundary_preserved':metrics['boundary_mse']<=initial['boundary_mse'],
            'high_temporal_preserved':metrics['high_temporal']<=initial['high_temporal']}
        del model,optimizer
    m=full.cpu().expand_as(predictions['isolated_full']);difference=float((predictions['isolated_full'][m]-predictions['shared_full'][m]).abs().max())
    report={'initial':initial,'analytic':measure(base),'arms':results,'isolated_vs_shared_full_max_abs':difference,
            'registration_sha256':sha(a.registration),'tool_sha256':sha(__file__),'scope':reg['scope'],'release_authorized':False}
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:report[k] for k in ('initial','isolated_vs_shared_full_max_abs')},indent=2))


if __name__=='__main__':main()
