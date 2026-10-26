"""Registered TRAIN-only fitting sanity check, with matched objective arms."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint
from training.aces_context_model import aligned_pair_residual_loss
from training.aces_detail_loss import highlight_gradient_loss
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_single_pair_sanity_20261024.json'));parser.add_argument('--out',type=Path,default=Path('G:/rudra_training/aces_single_pair_sanity_20261024'))
    args=parser.parse_args();regpath=args.registration;reg=json.loads(regpath.read_text())
    if sha(reg['pilot'])!=reg['pilot_sha256'] or sha(reg['parent'])!=reg['parent_sha256']:raise ValueError('registered identities')
    rows={r['asset_id']:r for r in json.loads(Path(reg['pilot']).read_text())['rows']}
    selected=[rows[k] for k in reg['asset_ids']]
    if any(r['split']!='train' or r['group_id']!=reg['group_id'] for r in selected):raise ValueError('TRAIN group')
    if [r['frame'] for r in selected]!=[0,1,2] or [r['translation_xy'] for r in selected]!=[[0,0],[.5,.25],[1,.5]]:raise ValueError('motion identity')
    root=args.out;root.mkdir(exist_ok=False)
    for path in (Path(__file__),Path('training/aces_context_model.py'),Path('training/aces_detail_loss.py')):(root/path.name).write_bytes(path.read_bytes())
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2);reference_view().set_threads(4);torch.manual_seed(reg['seed'])
    sources=[];targets=[]
    for r in selected:
        for k in ('sdr','target'):
            if sha(r[k+'_path'])!=r[k+'_sha256']:raise ValueError('source content')
        sources.append(canonical_to_source(load_rgb(r['sdr_path'],hdr=False)));targets.append(np.load(r['target_path']))
    source=torch.from_numpy(np.stack(sources)).permute(0,3,1,2).to('cuda');target=torch.from_numpy(np.stack(targets)).permute(0,3,1,2).to('cuda')
    state=torch.load(reg['parent'],map_location='cpu',weights_only=True);parent=model_from_checkpoint(state).to('cuda').eval()
    with torch.inference_mode():cached=parent(source)
    # Clone outside inference mode so tensors may be used by autograd.
    base=cached.baseline.clone();restored=cached.restored_acescg.clone();weight=cached.reconstruction_weight.clone()
    high=(weight>0)&(source.amax(1,keepdim=True)>.5);valid=high.clone();valid[:,:,:4]=False;valid[:,:,-4:]=False;valid[:,:,:,:4]=False;valid[:,:,:,-4:]=False
    full=valid&(source.amin(1,keepdim=True)>=254.5/255)
    def errors(pred):
        e=(torch.asinh(pred)-torch.asinh(target)).square().mean(1,keepdim=True)
        photo=(e*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)
        return photo,e
    def measure(pred):
        photo,e=errors(pred)
        return {'fit_photo':float(photo[:2].mean()),'phase_probe_photo':float(photo[2]),
                'full_photo':float((e*full).sum()/full.sum()) if full.any() else None,
                'gradient':float(highlight_gradient_loss(pred[:2],target[:2],valid[:2])),
                'temporal':float(aligned_pair_residual_loss(pred[:2],target[:2],high[:2]))}
    results={};initial=measure(cached.acescg);analytic=measure(base)
    for arm,losses in reg['arms'].items():
        torch.manual_seed(reg['seed']);model=model_from_checkpoint(state).to('cuda').train()
        optimizer=torch.optim.Adam(model.highlight_context.parameters(),lr=reg['learning_rate']);history=[];start=time.time()
        for step in range(1,reg['steps_per_arm']+1):
            pred,_=model.predict_highlights(source[:2],base[:2],restored[:2],weight[:2])
            e=(torch.asinh(pred)-torch.asinh(target[:2])).square().mean(1,keepdim=True)
            photo=((e*valid[:2]).flatten(1).sum(1)/valid[:2].flatten(1).sum(1).clamp_min(1)).mean()
            gradient=highlight_gradient_loss(pred,target[:2],valid[:2]);temporal=aligned_pair_residual_loss(pred,target[:2],high[:2])
            loss=photo+losses['gradient']*gradient+losses['temporal']*temporal
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.highlight_context.parameters()):raise ValueError('nonfinite gradient')
            optimizer.step()
            if step==1 or step%100==0:
                row={'step':step,'fit_photo':float(photo.detach()),'gradient':float(gradient.detach()),'temporal':float(temporal.detach())};history.append(row);print(arm,row,flush=True)
        with torch.inference_mode():final=model(source)
        frozen=all(torch.equal(v.cpu(),state['model'][k]) for k,v in model.state_dict().items() if not k.startswith('highlight_context.'))
        outside=torch.equal(final.acescg[~high.expand_as(source)],cached.acescg[~high.expand_as(source)])
        if not frozen or not outside:raise ValueError('isolation')
        np.save(root/(arm+'_prediction.npy'),final.acescg.permute(0,2,3,1).cpu().numpy())
        metrics=measure(final.acescg);results[arm]={'final':metrics,'history':history,'seconds':time.time()-start,'frozen_parent':frozen,'outside_highlight_exact':outside,'fit_error_halved':metrics['fit_photo']<=initial['fit_photo']*.5}
        del model,optimizer
    report={'initial':initial,'analytic':analytic,'arms':results,'registration_sha256':sha(regpath),'tool_sha256':sha(__file__),'scope':reg['diagnostic'],'release_authorized':False}
    (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({'initial':initial,'analytic':analytic,'final':{k:v['final'] for k,v in results.items()}},indent=2))


if __name__=='__main__':main()
