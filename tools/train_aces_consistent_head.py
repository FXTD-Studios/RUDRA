"""Freeze v7 except its final three highlight-radiance output channels."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint,reconstruction_weight
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.train_aces_restorer import merge_training_ancestry
from tools.summarize_aces_generated import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_consistent_head_20261024.json'))
    parser.add_argument('--out',type=Path,default=Path('G:/rudra_training/aces_consistent_head600_20261024'))
    parser.add_argument('--equal-image-loss',action='store_true');args=parser.parse_args()
    reg_path=args.registration;reg=json.loads(reg_path.read_text());root=args.out
    if root.exists():raise ValueError('new output required')
    pilot_path=Path(reg['pilot']);pilot=json.loads(pilot_path.read_text())
    summary=json.loads((pilot_path.parent.parent/'summary.json').read_text())
    if not summary['build_pass'] or any(r['split']!='train' for r in pilot['rows']):raise ValueError('accepted TRAIN corpus required')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2);torch.manual_seed(20261024);rng=np.random.default_rng(20261024)
    reference_view().set_threads(4)
    state=torch.load(reg['parent'],map_location='cpu',weights_only=True)
    ancestry_path=Path('G:/rudra_training/aces_coverage_restorer4800_20261023/training_ancestry.json')
    if state['pilot_sha256']!=sha(ancestry_path):raise ValueError('parent ancestry mismatch')
    model=model_from_checkpoint(state).to('cuda').eval()
    for p in model.parameters():p.requires_grad_(False)
    head=torch.nn.Conv2d(model.predictor.head.in_channels,3,3,padding=1).to('cuda')
    with torch.no_grad():
        head.weight.copy_(model.predictor.head.weight[3:]);head.bias.copy_(model.predictor.head.bias[3:])
    root.mkdir();(root/'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes());ancestry=root/'training_ancestry.json'
    ancestry.write_text(json.dumps({'rows':merge_training_ancestry(json.loads(ancestry_path.read_text())['rows'],pilot['rows'])},indent=2))
    cache=[];groups={};capture={}
    hook=model.predictor.head.register_forward_pre_hook(lambda module,args:capture.update(features=args[0].detach()))
    for row in pilot['rows']:
        if not row['highlight_pixels']:continue
        for p,h in ((row['sdr_path'],row['sdr_sha256']),(row['target_path'],row['target_sha256'])):
            if sha(p)!=h:raise ValueError('input identity mismatch')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
        loaded_weight=reconstruction_weight(torch.from_numpy(source).permute(2,0,1)[None])[0,0].numpy()
        mask=(loaded_weight>0)&(source.max(-1)>.5)
        if int(mask.sum())!=row['highlight_pixels']:raise ValueError('saved source/support differs from builder')
        if np.abs(reference_view().forward(target)-source).max()*255>1:raise ValueError('saved target/source view mismatch')
        mask[[0,-1],:]=False;mask[:,[0,-1]]=False
        points=np.argwhere(mask)
        if not len(points):continue
        with torch.inference_mode():
            result=model(torch.from_numpy(source).permute(2,0,1)[None].to('cuda'))
            features=capture['features'][0].cpu().numpy().copy()
            restored=result.restored_acescg[0].cpu().numpy().copy()
            weights=result.reconstruction_weight[0,0].cpu().numpy().copy()*mask
            initial=result.restored_acescg+(result.restored_acescg.abs().amax(1,keepdim=True)+.001)*result.reconstruction_weight*head(capture['features'])
            valid=torch.from_numpy(mask)[None,None].to('cuda').expand_as(initial)
            if not torch.allclose(initial[valid],result.acescg[valid],rtol=1e-4,atol=1e-4):raise ValueError('cached-head composition mismatch')
        index=len(cache);groups.setdefault(row['scene_id'],[]).append(index)
        cache.append((features,restored,target.transpose(2,0,1).copy(),weights,points,row['asset_id']))
        print('cached',len(cache),row['scene_id'],flush=True)
    hook.remove()
    if len(groups)<3:raise ValueError('insufficient TRAIN highlight scenes')
    labels=sorted(groups);visits=np.zeros(len(cache),int);optimizer=torch.optim.Adam(head.parameters(),lr=.0001)
    history=[];start=time.time()
    for step in range(1,601):
        batches=[[],[],[],[]]
        for _ in range(4):
            scene=labels[int(rng.integers(len(labels)))];index=int(rng.choice(groups[scene]))
            features,restored,target,mask,points,_=cache[index];h,w=mask.shape
            py,px=points[int(rng.integers(len(points)))];y=int(np.clip(py-rng.integers(16,176),0,h-192));x=int(np.clip(px-rng.integers(16,176),0,w-192))
            if min(h,w)<192:raise ValueError('crop exceeds image')
            for bucket,array in zip(batches,(features,restored,target,mask[None])):
                bucket.append(torch.from_numpy(array[:,y:y+192,x:x+192].copy()))
            visits[index]+=1
        features,restored,target,mask=[torch.stack(b).to('cuda') for b in batches]
        predicted=restored+(restored.abs().amax(1,keepdim=True)+.001)*mask*head(features)
        error=(torch.asinh(predicted)-torch.asinh(target)).square().mean(1,keepdim=True)[:,:,1:-1,1:-1]
        valid=mask[:,:,1:-1,1:-1]>0
        loss=((error*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)).mean() if args.equal_image_loss else (error*valid).sum()/valid.sum().clamp_min(1)
        if reg.get('gradient_loss_weight',0):
            from training.aces_detail_loss import highlight_gradient_loss
            loss=loss+reg['gradient_loss_weight']*highlight_gradient_loss(predicted[:,:,1:-1,1:-1],target[:,:,1:-1,1:-1],valid)
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in head.parameters()):raise ValueError('nonfinite gradient')
        optimizer.step()
        if step==1 or step%100==0:
            item={'step':step,'loss':float(loss.detach()),'seconds':time.time()-start};history.append(item);print(item,flush=True)
    with torch.no_grad():model.predictor.head.weight[3:].copy_(head.weight);model.predictor.head.bias[3:].copy_(head.bias)
    final={k:v.detach().cpu() for k,v in model.state_dict().items()}
    frozen=all(torch.equal(v[:3] if k in ('predictor.head.weight','predictor.head.bias') else v,
        state['model'][k][:3] if k in ('predictor.head.weight','predictor.head.bias') else state['model'][k]) for k,v in final.items())
    if not frozen or any(not torch.isfinite(v).all() for v in final.values()):raise ValueError('frozen/finite validation failed')
    recipe={'registration_sha256':sha(reg_path),'trainer_sha256':sha(__file__),'parent_sha256':sha(reg['parent']),
            'pilot_sha256':sha(pilot_path),'seed':20261024,'steps':600,'learning_rate':.0001,'cached_full_frame_features':True,
            'equal_image_loss':args.equal_image_loss,'gradient_loss_weight':reg.get('gradient_loss_weight',0)}
    checkpoint=root/'step600.pt';torch.save({'model':final,'contract':model.contract(),'pilot_sha256':sha(ancestry),
         'step':600,'recipe':recipe,'research_only':True,'optimizer':optimizer.state_dict()},checkpoint)
    (root/'report.json').write_text(json.dumps({'recipe':recipe,'history':history,'frozen_weights_bit_exact':frozen,
         'checkpoint_sha256':sha(checkpoint),'eligible_scenes':len(groups),'visits':{entry[-1]:int(v) for entry,v in zip(cache,visits)}},indent=2))
    summary['used_for_optimization']=True
    summary.setdefault('optimizer_reports',[])
    if summary.get('optimizer_report') and summary['optimizer_report'] not in summary['optimizer_reports']:summary['optimizer_reports'].append(summary['optimizer_report'])
    summary['optimizer_reports'].append(str(root/'report.json'))
    (pilot_path.parent.parent/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
