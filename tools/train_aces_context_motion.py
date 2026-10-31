"""Train only the independent highlight branch on registered known-motion pairs."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_context_model import ACESContextModel,aligned_pair_residual_loss
from training.aces_detail_loss import highlight_gradient_loss
from training.aces_context_batches import stack_context_arrays,sample_replay_crop
from training.aces_model import model_from_checkpoint
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha
from tools.train_aces_restorer import merge_training_ancestry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_context_motion600_20261024.json'))
    parser.add_argument('--out',type=Path,default=Path('G:/rudra_training/aces_context_motion600_20261024'))
    args=parser.parse_args();regpath=args.registration;reg=json.loads(regpath.read_text())
    if reg.get('replay_sampling','legacy') not in ('legacy','nonempty_phase_aligned_v2'):raise ValueError('unknown replay sampler')
    pilotpath=Path(reg['pilot']);pilot=json.loads(pilotpath.read_text());summarypath=pilotpath.parent.parent/'summary.json'
    summary=json.loads(summarypath.read_text())
    if not summary['build_pass'] or any(r['split']!='train' for r in pilot['rows']):raise ValueError('valid TRAIN corpus required')
    if sha(reg['parent'])!=reg['parent_sha256']:raise ValueError('parent identity')
    replay=json.loads(Path(reg['replay_pilot']).read_text()) if reg.get('replay_pilot') else {'rows':[]}
    if any(r['split']!='train' for r in replay['rows']):raise ValueError('TRAIN-only replay')
    root=args.out;root.mkdir(exist_ok=False)
    (root/'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (root/'model_snapshot.py').write_bytes(Path('training/aces_context_model.py').read_bytes())
    (root/'batch_preparation_snapshot.py').write_bytes(Path('training/aces_context_batches.py').read_bytes())
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2);torch.manual_seed(reg['seed']);rng=np.random.default_rng(reg['seed']);reference_view().set_threads(4)
    state=torch.load(reg['parent'],map_location='cpu',weights_only=True);parent=model_from_checkpoint(state).eval().to('cuda')
    if parent.contract().get('full_context') is not None:raise ValueError('isolated full branch requires its own registered trainer')
    ancestry_path=Path(reg['parent']).parent/'training_ancestry.json'
    if state['pilot_sha256']!=sha(ancestry_path):raise ValueError('ancestry identity')
    ancestry=root/'training_ancestry.json';ancestry.write_text(json.dumps({'rows':merge_training_ancestry(json.loads(ancestry_path.read_text())['rows'],pilot['rows']+replay['rows'])},indent=2))
    model=(model_from_checkpoint(state) if isinstance(parent,ACESContextModel) else ACESContextModel.from_parent(parent)).to('cuda').train()
    cache=[];groups={};scenes={};points={};metadata={};excluded_replay=[]
    replay_v2=reg.get('replay_sampling')=='nonempty_phase_aligned_v2'
    tagged=[('motion',r) for r in pilot['rows']]+[('replay',r) for r in replay['rows'] if r['highlight_pixels']>0]
    for domain,row in tagged:
        if sha(row['sdr_path'])!=row['sdr_sha256'] or sha(row['target_path'])!=row['target_sha256']:raise ValueError('input identity')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
        with torch.inference_mode():
            value=parent(torch.from_numpy(source).permute(2,0,1)[None].to('cuda'))
        arrays=[source.transpose(2,0,1),value.baseline[0].cpu().numpy(),value.restored_acescg[0].cpu().numpy(),
                value.reconstruction_weight[0].cpu().numpy(),target.transpose(2,0,1)]
        if domain=='replay':
            eligible=(arrays[3][0]>0)&(source.max(-1)>.5)
            if replay_v2:
                eligible=eligible.copy();eligible[:4]=False;eligible[-4:]=False;eligible[:,:4]=False;eligible[:,-4:]=False
            replay_points=np.argwhere(eligible)
            if replay_v2 and not len(replay_points):
                excluded_replay.append(row['asset_id']);continue
        cache.append(tuple(np.ascontiguousarray(x) for x in arrays));index=len(cache)-1
        group=(domain,row.get('group_id',row['asset_id']));groups.setdefault(group,[]).append(index);metadata[index]=row
        scenes.setdefault(domain,{}).setdefault(row['scene_id'],set()).add(group)
        if domain=='replay':points[index]=replay_points
        if len(cache)%18==0:print('cached',len(cache),flush=True)
    if replay['rows'] and 'replay' not in scenes:raise ValueError('registered replay has no eligible supervision')
    for (domain,_),indices in groups.items():
        if domain=='motion' and ([metadata[i]['frame'] for i in indices]!=[0,1,2] or [metadata[i]['translation_xy'] for i in indices]!=[[0,0],[.5,.25],[1,.5]]):raise ValueError('known-motion metadata mismatch')
        if domain=='replay' and len(indices)!=1:raise ValueError('single-frame replay required')
    del parent;domains=sorted(scenes);labels={d:sorted(scenes[d]) for d in domains};scenes={d:{s:sorted(g) for s,g in ss.items()} for d,ss in scenes.items()}
    optimizer=torch.optim.Adam(model.highlight_context.parameters(),lr=reg['learning_rate']);history=[];visits={d+':'+s:0 for d in domains for s in labels[d]};start=time.time()
    for step in range(1,reg['steps']+1):
        domain=domains[int(rng.integers(len(domains)))];scene=labels[domain][int(rng.integers(len(labels[domain])))];group=scenes[domain][scene][int(rng.integers(len(scenes[domain][scene])))];first=int(rng.integers(2)) if domain=='motion' else 0
        indices=groups[group][first:first+2];visits[domain+':'+scene]+=1;borders=[4,4,4,4]
        crop=None
        if domain=='replay':
            index=indices[0];h,w=cache[index][0].shape[-2:]
            if h<320 or w<512 or not len(points[index]):raise ValueError('replay crop/support')
            if replay_v2:crop,borders=sample_replay_crop(h,w,points[index],rng)
            else:
                py,px=points[index][int(rng.integers(len(points[index])))];y=int(np.clip(py-rng.integers(64,256),0,h-320))//4*4;x=int(np.clip(px-rng.integers(64,448),0,w-512))//4*4
                crop=(y,x,320,512);borders=[64 if y else 4,64 if y+320<h else 4,64 if x else 4,64 if x+512<w else 4]
        arrays=stack_context_arrays(cache,indices,crop)
        source,base,restored,weight,target=[torch.from_numpy(np.ascontiguousarray(v)).to('cuda') for v in arrays]
        high=(weight>0)&(source.amax(1,keepdim=True)>.5);valid=high.clone()
        top,bottom,left,right=borders;valid[:,:,:top]=False;valid[:,:,-bottom:]=False;valid[:,:,:,:left]=False;valid[:,:,:,-right:]=False
        if replay_v2 and domain=='replay' and not valid.any():raise ValueError('empty replay supervision')
        prediction,_=model.predict_highlights(source,base,restored,weight)
        error=(torch.asinh(prediction)-torch.asinh(target)).square().mean(1,keepdim=True)
        photo=((error*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)).mean()
        gradient=highlight_gradient_loss(prediction,target,valid)
        temporal=aligned_pair_residual_loss(prediction,target,high) if domain=='motion' else photo*0
        loss=photo+reg['loss']['highlight_gradient']*gradient+reg['loss']['known_aligned_temporal_residual']*temporal
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.highlight_context.parameters()):raise ValueError('nonfinite gradient')
        optimizer.step()
        if step==1 or step%100==0:
            row={'step':step,'domain':domain,'loss':float(loss.detach()),'photo':float(photo.detach()),'gradient':float(gradient.detach()),'temporal':float(temporal.detach()),'seconds':time.time()-start}
            history.append(row);print(row,flush=True)
    final={k:v.detach().cpu() for k,v in model.state_dict().items()}
    frozen=all(torch.equal(final[k],v) for k,v in state['model'].items() if not k.startswith('highlight_context.'))
    if not frozen or any(not torch.isfinite(v).all() for v in final.values()):raise ValueError('frozen/finite invariant')
    recipe={'registration_sha256':sha(regpath),'pilot_sha256':sha(pilotpath),'parent_sha256':sha(reg['parent']),
            'replay_pilot_sha256':sha(reg['replay_pilot']) if replay['rows'] else None,
            'trainer_sha256':sha(__file__),'batch_preparation_sha256':sha('training/aces_context_batches.py'),'replay_sampling':reg.get('replay_sampling','legacy'),'excluded_replay_without_interior_highlights':excluded_replay,'model_code_sha256':sha('training/aces_context_model.py'),'steps':reg['steps'],'seed':reg['seed']}
    checkpoint=root/f'step{reg["steps"]}.pt';torch.save({'model':final,'contract':model.contract(),'pilot_sha256':sha(ancestry),'recipe':recipe,'research_only':True},checkpoint)
    (root/'report.json').write_text(json.dumps({'recipe':recipe,'history':history,'frozen_parent_bit_exact':frozen,'frozen_scope':'all parent keys except independent highlight_context.*','checkpoint_sha256':sha(checkpoint),'scene_visits':visits},indent=2))
    summary['used_for_optimization']=True;summary.setdefault('optimizer_reports',[]).append(str(root/'report.json'));summarypath.write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
