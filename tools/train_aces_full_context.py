"""Train a full-only branch, then compare identical learned weights with shared and isolated routing."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_context_model import ACESContextModel,aligned_pair_residual_loss
from training.aces_full_context_model import ACESFullContextModel
from training.aces_detail_loss import highlight_gradient_loss
from training.aces_active_temporal import active_temporal_loss
from training.aces_protected_update import guarded_update
from training.aces_accumulation import accumulate_step
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
    temporal_support=reg.get('temporal_support','fully_clipped')
    if temporal_support not in ('fully_clipped','all_highlight_composed','active_union_composed'):raise ValueError('unknown temporal support')
    composed=temporal_support!='fully_clipped'
    update_mode=reg.get('update_mode','adam')
    accumulation_steps=reg.get('accumulation_steps',1)
    balanced_windows=reg.get('balanced_windows',False)
    if accumulation_steps not in (1,6) or reg['steps']%accumulation_steps:raise ValueError('accumulation budget')
    if accumulation_steps!=1 and (update_mode!='adam' or not balanced_windows):raise ValueError('balanced Adam accumulation required')
    if update_mode not in ('adam','adam_search','projected_search'):raise ValueError('update mode')
    if update_mode!='adam' and temporal_support!='active_union_composed':raise ValueError('guarded active-union required')
    if reg.get('replay_sampling','legacy') not in ('legacy','nonempty_phase_aligned_v2'):raise ValueError('unknown replay sampler')
    if reg.get('replay_sampling')!='nonempty_phase_aligned_v2' or reg.get('training_region')!='fully_clipped':raise ValueError('registered full-only v2 sampler required')
    pilotpath=Path(reg['pilot']);pilot=json.loads(pilotpath.read_text());summarypath=pilotpath.parent.parent/'summary.json'
    summary=json.loads(summarypath.read_text())
    if not summary['build_pass'] or any(r['split']!='train' for r in pilot['rows']):raise ValueError('valid TRAIN corpus required')
    if sha(reg['parent'])!=reg['parent_sha256']:raise ValueError('parent identity')
    replay=json.loads(Path(reg['replay_pilot']).read_text()) if reg.get('replay_pilot') else {'rows':[]}
    if any(r['split']!='train' for r in replay['rows']):raise ValueError('TRAIN-only replay')
    root=args.out;root.mkdir(exist_ok=False)
    (root/'full_model_snapshot.py').write_bytes(Path('training/aces_full_context_model.py').read_bytes())
    (root/'factory_snapshot.py').write_bytes(Path('training/aces_model.py').read_bytes())
    (root/'registration.json').write_bytes(regpath.read_bytes())
    (root/'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (root/'model_snapshot.py').write_bytes(Path('training/aces_context_model.py').read_bytes())
    (root/'batch_preparation_snapshot.py').write_bytes(Path('training/aces_context_batches.py').read_bytes())
    (root/'active_temporal_snapshot.py').write_bytes(Path('training/aces_active_temporal.py').read_bytes())
    for name in ('training/aces_accumulation.py','training/aces_protected_update.py','tools/pilot_aces_protected_step.py','tools/pilot_aces_protected_trajectory.py'):
        (root/Path(name).name).write_bytes(Path(name).read_bytes())
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2);torch.manual_seed(reg['seed']);rng=np.random.default_rng(reg['seed']);reference_view().set_threads(4)
    state=torch.load(reg['parent'],map_location='cpu',weights_only=True);parent=model_from_checkpoint(state).eval().to('cuda')
    if parent.contract()['id']!='aces2_ap1_reconstruction_v8_highlight_context_experimental':raise ValueError('v8 parent required')
    ancestry_path=Path(reg['parent']).parent/'training_ancestry.json'
    if state['pilot_sha256']!=sha(ancestry_path):raise ValueError('ancestry identity')
    ancestry=root/'training_ancestry.json';ancestry.write_text(json.dumps({'rows':merge_training_ancestry(json.loads(ancestry_path.read_text())['rows'],pilot['rows']+replay['rows'])},indent=2))
    model=(model_from_checkpoint(state) if isinstance(parent,ACESContextModel) else ACESContextModel.from_parent(parent)).to('cuda').train()
    if composed:model=ACESFullContextModel.from_parent(parent).to('cuda').train()
    cache=[];groups={};scenes={};points={};metadata={};excluded_replay=[];ineligible_motion=set()
    replay_v2=reg.get('replay_sampling')=='nonempty_phase_aligned_v2'
    tagged=[('motion',r) for r in pilot['rows']]+[('replay',r) for r in replay['rows'] if r['highlight_pixels']>0]
    for domain,row in tagged:
        if sha(row['sdr_path'])!=row['sdr_sha256'] or sha(row['target_path'])!=row['target_sha256']:raise ValueError('input identity')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
        with torch.inference_mode():
            value=parent(torch.from_numpy(source).permute(2,0,1)[None].to('cuda'))
        arrays=[source.transpose(2,0,1),value.baseline[0].cpu().numpy(),value.restored_acescg[0].cpu().numpy(),
                value.reconstruction_weight[0].cpu().numpy(),target.transpose(2,0,1)]
        eligible=(arrays[3][0]>0)&(source.min(-1)>=254.5/255)
        interior=eligible[4:-4,4:-4]
        if domain=='motion' and not interior.any():ineligible_motion.add((domain,row['group_id']))
        if domain=='replay':
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
    for group in ineligible_motion:
        groups.pop(group,None)
        for scene_groups in scenes.get('motion',{}).values():scene_groups.discard(group)
    scenes={d:{s:g for s,g in ss.items() if g} for d,ss in scenes.items()}
    if not scenes.get('motion') or not scenes.get('replay'):raise ValueError('both domains require eligible full supervision')
    if replay['rows'] and 'replay' not in scenes:raise ValueError('registered replay has no eligible supervision')
    for (domain,_),indices in groups.items():
        if domain=='motion' and ([metadata[i]['frame'] for i in indices]!=[0,1,2] or [metadata[i]['translation_xy'] for i in indices]!=[[0,0],[.5,.25],[1,.5]]):raise ValueError('known-motion metadata mismatch')
        if domain=='replay' and len(indices)!=1:raise ValueError('single-frame replay required')
    del parent;domains=sorted(scenes);labels={d:sorted(scenes[d]) for d in domains};scenes={d:{s:sorted(g) for s,g in ss.items()} for d,ss in scenes.items()}
    schedule=[]
    if balanced_windows:
        if update_mode!='adam' or reg['steps']%6:raise ValueError('balanced six-sample windows require Adam')
        for window in range(reg['steps']//6):
            selected={d:rng.choice(labels[d],size=3,replace=False).tolist() for d in domains}
            for slot in range(3):
                for domain in ('motion','replay'):
                    scene=selected[domain][slot];available=scenes[domain][scene]
                    if domain=='motion':
                        fraction=(.001,.01,.05)[slot]
                        available=[g for g in available if metadata[groups[g][0]]['desired_fully_clipped_fraction']==fraction]
                        if not available:raise ValueError('missing registered clipping stratum')
                    group=available[int(rng.integers(len(available)))];first=int(rng.integers(2)) if domain=='motion' else 0
                    indices=groups[group][first:first+2];crop=None;borders=[4]*4
                    if domain=='replay':
                        index=indices[0];h,w=cache[index][0].shape[-2:];crop,borders=sample_replay_crop(h,w,points[index],rng)
                    schedule.append({'window':window,'domain':domain,'scene':scene,'group':group[1],'first':first,'indices':indices,'crop':crop,'borders':borders,'assets':[metadata[i]['asset_id'] for i in indices]})
        (root/'sample_schedule.json').write_text(json.dumps(schedule,indent=2))
    branch=model.full_context if composed else model.highlight_context
    optimizer=torch.optim.Adam(branch.parameters(),lr=reg['learning_rate']);history=[];visits={d+':'+s:0 for d in domains for s in labels[d]};start=time.time()
    gradient_balance=[]
    update_rows=[]
    for step in range(1,reg['steps']+1):
        if balanced_windows:
            item=schedule[step-1];domain=item['domain'];scene=item['scene'];group=(domain,item['group']);first=item['first']
        else:
            domain=domains[int(rng.integers(len(domains)))];scene=labels[domain][int(rng.integers(len(labels[domain])))];group=scenes[domain][scene][int(rng.integers(len(scenes[domain][scene])))];first=int(rng.integers(2)) if domain=='motion' else 0
        indices=groups[group][first:first+2];visits[domain+':'+scene]+=1;borders=[4,4,4,4]
        crop=None
        if balanced_windows:
            crop=item['crop'];borders=item['borders']
        elif domain=='replay':
            index=indices[0];h,w=cache[index][0].shape[-2:]
            if h<320 or w<512 or not len(points[index]):raise ValueError('replay crop/support')
            if replay_v2:crop,borders=sample_replay_crop(h,w,points[index],rng)
            else:
                py,px=points[index][int(rng.integers(len(points[index])))];y=int(np.clip(py-rng.integers(64,256),0,h-320))//4*4;x=int(np.clip(px-rng.integers(64,448),0,w-512))//4*4
                crop=(y,x,320,512);borders=[64 if y else 4,64 if y+320<h else 4,64 if x else 4,64 if x+512<w else 4]
        arrays=stack_context_arrays(cache,indices,crop)
        source,base,restored,weight,target=[torch.from_numpy(np.ascontiguousarray(v)).to('cuda') for v in arrays]
        high=(weight>0)&(source.amin(1,keepdim=True)>=254.5/255);valid=high.clone()
        top,bottom,left,right=borders;valid[:,:,:top]=False;valid[:,:,-bottom:]=False;valid[:,:,:,:left]=False;valid[:,:,:,-right:]=False
        if not valid.flatten(1).any(1).all():raise ValueError('empty full-region supervision')
        prediction,_=model.predict_highlights(source,base,restored,weight)
        error=(torch.asinh(prediction)-torch.asinh(target)).square().mean(1,keepdim=True)
        photo=((error*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)).mean()
        gradient=highlight_gradient_loss(prediction,target,valid)
        temporal_mask=(weight>0)&(source.amax(1,keepdim=True)>.5) if composed else high
        temporal=aligned_pair_residual_loss(prediction,target,temporal_mask) if domain=='motion' else photo*0
        active_count=valid_count=None
        if domain=='motion' and temporal_support=='active_union_composed':
            temporal,active_count,valid_count=active_temporal_loss(prediction,target,temporal_mask,high)
            if not active_count:raise ValueError('empty active temporal supervision')
        loss=photo+reg['loss']['highlight_gradient']*gradient+reg['loss']['known_aligned_temporal_residual']*temporal
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        if domain=='motion' and reg.get('log_gradient_balance',False):
            norms={}
            for name,term in [('photo',photo),('detail',reg['loss']['highlight_gradient']*gradient),('temporal',reg['loss']['known_aligned_temporal_residual']*temporal)]:
                grads=torch.autograd.grad(term,list(branch.parameters()),retain_graph=True)
                norm=torch.stack([g.square().sum() for g in grads]).sum().sqrt()
                if not torch.isfinite(norm):raise ValueError('nonfinite component gradient')
                norms[name]=float(norm)
            gradient_balance.append({'step':step,'group':group[1],'first':first,'active_count':active_count,'valid_count':valid_count,'weighted_parameter_gradient_norms':norms})
        if update_mode=='adam':
            accumulate_step(loss,optimizer,list(branch.parameters()),step,accumulation_steps)
        else:
            def terms():
                predicted,_=model.predict_highlights(source,base,restored,weight)
                ee=(torch.asinh(predicted)-torch.asinh(target)).square().mean(1,keepdim=True)
                pp=((ee*valid).flatten(1).sum(1)/valid.flatten(1).sum(1)).mean()
                dd=reg['loss']['highlight_gradient']*highlight_gradient_loss(predicted,target,valid)
                values=[pp,dd]
                if domain=='motion':values.append(reg['loss']['known_aligned_temporal_residual']*active_temporal_loss(predicted,target,temporal_mask,high)[0])
                return values
            initial=[photo,reg['loss']['highlight_gradient']*gradient]
            if domain=='motion':initial.append(reg['loss']['known_aligned_temporal_residual']*temporal)
            update=guarded_update(list(branch.parameters()),optimizer,initial,terms,update_mode,reg['search_scales'],reg['relative_tolerance'],reg['absolute_tolerance'])
            update.update(step=step,domain=domain,scene=scene,group=group[1],first=first,desired_fraction=metadata[indices[0]].get('desired_fully_clipped_fraction'))
            update_rows.append(update)
        if step==1 or step%100==0:
            row={'step':step,'domain':domain,'loss':float(loss.detach()),'photo':float(photo.detach()),'gradient':float(gradient.detach()),'temporal':float(temporal.detach()),'seconds':time.time()-start}
            history.append(row);print(row,flush=True)
    final={k:v.detach().cpu() for k,v in model.state_dict().items()}
    frozen=all(torch.equal(final[k],v) for k,v in state['model'].items() if composed or not k.startswith('highlight_context.'))
    if not frozen or any(not torch.isfinite(v).all() for v in final.values()):raise ValueError('frozen/finite invariant')
    recipe={'registration_sha256':sha(regpath),'pilot_sha256':sha(pilotpath),'parent_sha256':sha(reg['parent']),
            'replay_pilot_sha256':sha(reg['replay_pilot']) if replay['rows'] else None,
            'trainer_sha256':sha(__file__),'batch_preparation_sha256':sha('training/aces_context_batches.py'),'replay_sampling':reg.get('replay_sampling','legacy'),'excluded_replay_without_interior_highlights':excluded_replay,'model_code_sha256':sha('training/aces_context_model.py'),'steps':reg['steps'],'seed':reg['seed']}
    recipe.update(training_region='fully_clipped',temporal_support=temporal_support,excluded_motion_groups=sorted(g[1] for g in ineligible_motion),full_model_sha256=sha('training/aces_full_context_model.py'),active_temporal_sha256=sha('training/aces_active_temporal.py'))
    (root/'gradient_balance.json').write_text(json.dumps(gradient_balance,indent=2,allow_nan=False))
    (root/'update_acceptance.json').write_text(json.dumps(update_rows,indent=2,allow_nan=False))
    recipe.update(update_mode=update_mode,protected_update_sha256=sha('training/aces_protected_update.py'),accepted_updates=sum(r['accepted'] for r in update_rows) if update_rows else reg['steps'])
    recipe.update(accumulation_steps=accumulation_steps,balanced_windows=balanced_windows,sample_exposures=reg['steps'],optimizer_updates=reg['steps']//accumulation_steps if update_mode=='adam' else recipe['accepted_updates'],sample_schedule_sha256=sha(root/'sample_schedule.json') if balanced_windows else None,accumulation_helper_sha256=sha('training/aces_accumulation.py'))
    if update_mode=='adam':
        recipe['accepted_updates']=recipe['optimizer_updates']
        if any(int(v['step'])!=recipe['optimizer_updates'] for v in optimizer.state.values()):raise ValueError('accumulation optimizer counter')
    if update_rows and any(int(v['step'])!=recipe['accepted_updates'] for v in optimizer.state.values()):raise ValueError('optimizer counter mismatch')
    shared=root/'shared_control';shared.mkdir();(shared/'training_ancestry.json').write_bytes(ancestry.read_bytes())
    shared_ckpt=shared/f'step{reg["steps"]}.pt'
    learned={k:v.detach().cpu() for k,v in branch.state_dict().items()}
    shared_model=model_from_checkpoint(state).eval();shared_model.highlight_context.load_state_dict(learned)
    torch.save({'model':shared_model.state_dict(),'contract':shared_model.contract(),'pilot_sha256':sha(ancestry),'recipe':recipe,'research_only':True},shared_ckpt)
    (shared/'report.json').write_text(json.dumps({'recipe':recipe,'frozen_parent_bit_exact':frozen,'checkpoint_sha256':sha(shared_ckpt),'scene_visits':visits},indent=2))
    original=model_from_checkpoint(state).eval()
    isolated=ACESFullContextModel.from_parent(original).eval()
    isolated.full_context.load_state_dict(learned)
    isolated_state=isolated.state_dict()
    isolated_frozen=all(torch.equal(isolated_state[k],v) for k,v in state['model'].items())
    if not isolated_frozen:raise ValueError('isolated parent modified')
    checkpoint=root/f'step{reg["steps"]}.pt'
    torch.save({'model':isolated_state,'contract':isolated.contract(),'pilot_sha256':sha(ancestry),'recipe':recipe,'research_only':True},checkpoint)
    (root/'report.json').write_text(json.dumps({'recipe':recipe,'history':history,'frozen_parent_bit_exact':isolated_frozen,'checkpoint_sha256':sha(checkpoint),'scene_visits':visits,'shared_control_sha256':sha(shared_ckpt),'comparison':'One full-only optimization trajectory, identical learned weights under shared versus isolated routing. Not two independent optimization runs.'},indent=2))
    summary['used_for_optimization']=True;summary.setdefault('optimizer_reports',[]).append(str(root/'report.json'));summarypath.write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
