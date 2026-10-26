"""Replay registered sampler decisions to measure useful highlight supervision."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import reconstruction_weight
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha
from training.aces_context_batches import sample_replay_crop


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_context_lr1e4_6000_20261024.json'))
    parser.add_argument('--out',type=Path,default=Path('G:/rudra_training/aces_context_lr1e4_6000_20261024/sampling_audit.json'))
    parser.add_argument('--training-report',type=Path)
    args=parser.parse_args();regpath=args.registration;reg=json.loads(regpath.read_text());out=args.out
    mode=reg.get('replay_sampling','legacy')
    if mode not in ('legacy','nonempty_phase_aligned_v2'):raise ValueError('unknown replay sampler')
    replay_v2=mode=='nonempty_phase_aligned_v2';excluded=[]
    if out.exists():raise ValueError('new output required')
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);masks=[];groups={};scenes={};points={}
    for domain,path in (('motion',reg['pilot']),('replay',reg['replay_pilot'])):
        for row in json.loads(Path(path).read_text())['rows']:
            if domain=='replay' and row['highlight_pixels']<=0:continue
            if row['split']!='train' or sha(row['sdr_path'])!=row['sdr_sha256']:raise ValueError('TRAIN identity')
            source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False))
            weight=reconstruction_weight(torch.from_numpy(source).permute(2,0,1)[None])[0,0].numpy()
            high=(weight>0)&(source.max(-1)>.5)
            if domain=='replay':
                eligible=high.copy()
                if replay_v2:eligible[:4]=False;eligible[-4:]=False;eligible[:,:4]=False;eligible[:,-4:]=False
                replay_points=np.argwhere(eligible)
                if replay_v2 and not len(replay_points):excluded.append(row['asset_id']);continue
            index=len(masks);masks.append(high)
            group=(domain,row.get('group_id',row['asset_id']));groups.setdefault(group,[]).append(index)
            scenes.setdefault(domain,{}).setdefault(row['scene_id'],set()).add(group)
            if domain=='replay':points[index]=replay_points
    if 'replay' not in scenes:raise ValueError('registered replay has no eligible supervision')
    domains=sorted(scenes);labels={d:sorted(scenes[d]) for d in domains};scenes={d:{s:sorted(g) for s,g in ss.items()} for d,ss in scenes.items()}
    rng=np.random.default_rng(reg['seed']);visits={d+':'+s:0 for d in domains for s in labels[d]};counts={d:[] for d in domains};empty={d:0 for d in domains};small={d:0 for d in domains}
    for step in range(reg['steps']):
        domain=domains[int(rng.integers(len(domains)))];scene=labels[domain][int(rng.integers(len(labels[domain])))];group=scenes[domain][scene][int(rng.integers(len(scenes[domain][scene])))];first=int(rng.integers(2)) if domain=='motion' else 0
        indices=groups[group][first:first+2];visits[domain+':'+scene]+=1;borders=[4,4,4,4];selected=[masks[i] for i in indices]
        if domain=='replay':
            index=indices[0];h,w=masks[index].shape
            if replay_v2:
                (y,x,ch,cw),borders=sample_replay_crop(h,w,points[index],rng)
            else:
                py,px=points[index][int(rng.integers(len(points[index])))];y=int(np.clip(py-rng.integers(64,256),0,h-320))//4*4;x=int(np.clip(px-rng.integers(64,448),0,w-512))//4*4
                ch,cw=320,512;borders=[64 if y else 4,64 if y+320<h else 4,64 if x else 4,64 if x+512<w else 4]
            selected=[masks[index][y:y+ch,x:x+cw]]
        top,bottom,left,right=borders;values=[int(m[top:m.shape[0]-bottom,left:m.shape[1]-right].sum()) for m in selected]
        counts[domain].extend(values);empty[domain]+=int(sum(values)==0);small[domain]+=int(0<sum(values)<128)
    if args.training_report and visits!=json.loads(args.training_report.read_text())['scene_visits']:raise ValueError('sampler trajectory mismatch')
    if replay_v2 and empty['replay']:raise ValueError('v2 produced empty replay updates')
    summary={d:{'frames':len(v),'zero_support_frames':v.count(0),'zero_support_updates':empty[d],'updates_with_1_to_127_total_pixels':small[d],'pixel_count_percentiles':dict(zip(('min','p10','median','p90','max'),np.percentile(v,[0,10,50,90,100]).tolist()))} for d,v in counts.items()}
    report={'summary':summary,'sampler_scene_visits_match':True if args.training_report else None,'replay_sampling':mode,'excluded_replay_without_interior_highlights':excluded,'scene_visits':visits,'registration_sha256':sha(regpath),'tool_sha256':sha(__file__),'batch_preparation_sha256':sha('training/aces_context_batches.py'),'scope':'Registered TRAIN sampler decisions and source-defined support, before any parameter updates. Counts diagnose supervision coverage, not model quality.','release_authorized':False}
    out.with_suffix('.auditor.py').write_bytes(Path(__file__).read_bytes())
    out.with_suffix('.batches.py').write_bytes(Path('training/aces_context_batches.py').read_bytes())
    out.write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
