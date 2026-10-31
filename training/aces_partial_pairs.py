"""Validated adjacent TRAIN crops for partial-highlight temporal learning."""
import json
from pathlib import Path
import numpy as np
import torch
from training.aces_model import reconstruction_weight
from training.aces_partial_geometry import aligned_partial_origin
from training.aces_active_temporal import active_temporal_loss
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def validate_pair(a,b):
    if any(r['split']!='train' for r in (a,b)):raise ValueError('TRAIN pairs only')
    if a['group_id']!=b['group_id'] or a['scene_id']!=b['scene_id'] or b['frame']!=a['frame']+1:
        raise ValueError('adjacent same-group frames required')
    if not np.allclose(np.subtract(b['translation_xy'],a['translation_xy']),[.5,.25],rtol=0,atol=1e-12):
        raise ValueError('known half/quarter-pixel translation required')


def build_motion_pool(pilot,loc,sha,crop,border,seed,crops_per_pair):
    rows=json.loads(Path(pilot).read_text())['rows'];groups={};cache={}
    for row in rows:
        if row['split']!='train':raise ValueError('TRAIN only')
        for key in ('sdr','target'):
            if sha(loc(row[key+'_path']))!=row[key+'_sha256']:raise ValueError('input identity')
        source=canonical_to_source(load_rgb(loc(row['sdr_path']),hdr=False));target=np.load(loc(row['target_path'])).astype(np.float32)
        x=torch.from_numpy(source).permute(2,0,1)[None];w=reconstruction_weight(x)[0,0].numpy()
        high=(w>0)&(source.max(-1)>.5);partial=high&(source.min(-1)<254.5/255)
        cache[row['asset_id']]=(source,target,high,partial)
        groups.setdefault(row['group_id'],[]).append(row)
    pool=[];metadata=[];rng=np.random.default_rng(seed)
    for group,items in groups.items():
        items=sorted(items,key=lambda r:r['frame'])
        for a,b in zip(items,items[1:]):
            validate_pair(a,b);aa,bb=cache[a['asset_id']],cache[b['asset_id']]
            if aa[0].shape!=bb[0].shape:raise ValueError('pair geometry')
            h,w=aa[0].shape[:2];points=np.argwhere(aa[3][border:h-border,border:w-border])
            if not len(points):continue
            for index in rng.choice(len(points),min(crops_per_pair,len(points)),replace=False):
                cy,cx=points[index]+border;y,x=aligned_partial_origin(cy,cx,h,w,crop,border)
                cuts=[np.stack([v[j][y:y+crop,x:x+crop] for v in (aa,bb)]) for j in range(4)]
                sources,targets,high,partial=cuts
                if not partial[:,border:-border,border:-border].reshape(2,-1).any(1).all():continue
                dummy=torch.zeros((2,3,crop,crop))
                _,active,_=active_temporal_loss(dummy,dummy,torch.from_numpy(high[:,None]),torch.from_numpy(partial[:,None]),border=border)
                if not active:continue
                pool.append((sources,targets));metadata.append({'assets':[a['asset_id'],b['asset_id']],'scene':a['scene_id'],'group':group,'frames':[a['frame'],b['frame']],'origin':[y,x],'active_pixels':active})
    return pool,metadata
