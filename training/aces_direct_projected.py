"""Direct normalized PCGrad updates with a saved TRAIN movement schedule."""
import json
import math
from pathlib import Path
import torch
from training.aces_update_cap import cap_update


def load_movement_schedule(path,root,recipe):
    path=Path(path);root=Path(root);report=json.loads(path.read_text())
    keys=('checkpoint','steps','batch','crop','border','seed','crops_per_frame','display_weight','display_evs','relative',
          'aligned_crops','matched_crops','paired_motion','temporal_weight','fixed_display_scale','deterministic','pcgrad')
    for key in keys:
        if report['recipe'].get(key)!=recipe.get(key):raise ValueError('movement recipe mismatch: '+key)
    for name in ('crop_pool.json','motion_pool.json','schedule.json'):
        if (path.parent/name).read_bytes()!=(root/name).read_bytes():raise ValueError('movement pool/schedule mismatch')
    lengths=[r['pcgrad']['update_cap']['applied_l2'] for r in report['history']]
    if len(lengths)!=recipe['steps'] or any(not math.isfinite(v) or v<=0 for v in lengths):
        raise ValueError('positive finite movement length required for every update')
    return lengths


def direct_projected_step(params,before,projected,length):
    if not math.isfinite(length) or length<=0:raise ValueError('positive finite step length required')
    norm=float(projected.double().norm())
    if not math.isfinite(norm):raise ValueError('nonfinite projected direction')
    if norm==0:return {'requested_l2':length,'applied_l2':0.,'zero_direction':True}
    candidate=(before.double()-length*projected.double()/norm).to(before.dtype)
    offset=0
    with torch.no_grad():
        for p in params:
            p.copy_(candidate[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
    info=cap_update(params,before,length)
    return {**info,'requested_l2':length,'zero_direction':False}
