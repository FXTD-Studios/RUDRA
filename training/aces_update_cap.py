"""Cap an applied parameter update without touching optimizer moments."""
import math
import torch


def cap_update(params,before,limit):
    if not math.isfinite(limit) or limit<=0:raise ValueError('positive finite update cap required')
    after=torch.cat([p.detach().flatten() for p in params])
    delta=after.double()-before.double();norm=float(delta.norm())
    if not math.isfinite(norm):raise ValueError('nonfinite update')
    if norm<=limit:return {'proposed_l2':norm,'applied_l2':norm,'scale':1.,'rounding_adjustments':0}
    scale=limit/norm;adjustments=0
    while True:
        candidate=(before.double()+scale*delta).to(before.dtype)
        actual=float((candidate.double()-before.double()).norm())
        if actual<=limit:break
        scale*=.99;adjustments+=1
        if adjustments>16:raise ValueError('unable to represent bounded update')
    offset=0
    with torch.no_grad():
        for p in params:
            p.copy_(candidate[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
    assert offset==len(candidate)
    return {'proposed_l2':norm,'applied_l2':actual,'scale':scale,'rounding_adjustments':adjustments}
