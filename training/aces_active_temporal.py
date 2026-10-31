"""Opt-in known half/quarter-pixel motion loss for an isolated full branch."""
import torch


def active_temporal_loss(prediction,target,high,full,border=4):
    if prediction.shape[0]!=2 or high.shape!=full.shape:
        raise ValueError('two-frame pair with matching support required')
    m=high[1:2];f=full[1:2]
    neighbours=lambda x:(x[:,:,:-1,:-1],x[:,:,:-1,1:],x[:,:,1:,:-1],x[:,:,1:,1:])
    valid=high[:1,:,:-1,:-1].clone()
    for n in neighbours(m):valid &= n
    active=full[:1,:,:-1,:-1].clone()
    for n in neighbours(f):active |= n
    if border:
        valid[:,:,:border]=False;valid[:,:,-border:]=False
        valid[:,:,:,:border]=False;valid[:,:,:,-border:]=False
    active &= valid
    residual=torch.asinh(prediction)-torch.asinh(target)
    a,b,c,d=neighbours(residual[1:2]);aligned=.375*a+.375*b+.125*c+.125*d
    error=(residual[:1,:,:-1,:-1]-aligned).square().mean(1,keepdim=True)
    return (error*active).sum()/active.sum().clamp_min(1),int(active.sum()),int(valid.sum())
