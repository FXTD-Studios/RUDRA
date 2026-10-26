"""Measure counterfactual parameter steps, restoring the actual update exactly."""
import torch


def measure_step(params, before, projected, evaluate, small_length=1e-4):
    actual=[p.detach().clone() for p in params]
    after=torch.cat([p.flatten() for p in actual]);delta=after-before
    length=float(delta.double().norm())
    direction=-projected.double()/projected.double().norm().clamp_min(1e-30)
    adam_direction=delta.double()/delta.double().norm().clamp_min(1e-30)
    def assign(flat):
        offset=0
        for p in params:
            p.copy_(flat[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
        if offset!=flat.numel():raise ValueError('parameter length mismatch')
    devices=sorted({p.device.index for p in params if p.is_cuda})
    values={}
    try:
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            values['actual_adam']=evaluate()
            assign(before);baseline=evaluate()
            for name,d,scale in (('adam_small',adam_direction,small_length),
                                 ('projected_matched',direction,length),
                                 ('projected_small',direction,small_length)):
                assign((before.double()+scale*d).to(before.dtype));values[name]=evaluate()
    finally:
        with torch.no_grad():
            for p,original in zip(params,actual):p.copy_(original)
        assert all(torch.equal(p,original) for p,original in zip(params,actual))
    return {'baseline':baseline,'values':values,'deltas':{name:{k:v-baseline[k] for k,v in terms.items()} for name,terms in values.items()},
            'actual_step_l2':length,'small_step_l2':small_length,'actual_weights_restored':True}
