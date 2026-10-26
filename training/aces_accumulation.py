"""Mean-gradient accumulation for a registered fixed-size sample window."""
import torch


def accumulate_step(loss,optimizer,parameters,sample_step,window_size):
    if window_size<1 or sample_step<1:raise ValueError('positive sample step/window required')
    if (sample_step-1)%window_size==0:optimizer.zero_grad(set_to_none=True)
    (loss/window_size).backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise ValueError('nonfinite accumulated gradient')
    apply=sample_step%window_size==0
    if apply:optimizer.step()
    return apply
