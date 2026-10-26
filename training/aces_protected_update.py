"""Research-only transactional Adam update with measured multi-objective acceptance."""
import copy
import numpy as np
import torch
from tools.pilot_aces_protected_step import project_direction
from tools.pilot_aces_protected_trajectory import restore


def guarded_update(params,optimizer,losses,evaluate,mode,scales,rtol=1e-6,atol=1e-9):
    if mode not in ('adam_search','projected_search') or len(losses) not in (2,3):raise ValueError('update policy')
    before=np.array([float(v.detach()) for v in losses]);weights=torch.nn.utils.parameters_to_vector(params).detach().clone();saved=copy.deepcopy(optimizer.state_dict())
    grads=[]
    for i,loss in enumerate(losses):grads.append(torch.cat([g.flatten() for g in torch.autograd.grad(loss,params,retain_graph=i<len(losses)-1)]))
    if any(not torch.isfinite(g).all() for g in grads):raise ValueError('nonfinite component gradient')
    total=sum(grads);offset=0
    for p in params:p.grad=total[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
    optimizer.step();proposal=torch.nn.utils.parameters_to_vector(params).detach().clone()
    direction=(weights-proposal).double().cpu().numpy()
    if mode=='projected_search':direction=project_direction(direction,torch.stack(grads).double().cpu().numpy())
    tolerance=atol+rtol*np.abs(before);strict_index=2 if len(losses)==3 else 0;trials=[];accepted=False
    for scale in scales:
        if mode=='adam_search':candidate=proposal if scale==1 else weights+(proposal-weights)*scale
        else:candidate=weights-torch.from_numpy(direction*scale).to(weights)
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(candidate.clone(),params)
            after=np.array([float(v) for v in evaluate()])
        if not np.isfinite(after).all():raise ValueError('nonfinite trial loss')
        accepted=bool(np.all(after<=before+tolerance) and after[strict_index]<before[strict_index]-tolerance[strict_index])
        trials.append({'scale':scale,'after':after.tolist(),'accepted':accepted})
        if accepted:break
    if not accepted:restore(params,optimizer,weights,saved)
    return {'before':before.tolist(),'accepted':accepted,'trials':trials,'rollback_verified':not accepted,'component_gradient_norms':[float(g.norm()) for g in grads]}
