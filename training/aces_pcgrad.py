"""Opt-in PCGrad, algorithm1 of Yu et al. arXiv:2001.06782v4.

Written locally from the mathematical rule; original reference gradients stay
fixed while each task's working gradient is sequentially projected. Sum, not
average, preserves the existing sum-of-weighted-objectives convention.
"""
import torch


def project_conflicting(gradients, rng):
    if gradients.ndim!=2 or not len(gradients) or not torch.isfinite(gradients).all():
        raise ValueError('finite task-by-parameter gradients required')
    original=gradients.detach().to(torch.float64)
    projected=original.clone();orders=[];count=0
    for i in range(len(original)):
        order=rng.permutation([j for j in range(len(original)) if j!=i]).tolist();orders.append(order)
        for j in order:
            dot=projected[i]@original[j]
            if dot<0:
                denom=original[j]@original[j]
                if denom>0:
                    projected[i]-=(dot/denom)*original[j];count+=1
    result=projected.sum(0).to(gradients.dtype)
    if not torch.isfinite(result).all():raise ValueError('nonfinite projected gradient')
    return result,{'projections':count,'orders':orders}
