import numpy as np
import torch
import torch.nn.functional as F
from tools.audit_aces_receptive_context import dependency_bounds


def test_structural_bound_contains_independent_positive_graph_derivatives():
    def conv(x,stride=1):return F.avg_pool1d(x,3,stride=stride,padding=1)
    def blocks(x):
        for _ in range(2):x=x+conv(conv(x))
        return x
    for length in (128,129,256,321):
        x=torch.ones(1,1,length,dtype=torch.float64,requires_grad=True)
        e1=blocks(conv(x));e2=blocks(conv(e1,2));m=blocks(conv(e2,2))
        d2=blocks(conv(F.interpolate(m,size=e2.shape[-1],mode='linear',align_corners=False)+e2))
        d1=blocks(conv(F.interpolate(d2,size=e1.shape[-1],mode='linear',align_corners=False)+e1));y=conv(d1)
        lo,hi=dependency_bounds(length)
        for index in (0,1,length//2,length-2,length-1):
            grad=torch.autograd.grad(y[0,0,index],x,retain_graph=True)[0][0,0].numpy();support=np.flatnonzero(grad)
            assert support[0]>=lo[index] and support[-1]<=hi[index]
            assert support[-1]-support[0]>min(length-1,30)
