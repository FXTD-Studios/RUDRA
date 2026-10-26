"""Research-only multi-scene projection and differentiable positive-Y detail."""
import numpy as np
import torch
from scipy.optimize import nnls


def project_panel(direction, gradients):
    """Euclidean projection onto G d >= 0 via the nonnegative dual.

    min_lambda>=0 ||G.T lambda + d||^2 yields d_projected=d+G.T lambda.
    Normalizing constraint rows leaves their halfspaces unchanged.
    """
    d=np.asarray(direction,dtype=np.float64);g=np.asarray(gradients,dtype=np.float64)
    if d.ndim!=1 or g.ndim!=2 or g.shape[1]!=len(d):raise ValueError('shape')
    if not np.isfinite(d).all() or not np.isfinite(g).all():raise ValueError('nonfinite projection')
    norms=np.linalg.norm(g,axis=1);g=g[norms>0]/norms[norms>0,None]
    if not len(g):return d.copy(),{'constraints':0,'active_dual':0,'min_normalized_dot':None,'norm_ratio':1.}
    dual,_=nnls(g.T,-d,maxiter=max(300,30*len(g)))
    out=d+g.T@dual;dots=g@out;scale=max(np.linalg.norm(d),1e-15)
    if dots.min() < -1e-8*scale or np.max(np.abs(dual*dots))>1e-8*scale*scale:
        raise ValueError('projection feasibility/KKT failed')
    return out,{'constraints':len(g),'active_dual':int((dual>1e-12*scale).sum()),'min_normalized_dot':float(dots.min()),'norm_ratio':float(np.linalg.norm(out)/scale)}


def log_detail_loss(predicted,target,mask):
    """Mean per-image log2 AP1 luminance gradient RMSE, valid positive edges only.

    Matches the existing diagnostic, including its 1e-10 validity threshold.
    No model radiance is clipped. Zero-residual images use a zero subgradient.
    """
    luma=predicted.new_tensor([0.2722287168,0.6740817658,0.0536895174])[None,:,None,None]
    cy=(predicted*luma).sum(1,keepdim=True);ry=(target*luma).sum(1,keepdim=True)
    valid=mask&(cy>1e-10)&(ry>1e-10)
    c=torch.log2(torch.where(valid,cy,torch.ones_like(cy)));r=torch.log2(torch.where(valid,ry,torch.ones_like(ry)))
    residual=c-r;total=predicted.new_zeros(predicted.shape[0]);count=torch.zeros_like(total)
    for axis in (2,3):
        left=[slice(None)]*4;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
        edges=valid[tuple(left)]&valid[tuple(right)]
        total+=(torch.diff(residual,dim=axis).square()*edges).flatten(1).sum(1);count+=edges.flatten(1).sum(1)
    eligible=count>0
    if not eligible.any():return None
    mse=total[eligible]/count[eligible]
    positive=mse>0
    rmse=torch.where(positive,torch.sqrt(torch.where(positive,mse,torch.ones_like(mse))),mse*0)
    return rmse.mean()
