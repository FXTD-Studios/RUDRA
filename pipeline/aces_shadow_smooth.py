"""Experimental constrained log-exposure field for HDR-guided shadow targets."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize


def bound_smooth_shadows(reference,base,source,mask,view):
    """Minimize log-luminance gradient changes under verified viewed RGB caps.

    Outside-mask exposure is anchored to the baseline/reference luminance ratio.
    Chroma remains from reference. No claim of an exact RGB boundary match.
    """
    luma=np.array([.2722287168,.6740817658,.0536895174])
    luminance=reference@luma
    if np.any(luminance[mask]<0):raise ValueError('negative shadow reference luminance')
    zero=mask&(luminance==0)
    if np.any(reference[zero]!=0):raise ValueError('zero luminance with nonzero RGB')
    # Exact black has no positive radiance to scale and no log gradient.
    # Keep it exactly black; solve only positive reference pixels.
    mask=mask&~zero
    if not mask.any():return reference.copy(),{'pixels':0,'iterations':0,'exact_black_pixels':int(zero.sum())}
    if mask.all():raise ValueError('boundary anchor required')
    y,x=np.nonzero(mask);n=len(y);h,w=mask.shape
    ids=np.full((h,w),-1,np.int64);ids[mask]=np.arange(n)
    values=reference[mask];limit=source[mask]+.49/255
    lo=np.zeros(n);hi=np.ones(n)
    def safe(s):return (view.forward((values*s[:,None]).astype(np.float32))<=limit).all(-1)
    if not safe(lo).all():raise ValueError('zero exposure infeasible')
    lo[safe(hi)]=1
    for _ in range(22):
        mid=(lo+hi)/2;ok=safe(mid);lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid)
    if (lo<=0).any():raise ValueError('no positive feasible shadow scale')
    upper=np.log(lo)
    by=base@luma;ry=reference@luma
    anchor=np.log(np.maximum(by,1e-12)/np.maximum(ry,1e-12))
    degree=np.zeros(n);rhs=np.zeros(n);rows=[];cols=[]
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
        yy=y+dy;xx=x+dx;valid=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)
        origin=np.flatnonzero(valid);yy=yy[valid];xx=xx[valid]
        degree[origin]+=1;neighbour=ids[yy,xx];inside=neighbour>=0
        rows.extend(origin[inside]);cols.extend(neighbour[inside])
        rhs[origin[~inside]]+=anchor[yy[~inside],xx[~inside]]
    matrix=sparse.diags(degree,format='csr')+sparse.coo_matrix(
        (-np.ones(len(rows)),(rows,cols)),shape=(n,n)).tocsr()
    def objective(z):
        az=matrix@z
        return .5*np.dot(z,az)-np.dot(rhs,z),az-rhs
    result=minimize(objective,upper,method='L-BFGS-B',jac=True,
                    bounds=[(None,float(u)) for u in upper],
                    options={'maxiter':2000,'ftol':1e-12,'gtol':1e-7,'maxcor':20})
    if not result.success:raise RuntimeError(f'shadow optimization failed: {result.message}')
    output=reference.copy();output[mask]=values*np.exp(result.x[:,None])
    if ((view.forward(output[mask])-source[mask])*255).max()>.4901:
        raise ValueError('optimized shadow brightness failed exact verification')
    return output,{'pixels':n,'iterations':int(result.nit),'exact_black_pixels':int(zero.sum()),'min_scale':float(np.exp(result.x).min()),
                   'objective':float(result.fun)}
