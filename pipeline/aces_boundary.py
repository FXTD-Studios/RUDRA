"""Experimental gradient-domain AP1 target join; not a production inference filter.

Original implementation of Dirichlet gradient-domain reconstruction (Perez et
al., SIGGRAPH 2003), in signed asinh AP1 coordinates. No external code/weights.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg


def match_boundary(base, reference, mask):
    """Keep reference gradients inside mask, base values outside exactly.

    Image borders use only existing neighbours. An entirely missing image has
    no Dirichlet anchor and is rejected rather than inventing a calibration.
    """
    base=np.asarray(base);reference=np.asarray(reference);mask=np.asarray(mask,bool)
    if base.shape!=reference.shape or base.shape!=(*mask.shape,3):
        raise ValueError('matching HWC RGB and HW mask required')
    if not np.isfinite(base).all() or not np.isfinite(reference).all():
        raise ValueError('finite inputs required')
    if not mask.any():return base.copy()
    if mask.all():raise ValueError('boundary anchor required')
    h,w=mask.shape;y,x=np.nonzero(mask);n=len(y)
    ids=np.full((h,w),-1,np.int64);ids[y,x]=np.arange(n)
    delta=np.arcsinh(base.astype(np.float64))-np.arcsinh(reference.astype(np.float64))
    rhs=np.zeros((n,3));degree=np.zeros(n);rows=[];cols=[]
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
        yy=y+dy;xx=x+dx;valid=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)
        origin=np.flatnonzero(valid);yy=yy[valid];xx=xx[valid]
        degree[origin]+=1
        neighbour=ids[yy,xx];inside=neighbour>=0
        rows.extend(origin[inside]);cols.extend(neighbour[inside])
        rhs[origin[~inside]]+=delta[yy[~inside],xx[~inside]]
    matrix=sparse.diags(degree,format='csr')+sparse.coo_matrix(
        (-np.ones(len(rows)),(rows,cols)),shape=(n,n)).tocsr()
    correction=np.empty((n,3))
    for c in range(3):
        solution,info=cg(matrix,rhs[:,c],rtol=1e-7,atol=1e-10,maxiter=4000)
        if info!=0:raise RuntimeError(f'boundary solver did not converge: {info}')
        correction[:,c]=solution
    output=base.copy()
    output[mask]=np.sinh(np.arcsinh(reference[mask].astype(np.float64))+correction)
    if not np.isfinite(output).all():raise ValueError('nonfinite boundary result')
    return output
