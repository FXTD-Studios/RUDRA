"""Conservative structural context bound and exact-white TRAIN ambiguity audit."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import minimum_filter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint,PixelRMSNorm
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def dependency_bounds(length):
    """Conservative1D union of this branch's conv/skip/bilinear dependencies.

    Bilinear neighbours include zero-weight endpoints, so bounds can overestimate.
    Padding is constant and adds no input dependencies.
    """
    def conv(bounds,stride=1):
        lo,hi=bounds;n=len(lo);centres=np.arange((n+1)//2 if stride==2 else n)*stride
        indices=np.clip(centres[:,None]+np.arange(-1,2)[None],0,n-1)
        return lo[indices].min(1),hi[indices].max(1)
    def residuals(bounds):
        for _ in range(4):bounds=conv(bounds)
        return bounds
    def up(bounds,n):
        lo,hi=bounds;position=(np.arange(n)+.5)*len(lo)/n-.5
        indices=np.clip(np.stack([np.floor(position).astype(int),np.floor(position).astype(int)+1],1),0,len(lo)-1)
        return lo[indices].min(1),hi[indices].max(1)
    def union(a,b):return np.minimum(a[0],b[0]),np.maximum(a[1],b[1])
    source=(np.arange(length),np.arange(length));e1=residuals(conv(source));e2=residuals(conv(e1,2));mid=residuals(conv(e2,2))
    d2=residuals(conv(union(up(mid,len(e2[0])),e2)));d1=residuals(conv(union(up(d2,len(e1[0])),e1)))
    return conv(d1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();reg=json.loads(a.registration.read_text())
    for path,digest in reg['input_hashes'].items():
        if sha(path)!=digest:raise ValueError('input identity')
    a.out.mkdir(exist_ok=False);(a.out/'registration.json').write_bytes(a.registration.read_bytes());(a.out/'auditor.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2);view=reference_view();view.set_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model_from_checkpoint(torch.load(reg['checkpoint'],map_location='cpu',weights_only=True)).eval().to('cuda');branch=model.full_context
    convs={name:m for name,m in branch.named_modules() if isinstance(m,torch.nn.Conv2d)}
    if len(convs)!=26 or any(m.kernel_size!=(3,3) or m.padding!=(1,1) or m.dilation!=(1,1) for m in convs.values()):raise ValueError('unrecognized conv graph')
    if {name for name,m in convs.items() if m.stride==(2,2)}!={'down1','down2'}:raise ValueError('stride graph')
    if any(isinstance(m,(torch.nn.GroupNorm,torch.nn.BatchNorm2d,torch.nn.InstanceNorm2d)) for m in branch.modules()):raise ValueError('spatial normalization invalidates locality')
    if sum(isinstance(m,PixelRMSNorm) for m in branch.modules())!=20:raise ValueError('normalization graph')
    bounds={};rows=[]
    def radius(n):
        if n not in bounds:
            lo,hi=dependency_bounds(n);bounds[n]=int(max((np.arange(n)-lo).max(),(hi-np.arange(n)).max()))
        return bounds[n]
    for domain,path in reg['pilots'].items():
        for row in json.loads(Path(path).read_text())['rows']:
            if row['split']!='train':raise ValueError('TRAIN only')
            for key in ('sdr','target'):
                if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('content identity')
            source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path']);h,w=source.shape[:2];r=max(radius(h),radius(w))
            exact=(source==1).all(-1);full=source.min(-1)>=254.5/255
            blind=minimum_filter(exact,size=2*r+1,mode='constant',cval=0)
            values=np.arcsinh(target[blind]);count=int(blind.sum())
            rows.append({'domain':domain,'asset_id':row['asset_id'],'scene':row['scene_id'],'shape':[h,w],'radius_bound':r,'full_pixels':int(full.sum()),'exact_white_pixels':int(exact.sum()),'blind_pixels':count,
                         'target_asinh_rgb_std':values.std(0).tolist() if count else None,'target_asinh_rgb_range':np.ptp(values,axis=0).tolist() if count else None})
        print('audited',domain,len(rows),flush=True)
    probes=[];rng=np.random.default_rng(reg['seed'])
    for h,w in reg['probe_shapes']:
        y,x=h//2,w//2;ry,rx=radius(h),radius(w);a0=np.ones((h,w,3),np.float32);a1=rng.uniform(.05,.95,(h,w,3)).astype(np.float32)
        a1[y-ry:y+ry+1,x-rx:x+rx+1]=1;a2=a1.copy();a2[y,x]=.5
        values=[]
        for source in (a0,a1,a2):
            tensor=torch.from_numpy(source).permute(2,0,1)[None].to('cuda');base=torch.from_numpy(view.inverse(source)).permute(2,0,1)[None].to('cuda')
            with torch.inference_mode():out=branch(tensor,base)[0,:,y,x].cpu().numpy()
            values.append(out)
        error=float(np.max(np.abs(values[0]-values[1])));inside=float(np.max(np.abs(values[1]-values[2])))
        if error>reg['outside_tolerance']:raise ValueError('outside influence exceeds bound')
        probes.append({'shape':[h,w],'radius_y':ry,'radius_x':rx,'outside_change_max_field_error':error,'centre_change_max_field_error':inside})
    summary={}
    for domain in reg['pilots']:
        rr=[r for r in rows if r['domain']==domain];summary[domain]={'images':len(rr),'images_with_blind_pixels':sum(r['blind_pixels']>0 for r in rr),'full_pixels':sum(r['full_pixels'] for r in rr),'exact_white_pixels':sum(r['exact_white_pixels'] for r in rr),'blind_pixels':sum(r['blind_pixels'] for r in rr),'max_target_asinh_rgb_range':max((max(r['target_asinh_rgb_range']) for r in rr if r['blind_pixels']),default=None)}
    result={'summary':summary,'rows':rows,'probes':probes,'radius_by_dimension':bounds,'registration_sha256':sha(a.registration),'tool_sha256':sha(__file__),'scope':reg['scope'],'release_authorized':False}
    (a.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'summary':summary,'probes':probes},indent=2))


if __name__=='__main__':main()
