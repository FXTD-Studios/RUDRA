"""Train only the v4 code restorer; preserve the frozen HDR predictor exactly."""
import argparse,json,os,sys,time,math
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import ACESModel,model_from_checkpoint,reconstruction_weight
from training.aces_reference import reference_view
from tools.train_aces_spatial import load_training_rows,sample_crop
from tools.evaluate_run05_dequant import sha


def merge_training_ancestry(base_rows,new_rows):
    combined={}
    for row in base_rows+new_rows:
        if row['split']!='train':raise ValueError('TRAIN ancestry required')
        old=combined.get(row['asset_id'])
        if old and any(old[key]!=row[key] for key in ('scene_id','sdr_sha256','target_sha256')):
            raise ValueError('conflicting ancestry identity')
        combined[row['asset_id']]=row
    return list(combined.values())


def ordinary_code_mse(source,restored,target,border=40):
    area=slice(border,-border) if border else slice(None)
    mask=(reconstruction_weight(source)==0)[:,:,area,area]
    error=((restored-target)*255).square().mean(1,keepdim=True)[:,:,area,area]
    return (error*mask).sum()/mask.sum().clamp_min(1),int(mask.sum())


def ordinary_gradient_mse(source,restored,target,border=40):
    area=slice(border,-border) if border else slice(None)
    valid=(reconstruction_weight(source)==0)[:,:,area,area]
    prediction=restored[:,:,area,area]*255;reference=target[:,:,area,area]*255
    total=prediction.sum()*0;count=valid.sum()*0
    for axis in (2,3):
        left=[slice(None)]*4;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
        edges=valid[tuple(left)]&valid[tuple(right)]
        error=(torch.diff(prediction,dim=axis)-torch.diff(reference,dim=axis)).square().mean(1,keepdim=True)
        total=total+(error*edges).sum();count=count+edges.sum()
    return total/count.clamp_min(1)


def training_domains(rows):
    groups={}
    for index,row in enumerate(rows):
        label=row.get('training_domain')
        if not isinstance(label,str) or not label:raise ValueError('explicit training_domain required')
        groups.setdefault(label,[]).append(index)
    if len(groups)!=2:raise ValueError('exactly two training domains required')
    return groups


def domain_crop(data,indices,rng,size):
    source,target,(index,x,y,mode)=sample_crop([data[i] for i in indices],rng,size)
    return source,target,(indices[index],x,y,mode)


def quota_domains(rows,quotas):
    groups={}
    for i,row in enumerate(rows):
        label=row.get('training_domain')
        if not isinstance(label,str) or not label:raise ValueError('explicit training_domain required')
        groups.setdefault(label,[]).append(i)
    counts={}
    for quota in quotas:
        label,value=quota.split('=',1);count=int(value)
        if label in counts or count<=0:raise ValueError('unique positive domain quotas required')
        counts[label]=count
    if set(counts)!=set(groups) or sum(counts.values())!=4:
        raise ValueError('quotas must cover every domain and total four')
    return groups,counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base-checkpoint','base-pilot','pilot','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--gradient-weight',type=float,default=0.)
    p.add_argument('--balanced-domains',action='store_true',help='exactly two crops per explicit training_domain per batch')
    p.add_argument('--domain-crops',nargs='+',help='explicit domain=count quotas totalling four crops')
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    if a.steps<=0:raise ValueError('positive steps required')
    if not math.isfinite(a.gradient_weight) or a.gradient_weight<0:raise ValueError('finite nonnegative gradient weight required')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2);torch.manual_seed(20261023);rng=np.random.default_rng(20261023)
    state=torch.load(a.base_checkpoint,map_location='cpu',weights_only=True)
    if state['pilot_sha256']!=sha(a.base_pilot):raise ValueError('base training identity mismatch')
    base=model_from_checkpoint(state)
    if base.restoration!='shared':raise ValueError('shared-restoration base required')
    pilot=json.loads(a.pilot.read_text());data=load_training_rows(pilot)
    if a.balanced_domains and a.domain_crops:raise ValueError('choose one domain sampling mode')
    domains=training_domains(pilot['rows']) if a.balanced_domains else None
    counts={k:2 for k in domains} if domains else None
    if a.domain_crops:domains,counts=quota_domains(pilot['rows'],a.domain_crops)
    ancestry=json.loads(a.base_pilot.read_text())['rows']
    if any(r['split']!='train' for r in ancestry):raise ValueError('base TRAIN rows required')
    combined=merge_training_ancestry(ancestry,pilot['rows'])
    codes=[reference_view().forward(target) for _,target,_,_ in data]
    model=ACESModel(state['recipe']['channels'],base.normalization,'local_codes')
    model.predictor.load_state_dict(base.predictor.state_dict(),strict=True)
    model.predictor.requires_grad_(False)
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    optimizer=torch.optim.Adam(model.restorer.parameters(),lr=.001)
    a.out.mkdir(parents=True);union=a.out/'training_ancestry.json'
    union.write_text(json.dumps({'rows':combined,'scope':'deduplicated union of frozen HDR and restoration training; optimizer sees supplied restoration pilot only'},indent=2))
    recipe={'steps':a.steps,'seed':20261023,'lr':.001,'batch':4,'crop':128,'loss_border':40,
            'gradient_weight':a.gradient_weight,
            'sampling':'explicit crops per domain per batch' if domains else 'uniform image',
            'domain_crops':counts,
            'domain_sizes':{k:len(v) for k,v in domains.items()} if domains else None,
            'channels':state['recipe']['channels'],'base_checkpoint_sha256':sha(a.base_checkpoint),
            'restoration_pilot_sha256':sha(a.pilot),'frozen_base_pilot_sha256':sha(a.base_pilot),
            'code_sha256':{name:sha(name) for name in ('tools/train_aces_restorer.py','training/aces_restorer.py','training/aces_model.py')},
            'scope':'restorer-only code MSE; frozen HDR weights; deterministic, fixed final step'}
    history=[];start=time.time();updates=0;visits=np.zeros(len(data),dtype=int)
    for step in range(1,a.steps+1):
        batch=([domain_crop(data,indices,rng,128) for label,indices in domains.items() for _ in range(counts[label])]
               if domains else [sample_crop(data,rng,128) for _ in range(4)])
        for _,_,location in batch:visits[location[0]]+=1
        source=torch.stack([b[0] for b in batch]).to(device)
        target=[]
        for _,_,(index,x,y,_) in batch:target.append(torch.from_numpy(codes[index][y:y+128,x:x+128].copy()).permute(2,0,1))
        target=torch.stack(target).to(device)
        raw=model.restorer(source);restored=(source+.49/255*torch.tanh(raw)).clamp(0,1)
        loss,pixels=ordinary_code_mse(source,restored,target)
        if a.gradient_weight:loss=loss+a.gradient_weight*ordinary_gradient_mse(source,restored,target)
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        if pixels:
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();updates+=1
        if step==1 or step%50==0 or step==a.steps:
            row={'step':step,'loss':float(loss.detach()),'seconds':time.time()-start}
            history.append(row);print(row,flush=True)
    frozen=all(torch.equal(value.detach().cpu(),base.predictor.state_dict()[name]) for name,value in model.predictor.state_dict().items())
    if not frozen:raise ValueError('frozen HDR predictor changed')
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite final weights')
    torch.save({'model':model.state_dict(),'contract':model.contract(),'step':a.steps,'recipe':recipe,
                'pilot_sha256':sha(union),'optimizer':optimizer.state_dict(),'research_only':True},a.out/f'step{a.steps}.pt')
    (a.out/'report.json').write_text(json.dumps({'recipe':recipe,'history':history,'visits':visits.tolist(),'optimizer_updates':updates,'frozen_predictor_bit_exact':frozen,
        'checkpoint_sha256':sha(a.out/f'step{a.steps}.pt'),'training_ancestry_sha256':sha(union)},indent=2))


if __name__=='__main__':main()
