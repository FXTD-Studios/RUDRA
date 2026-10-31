"""Matched local/wider-context shadow trial with equal domain loss weights."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import ACESModel,model_from_checkpoint,reconstruction_weight
from training.aces_shadow import compose_shadow,luminance
from training.aces_losses import shadow_reference_scale
from training.aces_reference import reference_view
from tools.train_aces_spatial import load_training_rows
from tools.train_aces_restorer import merge_training_ancestry,training_domains
from tools.evaluate_run05_dequant import sha


def equal_domain_loss(error,weight):
    """First two batch members and last two form two equally weighted domains."""
    losses=[]
    for area in (slice(0,2),slice(2,4)):
        w=weight[area];losses.append((error[area]*w).sum()/w.sum().clamp_min(1))
    return sum(losses)/2


def initialize_shadow_pair(local,wide):
    state=local.state_dict();new=wide.state_dict()
    for key,value in state.items():
        if key=='layers.0.weight':
            new[key].zero_();new[key][:,:6].copy_(value)
        else:new[key].copy_(value)
    wide.load_state_dict(new)


def valid_crop_context(x,y,width,height,size=192,border=40):
    """Exclude only artificial crop edges; true image edges match full inference."""
    if x<0 or y<0 or x+size>width or y+size>height:raise ValueError('crop outside image')
    valid=torch.zeros(1,size,size)
    left=0 if x==0 else border;right=size if x+size==width else size-border
    top=0 if y==0 else border;bottom=size if y+size==height else size-border
    valid[:,top:bottom,left:right]=1
    return valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','base-pilot','pilot','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--wide',action='store_true');p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--include-image-edges',action='store_true');p.add_argument('--seed',type=int,default=20261023);a=p.parse_args()
    if a.out.exists() or a.steps<=0:raise ValueError('new output and positive steps required')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if state['pilot_sha256']!=sha(a.base_pilot):raise ValueError('base ancestry mismatch')
    base=model_from_checkpoint(state)
    if base.shadow!='shared' or base.restoration!='local_codes':raise ValueError('v4 base required')
    pilot=json.loads(a.pilot.read_text());data=load_training_rows(pilot);groups=training_domains(pilot['rows'])
    groups={name:[i for i in indices if len(data[i][3])] for name,indices in groups.items()}
    if any(not indices for indices in groups.values()):raise ValueError('shadow support required in both domains')
    baselines=[reference_view().inverse(src) for src,_,_,_ in data]
    local=ACESModel(state['recipe']['channels'],base.normalization,base.restoration,'sqrt_luma')
    model=ACESModel(state['recipe']['channels'],base.normalization,base.restoration,'sqrt_luma_context') if a.wide else local
    if a.wide:initialize_shadow_pair(local.shadow_predictor,model.shadow_predictor)
    model.predictor.load_state_dict(base.predictor.state_dict());model.restorer.load_state_dict(base.restorer.state_dict())
    model.predictor.requires_grad_(False);model.restorer.requires_grad_(False)
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    optimizer=torch.optim.Adam(model.shadow_predictor.parameters(),lr=.0001);a.out.mkdir(parents=True)
    ancestry=a.out/'training_ancestry.json';ancestry.write_text(json.dumps({'rows':merge_training_ancestry(json.loads(a.base_pilot.read_text())['rows'],pilot['rows'])},indent=2))
    recipe={'steps':a.steps,'seed':a.seed,'channels':state['recipe']['channels'],'crop':192,'border':40,'lr':.0001,
            'wide':a.wide,'loss':'equal domain mean of source-shadow-weighted signed RGB asinh MSE at view-defined scale',
            'include_image_edges':a.include_image_edges,
            'sampling':'two shadow-targeted crops per domain; uniform eligible image then shadow pixel',
            'base_checkpoint_sha256':sha(a.checkpoint),'training_pilot_sha256':sha(a.pilot),
            'initialization':'same seeded local weights; wide extra input weights zero, remaining layers copied',
            'code_sha256':{str(x):sha(x) for x in map(Path,('tools/train_aces_shadow_context.py','training/aces_shadow.py','training/aces_model.py'))}}
    history=[];visits=np.zeros(len(data),int);start=time.time();updates=0
    for step in range(1,a.steps+1):
        source=[];truth=[];baseline=[];valid=[]
        for indices in groups.values():
            for _ in range(2):
                i=int(rng.choice(indices));src,target,_,points=data[i];h,w=src.shape[:2]
                if min(h,w)<192:raise ValueError('crop exceeds input')
                py,px=points[int(rng.integers(len(points)))];y=int(np.clip(py-rng.integers(40,152),0,h-192));x=int(np.clip(px-rng.integers(40,152),0,w-192))
                for values,array in ((source,src),(truth,target),(baseline,baselines[i])):values.append(torch.from_numpy(array[y:y+192,x:x+192].copy()).permute(2,0,1))
                valid.append(valid_crop_context(x,y,w,h))
                visits[i]+=1
        source,target,baseline=[torch.stack(v).to(device) for v in (source,truth,baseline)]
        weight=reconstruction_weight(source)*(source.amax(1,keepdim=True)<=.5)
        if (luminance(baseline)[weight>0]<0).any():raise ValueError('negative baseline shadow Y')
        prediction=compose_shadow(baseline,model.shadow_predictor(source,baseline),weight);s=shadow_reference_scale()
        error=(torch.asinh(prediction/s)-torch.asinh(target/s)).square().mean(1,keepdim=True)
        if a.include_image_edges:weight=weight*torch.stack(valid).to(device)
        else:error=error[:,:,40:-40,40:-40];weight=weight[:,:,40:-40,40:-40]
        loss=equal_domain_loss(error,weight)
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.shadow_predictor.parameters() if p.grad is not None):raise ValueError('nonfinite gradient')
        optimizer.step();updates+=1
        if step==1 or step%100==0 or step==a.steps:
            item={'step':step,'loss':float(loss.detach()),'seconds':time.time()-start};history.append(item);print(item,flush=True)
    frozen=all(torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in base.state_dict().items())
    if not frozen or any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('frozen/finite failure')
    checkpoint=a.out/f'step{a.steps}.pt';torch.save({'model':model.state_dict(),'contract':model.contract(),'recipe':recipe,'step':a.steps,'pilot_sha256':sha(ancestry),'research_only':True,'optimizer':optimizer.state_dict()},checkpoint)
    (a.out/'report.json').write_text(json.dumps({'recipe':recipe,'visits':visits.tolist(),'history':history,'optimizer_updates':updates,'frozen_predictor_and_restorer_bit_exact':frozen,'checkpoint_sha256':sha(checkpoint)},indent=2))


if __name__=='__main__':main()
