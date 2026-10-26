"""Train a separate shadow representation while freezing v4 restoration/HDR."""
import argparse,json,sys,os,time,math
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import ACESModel,model_from_checkpoint,reconstruction_weight
from training.aces_shadow import compose_shadow,luminance
from training.aces_losses import shadow_reference_scale
from training.aces_reference import reference_view
from tools.train_aces_spatial import load_training_rows
from tools.train_aces_restorer import merge_training_ancestry
from tools.evaluate_run05_dequant import sha
from training.aces_consistency import translate_constant,shadow_error_consistency,quantize_source8


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','base-pilot','pilot','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--shadow-initialization',type=Path)
    p.add_argument('--consistency-weight',type=float,default=0.)
    p.add_argument('--consistency-quantize',action='store_true')
    a=p.parse_args()
    if not math.isfinite(a.consistency_weight) or a.consistency_weight<0:raise ValueError('finite nonnegative consistency weight required')
    if a.out.exists() or a.steps<=0:raise ValueError('new output and positive steps required')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2);torch.manual_seed(20261023);rng=np.random.default_rng(20261023)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if state['pilot_sha256']!=sha(a.base_pilot):raise ValueError('base ancestry mismatch')
    base=model_from_checkpoint(state)
    if base.shadow!='shared' or base.restoration!='local_codes':raise ValueError('v4 local restoration base required')
    pilot=json.loads(a.pilot.read_text());data=load_training_rows(pilot)
    candidates=[i for i,(_,_,_,lo) in enumerate(data) if len(lo)]
    if not candidates:raise ValueError('shadow training support required')
    baselines=[reference_view().inverse(src) for src,_,_,_ in data]
    model=ACESModel(state['recipe']['channels'],base.normalization,base.restoration,'sqrt_luma')
    model.predictor.load_state_dict(base.predictor.state_dict());model.restorer.load_state_dict(base.restorer.state_dict())
    if a.shadow_initialization:
        initial=torch.load(a.shadow_initialization,map_location='cpu',weights_only=True)
        previous=model_from_checkpoint(initial)
        if previous.shadow!='sqrt_luma' or initial['recipe']['base_checkpoint_sha256']!=sha(a.checkpoint):raise ValueError('shadow initialization ancestry mismatch')
        if initial['recipe']['training_pilot_sha256']!=sha(a.pilot):raise ValueError('shadow initialization TRAIN mismatch')
        model.shadow_predictor.load_state_dict(previous.shadow_predictor.state_dict())
    model.predictor.requires_grad_(False);model.restorer.requires_grad_(False)
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    optimizer=torch.optim.Adam(model.shadow_predictor.parameters(),lr=.0001)
    a.out.mkdir(parents=True);ancestry=a.out/'training_ancestry.json'
    ancestry.write_text(json.dumps({'rows':merge_training_ancestry(json.loads(a.base_pilot.read_text())['rows'],pilot['rows']),
        'scope':'union of frozen-base and shadow TRAIN ancestry'},indent=2))
    recipe={'steps':a.steps,'seed':20261023,'lr':.0001,'batch':4,'crop':128,'border':8,
            'channels':state['recipe']['channels'],'base_checkpoint_sha256':sha(a.checkpoint),'training_pilot_sha256':sha(a.pilot),
            'shadow_scale':shadow_reference_scale(),'sampling':'uniform TRAIN images with shadow pixels; uniform shadow pixel then random interior crop offset',
            'shadow_initialization_sha256':sha(a.shadow_initialization) if a.shadow_initialization else None,
            'consistency_weight':a.consistency_weight,'consistency_transform':'independent per-image translation uniform[-1,1]px on each axis; CPU bilinear fixed data; RNG20261030',
            'consistency_quantize8':a.consistency_quantize,
            'loss':'source-weighted mean signed RGB asinh error at view-defined shadow scale, shadow-only,8px border',
            'code_sha256':{name:sha(name) for name in ('tools/train_aces_shadow.py','training/aces_shadow.py','training/aces_model.py','training/aces_consistency.py')},
            'scope':'shadow-only training with fresh optimizer; optional weights initialization is not resume; final checkpoint only, not release'}
    visits=np.zeros(len(data),int);history=[];updates=0;start=time.time();transform_rng=np.random.default_rng(20261030)
    for step in range(1,a.steps+1):
        sources=[];targets=[];bases=[]
        for _ in range(4):
            i=int(rng.choice(candidates));src,target,_,points=data[i];h,w=src.shape[:2]
            if min(h,w)<128:raise ValueError('crop exceeds image')
            py,px=points[int(rng.integers(len(points)))];y=int(np.clip(py-rng.integers(8,120),0,h-128));x=int(np.clip(px-rng.integers(8,120),0,w-128))
            for values,array in ((sources,src),(targets,target),(bases,baselines[i])):
                values.append(torch.from_numpy(array[y:y+128,x:x+128].copy()).permute(2,0,1))
            visits[i]+=1
        cpu_source=torch.stack(sources);cpu_target=torch.stack(targets)
        source,target,baseline=cpu_source.to(device),cpu_target.to(device),torch.stack(bases).to(device)
        weight=reconstruction_weight(source)*(source.amax(1,keepdim=True)<=.5)
        if (luminance(baseline)[weight>0]<0).any():raise ValueError('negative baseline shadow luminance')
        prediction=compose_shadow(baseline,model.shadow_predictor(source,baseline),weight)
        consistency=prediction.sum()*0
        if a.consistency_weight:
            shifts=torch.from_numpy(transform_rng.uniform(-1,1,(4,2)).astype(np.float32))
            shifted_source_cpu=translate_constant(cpu_source,shifts)
            if a.consistency_quantize:shifted_source_cpu=quantize_source8(shifted_source_cpu)
            shifted_target=translate_constant(cpu_target,shifts).to(device)
            values=shifted_source_cpu.permute(0,2,3,1).contiguous().numpy()
            shifted_baseline=torch.from_numpy(reference_view().inverse(values)).permute(0,3,1,2).to(device)
            shifted_source=shifted_source_cpu.to(device)
            shifted_weight=reconstruction_weight(shifted_source)*(shifted_source.amax(1,keepdim=True)<=.5)
            shifted_prediction=compose_shadow(shifted_baseline,model.shadow_predictor(shifted_source,shifted_baseline),shifted_weight)
            consistency=shadow_error_consistency(prediction,target,shifted_prediction,shifted_target,weight,shifted_weight)
        scale=shadow_reference_scale();error=(torch.asinh(prediction/scale)-torch.asinh(target/scale)).square().mean(1,keepdim=True)
        weight=weight[:,:,8:-8,8:-8];error=error[:,:,8:-8,8:-8]
        loss=(error*weight).sum()/weight.sum().clamp_min(1)
        reconstruction=loss;loss=loss+a.consistency_weight*consistency
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        if weight.sum()>0:
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.shadow_predictor.parameters() if p.grad is not None):raise ValueError('nonfinite gradient')
            optimizer.step();updates+=1
        if step==1 or step%100==0 or step==a.steps:
            item={'step':step,'loss':float(loss.detach()),'reconstruction':float(reconstruction.detach()),'consistency':float(consistency.detach()),'seconds':time.time()-start};history.append(item);print(item,flush=True)
    frozen=all(torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in base.state_dict().items())
    if not frozen:raise ValueError('frozen predictor/restorer changed')
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite weights')
    checkpoint=a.out/f'step{a.steps}.pt'
    torch.save({'model':model.state_dict(),'contract':model.contract(),'recipe':recipe,'step':a.steps,
                'pilot_sha256':sha(ancestry),'optimizer':optimizer.state_dict(),'research_only':True},checkpoint)
    (a.out/'report.json').write_text(json.dumps({'recipe':recipe,'history':history,'visits':visits.tolist(),
        'shadow_candidate_images':candidates,'optimizer_updates':updates,'frozen_predictor_and_restorer_bit_exact':frozen,
        'checkpoint_sha256':sha(checkpoint)},indent=2))


if __name__=='__main__':main()
