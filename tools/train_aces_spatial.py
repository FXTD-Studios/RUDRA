"""Fresh, bounded development training with varied TRAIN crops; not a release trainer."""
import argparse,json,sys,time,math,os
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import ACESModel,reconstruction_weight
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha
from training.aces_losses import shadow_relative_loss,shadow_reference_scale,restoration_code_loss


def load_training_rows(pilot):
    rows=pilot['rows']
    if not rows or any(r['split']!='train' for r in rows):
        raise ValueError('only nonempty TRAIN data may reach optimizer')
    loaded=[]
    for row in rows:
        if sha(row['sdr_path'])!=row['sdr_sha256'] or sha(row['target_path'])!=row['target_sha256']:
            raise ValueError('training input identity mismatch')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
        if source.shape!=target.shape or not np.isfinite(target).all():raise ValueError('invalid target')
        top=source.max(-1)
        loaded.append((source,target,np.argwhere(top>=254.5/255),np.argwhere(top<3/255)))
    return loaded


def sample_crop(data,rng,size):
    index=int(rng.integers(len(data)));source,target,hi,lo=data[index]
    mode=int(rng.integers(4));points=hi if mode==0 else lo if mode==1 else ()
    h,w=source.shape[:2]
    if h<size or w<size:raise ValueError('crop exceeds image')
    if len(points):
        py,px=points[int(rng.integers(len(points)))]
        # Random offset within the crop avoids centring every clipped pixel.
        y=int(np.clip(py-rng.integers(size),0,h-size));x=int(np.clip(px-rng.integers(size),0,w-size))
    else:y=int(rng.integers(h-size+1));x=int(rng.integers(w-size+1))
    return (torch.from_numpy(source[y:y+size,x:x+size].copy()).permute(2,0,1),
            torch.from_numpy(target[y:y+size,x:x+size].copy()).permute(2,0,1),
            (index,x,y,mode))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=300);p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--shadow-weight',type=float,default=0.)
    p.add_argument('--restoration-weight',type=float,default=0.)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--deterministic',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args()
    if args.out.exists() or args.steps<=0:raise ValueError('new output and positive steps required')
    if not math.isfinite(args.shadow_weight) or args.shadow_weight<0:raise ValueError('finite nonnegative shadow weight required')
    if not math.isfinite(args.restoration_weight) or args.restoration_weight<0:raise ValueError('finite nonnegative restoration weight required')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0:raise ValueError('finite positive learning rate required')
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    pilot=json.loads(args.pilot.read_text());data=load_training_rows(pilot)
    args.out.mkdir(parents=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ACESModel(8,normalization='pixel_rms').to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    recipe={'seed':args.seed,'steps':args.steps,'channels':8,'crop':128,'batch':4,'lr':args.learning_rate,
            'shadow_weight':args.shadow_weight,'shadow_scale_ap1':shadow_reference_scale(),
            'restoration_weight':args.restoration_weight,
            'deterministic':args.deterministic,
            'normalization':'pixel_rms','sampling':'uniform image;25%highlight25%shadow50%uniform locations',
            'pilot_sha256':sha(args.pilot),'device':device,
            'code_sha256':{str(path):sha(path) for path in map(Path,('tools/train_aces_spatial.py','training/aces_model.py','training/aces_reference.py','training/aces_losses.py','rudra/aces_view.py'))}}
    (args.out/'recipe.json').write_text(json.dumps(recipe,indent=2))
    history=[];visits=np.zeros(len(data),int);locations=set();start=time.time()
    for step in range(1,args.steps+1):
        batch=[sample_crop(data,rng,128) for _ in range(4)]
        for _,_,location in batch:visits[location[0]]+=1;locations.add(location[:3])
        source=torch.stack([b[0] for b in batch]).to(device);target=torch.stack([b[1] for b in batch]).to(device)
        optimizer.zero_grad(set_to_none=True);out=model(source)
        error=(torch.asinh(out.acescg)-torch.asinh(target)).square().mean(1,keepdim=True)
        weight=out.reconstruction_weight
        loss=(error*weight).sum()/weight.sum().clamp_min(1)+error.mean()
        if args.shadow_weight:
            loss=loss+args.shadow_weight*shadow_relative_loss(out,target,source)
        if args.restoration_weight:
            loss=loss+args.restoration_weight*restoration_code_loss(out,target)
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError('nonfinite gradient')
        optimizer.step()
        if step==1 or step%25==0 or step==args.steps:
            item={'step':step,'sampled_batch_loss':float(loss.detach()),'elapsed_seconds':time.time()-start}
            history.append(item);print(item,flush=True)
            (args.out/'progress.json').write_text(json.dumps({'history':history,'visited_images':int((visits>0).sum()),'unique_crops':len(locations)},indent=2))
        if step%100==0 or step==args.steps:
            torch.save({'contract':model.contract(),'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'step':step,'research_only':True,'pilot_sha256':sha(args.pilot),'recipe':recipe,
                        'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,
                        'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},args.out/f'step{step}.pt')
    checkpoint=args.out/f'step{args.steps}.pt'
    (args.out/'report.json').write_text(json.dumps({'recipe':recipe,'contract':model.contract(),
        'checkpoint_sha256':sha(checkpoint),'pilot_sha256':sha(args.pilot),'visits':visits.tolist(),
        'unique_crops':len(locations),'history':history,'scope':'fresh TRAIN spatial development run; checkpoint snapshots but no verified resume command; not validation or release'},indent=2))


if __name__=='__main__':main()
