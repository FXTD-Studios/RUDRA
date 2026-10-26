"""Fixed TRAIN crop learnability probe. Not production training or validation."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import ACESModel,reconstruction_weight
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--normalization',choices=('group','pixel_rms'),default='group')
    p.add_argument('--steps',type=int,default=60);a=p.parse_args()
    if a.out.exists():raise ValueError('new output directory required')
    torch.manual_seed(20261023);torch.set_num_threads(2)
    pilot=json.loads(a.pilot.read_text());a.out.mkdir(parents=True)
    inputs=[];targets=[];crops=[]
    for index,kind in ((0,'highlight'),(6,'highlight'),(6,'shadow'),(8,'shadow')):
        row=pilot['rows'][index];assert row['split']=='train'
        assert sha(row['sdr_path'])==row['sdr_sha256'] and sha(row['target_path'])==row['target_sha256']
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
        s=torch.from_numpy(source).permute(2,0,1)[None]
        weight=reconstruction_weight(s)[0,0].numpy();top=source.max(-1)
        support=weight*((top>.5) if kind=='highlight' else (top<=.5))
        # Choose the densest fixed-size support window, not a loss-selected crop.
        pooled=torch.nn.functional.avg_pool2d(torch.from_numpy(support)[None,None],128,stride=32)[0,0]
        yy,xx=np.unravel_index(int(pooled.argmax()),pooled.shape);y=int(yy)*32;x=int(xx)*32
        inputs.append(s[0,:,y:y+128,x:x+128]);targets.append(torch.from_numpy(target[y:y+128,x:x+128]).permute(2,0,1))
        crops.append({'asset_id':row['asset_id'],'kind':kind,'xy':[x,y],'size':128})
    device='cuda' if torch.cuda.is_available() else 'cpu'
    source=torch.stack(inputs).to(device);target=torch.stack(targets).to(device)
    model=ACESModel(8,normalization=a.normalization).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    weight=reconstruction_weight(source);history=[];start=time.time()
    for step in range(a.steps+1):
        optimizer.zero_grad(set_to_none=True);out=model(source)
        error=(torch.asinh(out.acescg)-torch.asinh(target)).square().mean(1,keepdim=True)
        loss=(error*weight).sum()/weight.sum().clamp_min(1)+error.mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite smoke loss')
        if step%10==0 or step==a.steps:
            history.append({'step':step,'loss':float(loss.detach())});print(history[-1],flush=True)
        if step<a.steps:
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError('nonfinite model gradient')
            optimizer.step()
    with torch.no_grad():
        output=model(source)
        from training.aces_reference import reference_view
        codes=reference_view().forward(output.acescg.cpu().permute(0,2,3,1).numpy())
        src=source.cpu().permute(0,2,3,1).numpy();top=src.max(-1)
        ordinary=(top>=5/255)&(top<=254/255)
        errors=np.abs(codes-src).max(-1)*255
    checkpoint=a.out/'smoke.pt'
    torch.save({'contract':model.contract(),'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'step':a.steps,'research_only':True,'pilot_sha256':sha(a.pilot)},checkpoint)
    (a.out/'report.json').write_text(json.dumps({'contract':model.contract(),'history':history,'crops':crops,
        'device':device,'elapsed_seconds':time.time()-start,'checkpoint_sha256':sha(checkpoint),
        'pilot_sha256':sha(a.pilot),'ordinary_pixels':int(ordinary.sum()),
        'ordinary_within1_fraction':float((errors[ordinary]<=1).mean()),
        'ordinary_max_code_error':float(errors[ordinary].max()),
        'scope':'four fixed128 TRAIN crops, channels8, Adam.001, no holdout/generalization/production claim'},indent=2))


if __name__=='__main__':main()
