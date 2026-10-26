"""Fixed TRAIN-only fit diagnosis; never an acceptance or release evaluation."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.summarize_aces_generated import sha
from training.aces_model import model_from_checkpoint
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_context_train_fit_20261024.json'));parser.add_argument('--out',type=Path,default=Path('G:/rudra_training/aces_context_balanced1800_20261024/train_fit.json'))
    args=parser.parse_args();regpath=args.registration
    reg=json.loads(regpath.read_text())
    out=args.out
    if out.exists():raise ValueError('new output required')
    torch.set_num_threads(2);reference_view().set_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out.with_suffix('.auditor.py').write_bytes(Path(__file__).read_bytes())
    models={}
    for name,path in reg['candidates'].items():
        if sha(path)!=reg['checkpoint_sha256'][name]:raise ValueError('checkpoint identity')
        models[name]=model_from_checkpoint(torch.load(path,map_location='cpu',weights_only=True)).eval().to('cuda')
    pilots={p:{r['asset_id']:r for r in json.loads(Path(p).read_text())['rows']} for p in {x['pilot'] for x in reg['rows']}}
    rows=[]
    for selection in reg['rows']:
        row=pilots[selection['pilot']][selection['asset_id']]
        if row['split']!='train' or row['scene_id']!=selection['scene_id']:raise ValueError('TRAIN identity')
        for k in ('sdr','target'):
            if sha(row[k+'_path'])!=row[k+'_sha256']:raise ValueError('content identity')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
        x=torch.from_numpy(source).permute(2,0,1)[None].to('cuda');arrays={}
        for name,model in models.items():
            with torch.inference_mode():value=model(x)
            arrays[name]=value.acescg[0].permute(1,2,0).cpu().numpy().copy()
        arrays['analytic']=value.baseline[0].permute(1,2,0).cpu().numpy()
        high=(value.reconstruction_weight[0,0].cpu().numpy()>0)&(source.max(-1)>.5)
        full=high&(source.min(-1)>=254.5/255)
        errors={}
        for region,mask in (('highlight',high),('fully_clipped',full)):
            errors[region]={name:float(np.mean((np.arcsinh(v[mask])-np.arcsinh(target[mask]))**2)) if mask.any() else None for name,v in arrays.items()}
        rows.append({**selection,'errors':errors});print('audited TRAIN',len(rows),flush=True)
    summary={}
    for domain in ('motion','replay'):
        summary[domain]={}
        for region in ('highlight','fully_clipped'):
            summary[domain][region]={}
            for name in arrays:
                values=[r['errors'][region][name] for r in rows if r['domain']==domain and r['errors'][region][name] is not None]
                summary[domain][region][name]=float(np.mean(values)) if values else None
    report={'summary':summary,'rows':rows,'registration_sha256':sha(regpath),'tool_sha256':sha(__file__),
            'pilot_sha256':{p:sha(p) for p in pilots},'scope':reg['purpose'],'release_authorized':False}
    out.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
