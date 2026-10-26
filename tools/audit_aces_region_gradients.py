"""TRAIN-only partial/full objective-gradient interference; no parameter updates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint
from training.aces_context_batches import sample_replay_crop
from training.aces_detail_loss import highlight_gradient_loss
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    reg=json.loads(a.registration.read_text());a.out.mkdir(exist_ok=False)
    (a.out/'auditor.py').write_bytes(Path(__file__).read_bytes());(a.out/'registration.json').write_bytes(a.registration.read_bytes())
    if sha(reg['checkpoint'])!=reg['checkpoint_sha256']:raise ValueError('checkpoint identity')
    torch.set_num_threads(2);reference_view().set_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model_from_checkpoint(torch.load(reg['checkpoint'],map_location='cpu',weights_only=True)).eval().to('cuda')
    params=list(model.highlight_context.parameters());selection=json.loads(Path(reg['selection']).read_text())['rows'];rng=np.random.default_rng(reg['seed'])
    pilots={p:{r['asset_id']:r for r in json.loads(Path(p).read_text())['rows']} for p in {s['pilot'] for s in selection}};rows=[]
    for s in selection:
        r=pilots[s['pilot']][s['asset_id']]
        if r['split']!='train' or r['scene_id']!=s['scene_id']:raise ValueError('TRAIN identity')
        for k in ('sdr','target'):
            if sha(r[k+'_path'])!=r[k+'_sha256']:raise ValueError('content identity')
        source=canonical_to_source(load_rgb(r['sdr_path'],hdr=False));target=np.load(r['target_path'])
        x=torch.from_numpy(source).permute(2,0,1)[None].to('cuda');t=torch.from_numpy(target).permute(2,0,1)[None].to('cuda')
        with torch.no_grad():out=model(x)
        base=out.baseline;restored=out.restored_acescg;weight=out.reconstruction_weight;borders=[4]*4;crop=None
        if s['domain']=='replay':
            eligible=((weight[0,0]>0)&(x[0].amax(0)>.5)).cpu().numpy();eligible[:4]=False;eligible[-4:]=False;eligible[:,:4]=False;eligible[:,-4:]=False
            points=np.argwhere(eligible)
            if not len(points):rows.append({**s,'excluded':'no interior support'});continue
            crop,borders=sample_replay_crop(*source.shape[:2],points,rng);y,z,h,w=crop
            x,t,base,restored,weight=[v[:,:,y:y+h,z:z+w] for v in (x,t,base,restored,weight)]
        high=(weight>0)&(x.amax(1,keepdim=True)>.5);full=high&(x.amin(1,keepdim=True)>=254.5/255)
        valid=torch.ones_like(high);top,bottom,left,right=borders;valid[:,:,:top]=False;valid[:,:,-bottom:]=False;valid[:,:,:,:left]=False;valid[:,:,:,-right:]=False
        masks={'partial':high&~full&valid,'full':full&valid};counts={k:int(m.sum()) for k,m in masks.items()}
        row={**s,'crop':crop,'pixels':counts,'cosine':None}
        if all(counts.values()):
            pred,_=model.predict_highlights(x,base,restored,weight);error=(torch.asinh(pred)-torch.asinh(t)).square().mean(1,keepdim=True);grads={};losses={}
            for name,m in masks.items():
                loss=(error*m).sum()/m.sum()+highlight_gradient_loss(pred,t,m)
                gradients=torch.autograd.grad(loss,params,retain_graph=name=='partial',allow_unused=False)
                grads[name]=torch.cat([v.detach().flatten() for v in gradients]).double();losses[name]=float(loss.detach())
            norms={k:float(v.norm()) for k,v in grads.items()};denom=norms['partial']*norms['full']
            row.update(losses=losses,norms=norms,cosine=float(torch.dot(grads['partial'],grads['full']))/denom if denom>0 else None)
            if any(not np.isfinite(v) for v in norms.values()) or (row['cosine'] is not None and not np.isfinite(row['cosine'])):raise ValueError('nonfinite gradient')
        rows.append(row);print('gradient audit',len(rows),flush=True)
    summary={}
    for domain in ('motion','replay'):
        rr=[r for r in rows if r['domain']==domain and r.get('cosine') is not None];values=[r['cosine'] for r in rr]
        summary[domain]={'eligible_both_regions':len(rr),'negative_cosine':sum(v<0 for v in values),'median_cosine':float(np.median(values)) if values else None}
    (a.out/'report.json').write_text(json.dumps({'summary':summary,'rows':rows,'registration_sha256':sha(a.registration),'tool_sha256':sha(__file__),'pilot_hashes':{p:sha(p) for p in pilots},'scope':reg['scope'],'release_authorized':False},indent=2,allow_nan=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
