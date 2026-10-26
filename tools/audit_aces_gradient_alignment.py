"""Read-only TRAIN gradient geometry and paired DEV regression attribution."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint
from training.aces_active_temporal import active_temporal_loss
from training.aces_detail_loss import highlight_gradient_loss
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def geometry(a,b):
    a=a.double();b=b.double();na=float(a.norm());nb=float(b.norm());dot=float(a@b)
    return {'dot':dot,'cosine':dot/(na*nb) if na and nb else None,'first_norm':na,'second_norm':nb}


def summarize(rows):
    out={}
    for key in ('temporal_photo','temporal_detail','temporal_spatial','photo_detail'):
        values=[r[key]['cosine'] for r in rows if r[key]['cosine'] is not None]
        out[key]={'valid':len(values),'negative':sum(x<0 for x in values),'median_cosine':float(np.median(values)) if values else None}
    out['combined_step_opposes_spatial']=sum(r['spatial_dot_total']<0 for r in rows)
    out['combined_step_opposes_temporal']=sum(r['temporal_dot_total']<0 for r in rows)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    reg=json.loads(a.registration.read_text())
    for path,digest in reg['input_hashes'].items():
        if sha(path)!=digest:raise ValueError('input identity: '+path)
    a.out.mkdir(exist_ok=False);(a.out/'auditor.py').write_bytes(Path(__file__).read_bytes());(a.out/'registration.json').write_bytes(a.registration.read_bytes())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models={name:model_from_checkpoint(torch.load(path,map_location='cpu',weights_only=True)).eval().to('cuda') for name,path in reg['models'].items()}
    if any(not hasattr(m,'full_context') for m in models.values()):raise ValueError('isolated models required')
    groups={}
    for row in json.loads(Path(reg['train_pilot']).read_text())['rows']:
        if row['split']!='train':raise ValueError('TRAIN only')
        groups.setdefault(row['group_id'],[]).append(row)
    result=[]
    for group,items in sorted(groups.items()):
        items=sorted(items,key=lambda r:r['frame'])
        if [r['translation_xy'] for r in items]!=[[0,0],[.5,.25],[1,.5]]:raise ValueError('known motion')
        sources=[];targets=[]
        for row in items:
            for key in ('sdr','target'):
                if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('source identity')
            sources.append(canonical_to_source(load_rgb(row['sdr_path'],hdr=False)));targets.append(np.load(row['target_path']))
        for first in (0,1):
            x=torch.from_numpy(np.stack(sources[first:first+2])).permute(0,3,1,2).to('cuda');target=torch.from_numpy(np.stack(targets[first:first+2])).permute(0,3,1,2).to('cuda')
            for name,model in models.items():
                output=model(x);prediction=output.acescg;weight=output.reconstruction_weight
                full=(weight>0)&(x.amin(1,keepdim=True)>=254.5/255);high=(weight>0)&(x.amax(1,keepdim=True)>.5)
                valid=full.clone();valid[:,:,:4]=False;valid[:,:,-4:]=False;valid[:,:,:,:4]=False;valid[:,:,:,-4:]=False
                if not valid.flatten(1).any(1).all():raise ValueError('empty photo support')
                error=(torch.asinh(prediction)-torch.asinh(target)).square().mean(1,keepdim=True)
                photo=((error*valid).flatten(1).sum(1)/valid.flatten(1).sum(1)).mean()
                detail=highlight_gradient_loss(prediction,target,valid);temporal,active,total=active_temporal_loss(prediction,target,high,full)
                grads={}
                for label,loss in [('photo',photo),('detail',detail),('temporal',temporal)]:
                    gg=torch.autograd.grad(loss,list(model.full_context.parameters()),retain_graph=label!='temporal')
                    vector=torch.cat([g.flatten() for g in gg]).detach().double()
                    if not torch.isfinite(vector).all():raise ValueError('nonfinite gradient')
                    grads[label]=vector
                gp,gd,gt=[grads[k] for k in ('photo','detail','temporal')];gs=gp+gd;allg=gs+gt
                result.append({'model':name,'group':group,'scene':items[0]['scene_id'],'first':first,'active_count':active,'valid_count':total,'losses':{'photo':float(photo.detach()),'detail':float(detail.detach()),'temporal':float(temporal.detach())},'temporal_photo':geometry(gt,gp),'temporal_detail':geometry(gt,gd),'temporal_spatial':geometry(gt,gs),'photo_detail':geometry(gp,gd),'spatial_dot_total':float(gs@allg),'temporal_dot_total':float(gt@allg)})
        print('group',len(result)//(2*len(models)),flush=True)
    dev={}
    for label,spec in reg['development'].items():
        pilot=json.loads(Path(spec['pilot']).read_text());reports={}
        for name,path in spec['reports'].items():
            report=json.loads(Path(path).read_text())
            if report['pilot_sha256']!=sha(spec['pilot']):raise ValueError('pilot report identity')
            reports[name]={r['asset_id']:r for r in report['rows']}
            if set(reports[name])!={r['asset_id'] for r in pilot['rows']}:raise ValueError('incomplete development')
        rows=[]
        for row in pilot['rows']:
            errors={name:rr[row['asset_id']]['target_errors']['model']['highlight_mse'] for name,rr in reports.items()}
            if any(v is None for v in errors.values()):continue
            rows.append({'asset_id':row['asset_id'],'scene':row['scene_id'],'full_fraction':row.get('fully_clipped_fraction'),'desired_fraction':row.get('desired_fully_clipped_fraction'),'errors':errors,'delta':errors['active']-errors['all_highlight']})
        def aggregate(key):
            out=[]
            for value in sorted({str(r[key]) for r in rows}):
                rr=[r for r in rows if str(r[key])==value]
                out.append({'group':value,'images':len(rr),'mean_delta':float(np.mean([r['delta'] for r in rr])),'contribution_to_total_mean':sum(r['delta'] for r in rr)/len(rows),'worse_images':sum(r['delta']>0 for r in rr)})
            return sorted(out,key=lambda r:r['mean_delta'],reverse=True)
        dev[label]={'rows':rows,'by_scene':aggregate('scene'),'by_desired_fraction':aggregate('desired_fraction'),'mean_delta':float(np.mean([r['delta'] for r in rows]))}
    summary={name:summarize([r for r in result if r['model']==name]) for name in models}
    out={'summary':summary,'gradient_rows':result,'development':dev,'registration_sha256':sha(a.registration),'auditor_sha256':sha(__file__),'scope':reg['scope'],'release_authorized':False}
    (a.out/'report.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'summary':summary,'development':{k:{j:v[j] for j in ('by_scene','by_desired_fraction','mean_delta')} for k,v in dev.items()}},indent=2))


if __name__=='__main__':main()
