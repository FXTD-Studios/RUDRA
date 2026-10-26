"""Fixed source-defined highlight strata; descriptive diagnosis, no optimization."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import reconstruction_weight,model_from_checkpoint
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def strata(source,high):
    full=high&(source.min(-1)>=254.5/255)
    # Padding makes image-boundary behavior explicit, including fully white images.
    distance=distance_transform_edt(np.pad(full,1))[1:-1,1:-1]
    valid=np.ones_like(full);valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
    return {'partial':high&~full&valid,'full_boundary':full&(distance<=2)&valid,
            'full_near':full&(distance>2)&(distance<=8)&valid,'full_deep':full&(distance>8)&valid}


def metrics(pred,target,mask):
    residual=np.arcsinh(pred)-np.arcsinh(target)
    count=int(mask.sum());edge_count=0;total=0.
    for axis in (0,1):
        a=[slice(None)]*2;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
        edges=mask[tuple(a)]&mask[tuple(b)]
        error=np.mean(np.diff(residual,axis=axis)**2,-1)
        edge_count+=int(edges.sum());total+=float(error[edges].sum())
    return {'pixels':count,'edges':edge_count,'asinh_mse':float(np.mean(residual[mask]**2)) if count else None,
            'signed_asinh_bias':float(residual[mask].mean()) if count else None,
            'gradient_mse':total/edge_count if edge_count else None}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    reg=json.loads(a.registration.read_text());a.out.mkdir(exist_ok=False)
    (a.out/'auditor.py').write_bytes(Path(__file__).read_bytes());(a.out/'registration.json').write_bytes(a.registration.read_bytes())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    view=reference_view();view.set_threads(4);models={}
    for name,item in reg['models'].items():
        if sha(item['checkpoint'])!=item['sha256']:raise ValueError('checkpoint identity')
        models[name]=model_from_checkpoint(torch.load(item['checkpoint'],map_location='cpu',weights_only=True)).eval().to('cuda')
    selection=json.loads(Path(reg['train_selection']).read_text())['rows'];work=[];pilot_hashes={}
    for item in selection:work.append(('train_'+item['domain'],item['pilot'],item['asset_id'],item['scene_id']))
    for label,pilot in reg['development'].items():
        for row in json.loads(Path(pilot).read_text())['rows']:work.append((label,pilot,row['asset_id'],row['scene_id']))
    pilots={p:{r['asset_id']:r for r in json.loads(Path(p).read_text())['rows']} for _,p,_,_ in work}
    pilot_hashes={p:sha(p) for p in pilots};saved={};report_hashes={}
    for label in reg['development']:
        for name,item in reg['models'].items():
            for version in ('candidate_raw','candidate'):
                path=Path(item['root'])/'paired_development'/label/(version+'_report.json')
                rr=json.loads(path.read_text())
                if rr['checkpoint_sha256']!=item['sha256'] or rr['pilot_sha256']!=pilot_hashes[reg['development'][label]]:raise ValueError('saved report identity')
                saved[label,name,version]={r['asset_id']:r for r in rr['rows']};report_hashes[str(path)]=sha(path)
    rows=[]
    for label,pilot,asset,scene in work:
        row=pilots[pilot][asset]
        if row['scene_id']!=scene or row['split']!=('train' if label.startswith('train_') else 'val'):raise ValueError('row identity')
        for key in ('sdr','target'):
            if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('content identity')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
        x=torch.from_numpy(source).permute(2,0,1)[None];weight=reconstruction_weight(x)[0,0].numpy()
        masks=strata(source,(weight>0)&(source.max(-1)>.5));arrays={'analytic':view.inverse(source)}
        for name,model in models.items():
            if label.startswith('train_'):
                with torch.inference_mode():value=model(x.to('cuda'))
                arrays[name+'_raw']=value.acescg[0].permute(1,2,0).cpu().numpy().copy()
            else:
                for version in ('candidate_raw','candidate'):
                    rr=saved[label,name,version][asset]
                    if sha(rr['prediction_path'])!=rr['prediction_sha256']:raise ValueError('prediction identity')
                    arrays[name+('_raw' if version=='candidate_raw' else '_constrained')]=np.load(rr['prediction_path'])
        if any(v.shape!=target.shape or not np.isfinite(v).all() for v in arrays.values()):raise ValueError('prediction shape/finite')
        result={region:{name:metrics(v,target,mask) for name,v in arrays.items()} for region,mask in masks.items()}
        rows.append({'dataset':label,'asset_id':asset,'scene_id':scene,'strata':result});print(label,len(rows),flush=True)
    summary={}
    for label in sorted({r['dataset'] for r in rows}):
        selected=[r for r in rows if r['dataset']==label];summary[label]={}
        for region in selected[0]['strata']:
            summary[label][region]={}
            for name in selected[0]['strata'][region]:
                values=[r['strata'][region][name] for r in selected]
                item={'images_with_pixels':sum(v['pixels']>0 for v in values),'total_pixels':sum(v['pixels'] for v in values)}
                for metric in ('asinh_mse','gradient_mse','signed_asinh_bias'):
                    vv=[v[metric] for v in values if v[metric] is not None];item[metric]=float(np.mean(vv)) if vv else None
                summary[label][region][name]=item
    result={'summary':summary,'rows':rows,'registration_sha256':sha(a.registration),'tool_sha256':sha(__file__),
            'pilot_hashes':pilot_hashes,'saved_report_hashes':report_hashes,'scope':reg['scope'],'release_authorized':False}
    (a.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False));print('Complete',a.out,flush=True)


if __name__=='__main__':main()
