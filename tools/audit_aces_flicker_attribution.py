"""Exact raw cell-EV counterfactual attribution; descriptive, not motion compensation."""
import argparse,itertools,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.rudra_aces_export import video_frames,load_model,sha
from tools.evaluate_aces_temporal import cell_luminance,temporal_change,LUMA_AP1
from training.aces_full_context_model import fully_clipped_support
from training.aces_reference import reference_view
from training.aces_colour_backtrack import constrain_colour
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def pool(value):
    return torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(np.asarray(value,dtype=np.float64))[None,None],(36,64))[0,0].numpy()


def attribute(parent0,parent1,base0,base1,corrections):
    """Shapley split over parent luminance, mask and branch delta at fixed cells.

    corrections[(mask_time,delta_time)] is pooled mask*unmasked branch delta.
    Parent contribution also subtracts analytic baseline temporal change.
    All eight counterfactual corners must be positive: do not floor radiance.
    """
    corners={(p,m,d):np.asarray((parent0,parent1)[p])+corrections[m,d] for p,m,d in itertools.product((0,1),repeat=3)}
    valid=(base0>0)&(base1>0)
    for value in corners.values():
        if not np.isfinite(value).all():raise ValueError('nonfinite corner')
        valid&=value>0
    result={'valid_cells':int(valid.sum()),'excluded_cells':int((~valid).sum())}
    if not valid.any():return {**result,'terms':None,'total_mean_abs_ev':None,'closure_max_abs_ev':None}
    logs={key:np.log2(v[valid]) for key,v in corners.items()};terms={}
    for axis,name in enumerate(('parent','mask','field')):
        other=[k for k in range(3) if k!=axis];term=np.zeros(int(valid.sum()))
        for bits in itertools.product((0,1),repeat=2):
            before=[0,0,0]
            for k,bit in zip(other,bits):before[k]=bit
            after=before.copy();after[axis]=1
            term+=(1/3 if sum(bits) in (0,2) else 1/6)*(logs[tuple(after)]-logs[tuple(before)])
        if axis==0:term-=np.log2(base1[valid])-np.log2(base0[valid])
        terms[name]=term
    total=logs[1,1,1]-logs[0,0,0]-(np.log2(base1[valid])-np.log2(base0[valid]))
    closure=float(np.max(np.abs(sum(terms.values())-total)))
    if closure>1e-10:raise ValueError('attribution does not close')
    return {**result,'terms':{k:{'mean_abs_ev':float(np.abs(v).mean()),'mean_signed_ev':float(v.mean())} for k,v in terms.items()},
            'total_mean_abs_ev':float(np.abs(total).mean()),'closure_max_abs_ev':closure}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();reg=json.loads(a.registration.read_text());a.out.mkdir(exist_ok=False)
    (a.out/'auditor.py').write_bytes(Path(__file__).read_bytes());(a.out/'registration.json').write_bytes(a.registration.read_bytes())
    for key in ('candidate','parent'):
        if sha(reg[key])!=reg[key+'_sha256']:raise ValueError('checkpoint identity')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    view=reference_view();view.set_threads(4);model,_=load_model(reg['candidate'],'cuda');parent,_=load_model(reg['parent'],'cuda')
    suite=json.loads(Path(reg['source_registration']).read_text());video=next(x for x in suite['expected_inputs'] if x['clip']==reg['clip'])
    if sha(video['path'])!=video['sha256']:raise ValueError('video identity')
    sequences=[('hunyuan48',video_frames(video['path'],max_frames=48))]
    pilot=json.loads(Path(reg['train_pilot']).read_text());groups={}
    for row in pilot['rows']:groups.setdefault(row['group_id'],[]).append(row)
    def train_frames(rows):
        if [r['frame'] for r in rows]!=[0,1,2] or [r['translation_xy'] for r in rows]!=[[0,0],[.5,.25],[1,.5]]:raise ValueError('known motion')
        for row in rows:
            if row['split']!='train' or sha(row['sdr_path'])!=row['sdr_sha256']:raise ValueError('TRAIN identity')
            yield canonical_to_source(load_rgb(row['sdr_path'],hdr=False))
    for group in reg['train_groups']:sequences.append((group,train_frames(groups[group])))
    clips=[]
    for label,frames in sequences:
        previous=None;rows=[];initial_exact=None
        for frame,source in enumerate(frames):
            x=torch.from_numpy(source).permute(2,0,1)[None].to('cuda')
            with torch.inference_mode():
                old=parent(x);fields=model.full_context(x,old.baseline)[:,3:]
                shared=old.restored_acescg+old.reconstruction_weight*((old.restored_acescg.abs().amax(1,keepdim=True)+.001)*fields)
                support=fully_clipped_support(x,old.reconstruction_weight);pred=torch.where(support,shared,old.acescg)
                if frame==0:
                    initial_exact=torch.equal(model(x).acescg,pred)
                    if not initial_exact:raise ValueError('reconstruction mismatch')
            raw=pred[0].permute(1,2,0).cpu().numpy();oldraw=old.acescg[0].permute(1,2,0).cpu().numpy();base=old.baseline[0].permute(1,2,0).cpu().numpy();w=old.reconstruction_weight[0,0].cpu().numpy();mask=support[0,0].cpu().numpy()
            delta=(shared[0].permute(1,2,0).cpu().numpy().astype(np.float64)-oldraw)@LUMA_AP1
            constrained,_=constrain_colour(source,raw,base,w,view,include_fully_clipped=True)
            oldconstrained,_=constrain_colour(source,oldraw,base,w,view,include_fully_clipped=True)
            current={'parent':cell_luminance(oldraw),'raw':cell_luminance(raw),'base':cell_luminance(base),'constrained':cell_luminance(constrained),'oldconstrained':cell_luminance(oldconstrained),'mask':mask,'delta':delta}
            row={'frame':frame,'full_pixels':int(mask.sum())}
            if previous is not None:
                corrections={(i,j):pool((previous,current)[i]['mask']*(previous,current)[j]['delta']) for i,j in itertools.product((0,1),repeat=2)}
                row['attribution']=attribute(previous['parent'],current['parent'],previous['base'],current['base'],corrections)
                row['support_changed_pixels']=int((mask!=previous['mask']).sum())
                row['transitions']={k:temporal_change(previous[k],current[k],previous['base'],current['base']) for k in ('raw','parent','constrained','oldconstrained')}
            rows.append(row);previous=current;print(label,frame,flush=True)
        if len(rows)!=(48 if label=='hunyuan48' else 3):raise ValueError('frame count')
        transitions=rows[1:];eligible=[r['attribution'] for r in transitions if r['attribution']['terms'] is not None]
        summary={'max_transition':{k:max(r['transitions'][k]['mean_abs_excess_ev'] for r in transitions) for k in ('raw','parent','constrained','oldconstrained')},
            'mean_absolute_attribution_ev':{k:float(np.mean([r['terms'][k]['mean_abs_ev'] for r in eligible])) for k in ('parent','mask','field')},
            'min_counterfactual_valid_cells':min(r['attribution']['valid_cells'] for r in transitions)}
        clips.append({'clip':label,'first_frame_reconstruction_exact':initial_exact,'summary':summary,'rows':rows})
    result={'clips':clips,'registration_sha256':sha(a.registration),'tool_sha256':sha(__file__),'source_sha256':video['sha256'],'train_pilot_sha256':sha(reg['train_pilot']),'scope':reg['scope'],'release_authorized':False}
    (a.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({c['clip']:c['summary'] for c in clips},indent=2))


if __name__=='__main__':main()
