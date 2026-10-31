"""Measure trainable temporal support and normalization; never optimize weights."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import reconstruction_weight,model_from_checkpoint
from training.aces_full_context_model import fully_clipped_support
from training.aces_context_model import aligned_pair_residual_loss
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def supports(high,full,border=4):
    def neighbours(m):
        return [m[1:2,:,:-1,:-1],m[1:2,:,:-1,1:],m[1:2,:,1:,:-1],m[1:2,:,1:,1:]]
    valid=high[:1,:,:-1,:-1].clone()
    for m in neighbours(high):valid &= m
    if border:
        valid[:,:,:border]=False;valid[:,:,-border:]=False
        valid[:,:,:,:border]=False;valid[:,:,:,-border:]=False
    first=full[:1,:,:-1,:-1];any_next=torch.stack(neighbours(full)).any(0);all_next=torch.stack(neighbours(full)).all(0)
    active=valid&(first|any_next)
    stable=valid&first&all_next
    return {'valid':valid,'active':active,'stable_full':stable,
            'entry':valid&~first&any_next,'exit':valid&first&~all_next}


def active_loss(prediction,target,masks):
    residual=torch.asinh(prediction)-torch.asinh(target);b=residual[1:2]
    aligned=.75*(.5*b[:,:,:-1,:-1]+.5*b[:,:,:-1,1:])+.25*(.5*b[:,:,1:,:-1]+.5*b[:,:,1:,1:])
    error=(residual[:1,:,:-1,:-1]-aligned).square().mean(1,keepdim=True)
    return (error*masks['active']).sum()/masks['active'].sum().clamp_min(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    reg=json.loads(a.registration.read_text());a.out.mkdir(exist_ok=False)
    (a.out/'auditor.py').write_bytes(Path(__file__).read_bytes());(a.out/'registration.json').write_bytes(a.registration.read_bytes())
    for key in ('pilot','candidate','attribution'):
        if sha(reg[key])!=reg[key+'_sha256']:raise ValueError(key+' identity')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model_from_checkpoint(torch.load(reg['candidate'],map_location='cpu',weights_only=True)).eval().to('cuda')
    if not hasattr(model,'full_context'):raise ValueError('isolated model required')
    groups={}
    for row in json.loads(Path(reg['pilot']).read_text())['rows']:
        if row['split']!='train':raise ValueError('TRAIN only')
        groups.setdefault(row['group_id'],[]).append(row)
    rows=[];gradient=[]
    for group,items in sorted(groups.items()):
        items=sorted(items,key=lambda r:r['frame'])
        if [r['translation_xy'] for r in items]!=[[0,0],[.5,.25],[1,.5]]:raise ValueError('motion')
        sources=[];targets=[]
        for r in items:
            for key in ('sdr','target'):
                if sha(r[key+'_path'])!=r[key+'_sha256']:raise ValueError('input identity')
            sources.append(canonical_to_source(load_rgb(r['sdr_path'],hdr=False)))
            if group in reg['gradient_groups']:targets.append(np.load(r['target_path']))
        for first in (0,1):
            x=torch.from_numpy(np.stack(sources[first:first+2])).permute(0,3,1,2)
            weight=reconstruction_weight(x);high=(weight>0)&(x.amax(1,keepdim=True)>.5);full=fully_clipped_support(x,weight)
            masks=supports(high,full);counts={k:int(v.sum()) for k,v in masks.items()}
            changed=int((full[0]!=full[1]).sum());union=int((full[0]|full[1]).sum())
            rows.append({'group':group,'first':first,**counts,'active_fraction':counts['active']/counts['valid'] if counts['valid'] else None,'fixed_change_over_full_union':changed/union if union else None})
            if group in reg['gradient_groups']:
                x=x.to('cuda');truth=torch.from_numpy(np.stack(targets[first:first+2])).permute(0,3,1,2).to('cuda')
                pred=model(x).acescg;params=list(model.full_context.parameters())
                all_loss=aligned_pair_residual_loss(pred,truth,high.to('cuda'))
                focused=active_loss(pred,truth,{k:v.to('cuda') for k,v in masks.items()})
                ga=torch.autograd.grad(all_loss,params,retain_graph=True);gf=torch.autograd.grad(focused,params)
                va=torch.cat([v.flatten() for v in ga]);vf=torch.cat([v.flatten() for v in gf]);ratio=counts['active']/max(counts['valid'],1)
                discrepancy=float((va-vf*ratio).norm()/va.norm().clamp_min(1e-12))
                if discrepancy>1e-4:raise ValueError('gradient scaling identity')
                gradient.append({'group':group,'first':first,'all_loss':float(all_loss),'active_loss':float(focused),'all_parameter_gradient_norm':float(va.norm()),'active_parameter_gradient_norm':float(vf.norm()),'expected_scale':ratio,'relative_scaling_error':discrepancy})
        print(len(rows),'pairs',flush=True)
    attr=json.loads(Path(reg['attribution']).read_text());video=next(c for c in attr['clips'] if c['clip']=='hunyuan48')['rows'];vr=[]
    for prev,cur in zip(video,video[1:]):
        change=cur['support_changed_pixels'];union=(prev['full_pixels']+cur['full_pixels']+change)/2
        if union:vr.append(change/union)
    def stats(values):
        v=[x for x in values if x is not None]
        return dict(zip(('min','median','mean','max'),map(float,(np.min(v),np.median(v),np.mean(v),np.max(v))))) if v else None
    summary={'pairs':len(rows),'groups':len(groups),'active_fraction':stats([r['active_fraction'] for r in rows]),'pairs_without_active':sum(r['active']==0 for r in rows),'totals':{k:sum(r[k] for r in rows) for k in ('valid','active','stable_full','entry','exit')},'train_fixed_change_over_full_union':stats([r['fixed_change_over_full_union'] for r in rows]),'video_fixed_change_over_full_union':stats(vr),'gradient_pairs':len(gradient),'gradient_scaling_max_relative_error':max(r['relative_scaling_error'] for r in gradient)}
    (a.out/'report.json').write_text(json.dumps({'summary':summary,'rows':rows,'gradient_rows':gradient,'registration_sha256':sha(a.registration),'auditor_sha256':sha(__file__),'scope':reg['scope'],'release_authorized':False},indent=2,allow_nan=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
