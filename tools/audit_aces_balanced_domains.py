"""Read-only domain gradient geometry on every frozen balanced TRAIN window."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.summarize_aces_generated import sha
from tools.audit_aces_gradient_alignment import geometry
from training.aces_model import model_from_checkpoint
from training.aces_full_context_model import ACESFullContextModel
from training.aces_reference import reference_view
from training.aces_context_batches import stack_context_arrays
from training.aces_detail_loss import highlight_gradient_loss
from training.aces_active_temporal import active_temporal_loss
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();reg=json.loads(a.registration.read_text())
    for path,digest in reg['input_hashes'].items():
        if sha(path)!=digest:raise ValueError('input identity '+path)
    schedule=json.loads(Path(reg['schedule']).read_text())
    if len(schedule)!=600 or {s['window'] for s in schedule}!=set(range(100)):raise ValueError('complete schedule required')
    a.out.mkdir(exist_ok=False);(a.out/'registration.json').write_bytes(a.registration.read_bytes());(a.out/'auditor.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2);reference_view().set_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent=model_from_checkpoint(torch.load(reg['parent'],map_location='cpu',weights_only=True)).eval().to('cuda')
    models={'initial':ACESFullContextModel.from_parent(parent).eval().to('cuda')}
    for name,path in reg['models'].items():models[name]=model_from_checkpoint(torch.load(path,map_location='cpu',weights_only=True)).eval().to('cuda')
    for model in models.values():
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in parent.state_dict().items()):raise ValueError('frozen parent differs')
    parameters={name:list(model.full_context.parameters()) for name,model in models.items()}
    initial_vectors={name:torch.nn.utils.parameters_to_vector(params).detach().clone() for name,params in parameters.items()}
    pilots={domain:{r['asset_id']:r for r in json.loads(Path(path).read_text())['rows']} for domain,path in reg['pilots'].items()}
    needed=sorted({(s['domain'],asset) for s in schedule for asset in s['assets']});cache=[];lookup={};metadata={}
    for domain,asset in needed:
        row=pilots[domain][asset]
        if row['split']!='train':raise ValueError('TRAIN only')
        for key in ('sdr','target'):
            if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('content identity')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
        with torch.inference_mode():value=parent(torch.from_numpy(source).permute(2,0,1)[None].to('cuda'))
        lookup[domain,asset]=len(cache);metadata[domain,asset]=row
        cache.append(tuple(np.ascontiguousarray(v) for v in [source.transpose(2,0,1),value.baseline[0].cpu().numpy(),value.restored_acescg[0].cpu().numpy(),value.reconstruction_weight[0].cpu().numpy(),target.transpose(2,0,1)]))
        if len(cache)%36==0:print('cached',len(cache),flush=True)
    rows=[];windows=[]
    for window in range(100):
        items=schedule[window*6:(window+1)*6]
        for domain in ('motion','replay'):
            if len({s['scene'] for s in items if s['domain']==domain})!=3:raise ValueError('balanced scene window')
        gradients={name:{'motion':[],'replay':[]} for name in models}
        for offset,item in enumerate(items):
            indices=[lookup[item['domain'],asset] for asset in item['assets']]
            meta=metadata[item['domain'],item['assets'][0]]
            if meta['scene_id']!=item['scene']:raise ValueError('scene identity')
            if item['domain']=='motion':
                rr=[metadata['motion',asset] for asset in item['assets']]
                if len(rr)!=2 or rr[1]['frame']-rr[0]['frame']!=1:raise ValueError('adjacent motion')
            x,base,restored,weight,target=[torch.from_numpy(np.ascontiguousarray(v)).to('cuda') for v in stack_context_arrays(cache,indices,item['crop'])]
            full=(weight>0)&(x.amin(1,keepdim=True)>=254.5/255);valid=full.clone();t,b,l,r=item['borders'];valid[:,:,:t]=False;valid[:,:,-b:]=False;valid[:,:,:,:l]=False;valid[:,:,:,-r:]=False
            if not valid.flatten(1).any(1).all():raise ValueError('empty supervision')
            for name,model in models.items():
                pred,_=model.predict_highlights(x,base,restored,weight);error=(torch.asinh(pred)-torch.asinh(target)).square().mean(1,keepdim=True)
                losses={'photo':((error*valid).flatten(1).sum(1)/valid.flatten(1).sum(1)).mean(),'detail':highlight_gradient_loss(pred,target,valid)}
                if item['domain']=='motion':
                    temporal,count,_=active_temporal_loss(pred,target,(weight>0)&(x.amax(1,keepdim=True)>.5),full)
                    if not count:raise ValueError('empty temporal')
                    losses['temporal']=temporal
                gg={}
                for j,(label,loss) in enumerate(losses.items()):
                    grad=torch.autograd.grad(loss,parameters[name],retain_graph=j<len(losses)-1)
                    vector=torch.cat([v.flatten() for v in grad]).detach().double().cpu()
                    if not torch.isfinite(vector).all():raise ValueError('nonfinite gradient')
                    gg[label]=vector
                total=sum(gg.values());gradients[name][item['domain']].append(gg)
                rows.append({'model':name,'window':window,'offset':offset,'domain':item['domain'],'scene':item['scene'],'asset':item['assets'][0],
                             'render_ev':meta.get('render_ev'),'desired_fraction':meta.get('desired_fully_clipped_fraction'),'valid_pixels':int(valid.sum()),
                             'losses':{k:float(v.detach()) for k,v in losses.items()},'gradient_norms':{k:float(v.norm()) for k,v in gg.items()},'total_gradient_norm':float(total.norm())})
        for name,domains in gradients.items():
            means={domain:{k:sum(v[k] for v in values)/3 for k in values[0]} for domain,values in domains.items()}
            motion=sum(means['motion'].values());replay=sum(means['replay'].values());combined=(motion+replay)/2
            windows.append({'model':name,'window':window,'domain_geometry':geometry(motion,replay),
                            'motion_dot_combined':float(motion@combined),'replay_dot_combined':float(replay@combined),
                            'motion_photo_detail':geometry(means['motion']['photo'],means['motion']['detail']),
                            'replay_photo_detail':geometry(means['replay']['photo'],means['replay']['detail']),
                            'motion_temporal_spatial':geometry(means['motion']['temporal'],means['motion']['photo']+means['motion']['detail'])})
        print('window',window+1,flush=True)
    summary={}
    for name in models:
        ww=[w for w in windows if w['model']==name];rr=[r for r in rows if r['model']==name]
        summary[name]={'windows':len(ww),'domain_conflicts':sum(w['domain_geometry']['cosine']<0 for w in ww if w['domain_geometry']['cosine'] is not None),
                       'median_domain_cosine':float(np.median([w['domain_geometry']['cosine'] for w in ww if w['domain_geometry']['cosine'] is not None])),
                       'combined_opposes_motion':sum(w['motion_dot_combined']<0 for w in ww),'combined_opposes_replay':sum(w['replay_dot_combined']<0 for w in ww),
                       'median_replay_to_motion_gradient_norm':float(np.median([w['domain_geometry']['second_norm']/w['domain_geometry']['first_norm'] for w in ww if w['domain_geometry']['first_norm']>0]))}
        summary[name]['domains']={d:{'mean_'+metric:float(np.mean([r['losses'][metric] for r in rr if r['domain']==d])) for metric in ('photo','detail')} for d in ('motion','replay')}
        if not torch.equal(initial_vectors[name],torch.nn.utils.parameters_to_vector(parameters[name])):raise ValueError('audit modified parameters')
    result={'summary':summary,'windows':windows,'rows':rows,'registration_sha256':sha(a.registration),'tool_sha256':sha(__file__),'parameters_unchanged':True,'scope':reg['scope'],'release_authorized':False}
    (a.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
