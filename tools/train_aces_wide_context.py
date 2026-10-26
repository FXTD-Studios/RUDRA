"""Matched local/coarse residual training with cached, frozen whole-frame parent."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint
from training.aces_wide_context_model import ACESWideContextModel
from training.aces_detail_loss import highlight_gradient_loss
from training.aces_active_temporal import active_temporal_loss
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha
from tools.train_aces_restorer import merge_training_ancestry


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();reg=json.loads(a.registration.read_text())
    for path,digest in reg['input_hashes'].items():
        if sha(path)!=digest:raise ValueError('input identity')
    a.out.mkdir(exist_ok=False);(a.out/'registration.json').write_bytes(a.registration.read_bytes())
    dependencies=[__file__,'training/aces_wide_context_model.py','training/aces_model.py','training/aces_full_context_model.py','training/aces_detail_loss.py','training/aces_active_temporal.py']
    for path in dependencies:(a.out/Path(path).name).write_bytes(Path(path).read_bytes())
    torch.set_num_threads(2);torch.manual_seed(reg['seed']);reference_view().set_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=torch.load(reg['initial_checkpoint'],map_location='cpu',weights_only=True);parent=model_from_checkpoint(state).eval().to('cuda');model=ACESWideContextModel.from_parent(parent,reg['context_mode']).eval().to('cuda')
    params=list(model.wide_context.parameters());initial_vector=torch.nn.utils.parameters_to_vector(params).detach().cpu().numpy();initial_hash=hashlib.sha256(initial_vector.tobytes()).hexdigest()
    schedule=json.loads(Path(reg['schedule']).read_text())
    if len(schedule)!=reg['steps']:raise ValueError('schedule count')
    (a.out/'sample_schedule.json').write_bytes(Path(reg['schedule']).read_bytes())
    pilots={domain:{r['asset_id']:r for r in json.loads(Path(path).read_text())['rows']} for domain,path in reg['pilots'].items()}
    ancestry_path=Path(reg['initial_checkpoint']).parent/'training_ancestry.json'
    if state['pilot_sha256']!=sha(ancestry_path):raise ValueError('initial ancestry')
    ancestry={'rows':merge_training_ancestry(json.loads(ancestry_path.read_text())['rows'],[row for rows in pilots.values() for row in rows.values()])}
    ancestry_out=a.out/'training_ancestry.json';ancestry_out.write_text(json.dumps(ancestry,indent=2))
    cache={};geometry={};metadata={}
    for domain,asset in sorted({(s['domain'],asset) for s in schedule for asset in s['assets']}):
        row=pilots[domain][asset]
        if row['split']!='train':raise ValueError('TRAIN only')
        for key in ('sdr','target'):
            if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('content identity')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path']);x=torch.from_numpy(source).permute(2,0,1)[None].to('cuda')
        with torch.no_grad():
            value=parent(x);condition=model.conditioning(x,value.baseline) if reg['context_mode']!='local5' else None
        arrays=[source.transpose(2,0,1),value.baseline[0].cpu().numpy(),value.restored_acescg[0].cpu().numpy(),value.reconstruction_weight[0].cpu().numpy(),value.acescg[0].cpu().numpy(),target.transpose(2,0,1)]
        cache[domain,asset]=([np.ascontiguousarray(v) for v in arrays],condition[0].cpu().numpy() if condition is not None else None)
        geometry[domain,asset]=source.shape[:2];metadata[domain,asset]=row
        if len(cache)%36==0:print('cached',len(cache),flush=True)
    optimizer=torch.optim.Adam(params,lr=reg['learning_rate']);history=[];start=time.perf_counter()
    for step,item in enumerate(schedule,1):
        keys=[(item['domain'],asset) for asset in item['assets']];crop=item['crop']
        if any(metadata[k]['scene_id']!=item['scene'] for k in keys):raise ValueError('scene identity')
        if item['domain']=='motion':
            if len(keys)!=2 or metadata[keys[1]]['frame']-metadata[keys[0]]['frame']!=1:raise ValueError('motion pair')
        def extract(array):
            if crop is None:return array
            y,x,h,w=crop;return array[:,y:y+h,x:x+w]
        tensors=[torch.from_numpy(np.stack([extract(cache[k][0][j]) for k in keys])).to('cuda') for j in range(6)]
        source,base,restored,weight,old,target=tensors
        if reg['context_mode']!='local5':
            conditioning=torch.from_numpy(np.stack([cache[k][1] for k in keys])).to('cuda');fields=model.context_fields(conditioning,geometry[keys[0]],crop)
        else:fields=model.context_fields(model.conditioning(source,base),source.shape[-2:])
        pred=model.apply_context(source,old,restored,weight,fields)
        full=(weight>0)&(source.amin(1,keepdim=True)>=254.5/255);valid=full.clone();t,b,l,r=item['borders'];valid[:,:,:t]=False;valid[:,:,-b:]=False;valid[:,:,:,:l]=False;valid[:,:,:,-r:]=False
        if not valid.flatten(1).any(1).all():raise ValueError('empty full supervision')
        error=(torch.asinh(pred)-torch.asinh(target)).square().mean(1,keepdim=True);photo=((error*valid).flatten(1).sum(1)/valid.flatten(1).sum(1)).mean();detail=highlight_gradient_loss(pred,target,valid)
        temporal=photo*0
        if item['domain']=='motion':
            temporal,count,_=active_temporal_loss(pred,target,(weight>0)&(source.amax(1,keepdim=True)>.5),full)
            if not count:raise ValueError('empty temporal')
        loss=photo+detail+temporal
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise ValueError('nonfinite gradient')
        optimizer.step()
        if step==1 or step%100==0:
            row={'step':step,'loss':float(loss.detach()),'photo':float(photo.detach()),'detail':float(detail.detach()),'temporal':float(temporal.detach()),'seconds':time.perf_counter()-start};history.append(row);print(row,flush=True)
    final={k:v.detach().cpu() for k,v in model.state_dict().items()}
    if any(not torch.equal(final[k],v) for k,v in state['model'].items()):raise ValueError('frozen parent changed')
    if any(not torch.isfinite(v).all() for v in final.values()):raise ValueError('nonfinite weights')
    if any(int(v['step'])!=reg['steps'] for v in optimizer.state.values()):raise ValueError('optimizer count')
    recipe={'registration_sha256':sha(a.registration),'initial_checkpoint_sha256':sha(reg['initial_checkpoint']),'initial_context_vector_sha256':initial_hash,'sample_schedule_sha256':sha(reg['schedule']),'steps':reg['steps'],'optimizer_updates':reg['steps'],'new_trainable_parameters':sum(p.numel() for p in params),'code_hashes':{str(path):sha(path) for path in dependencies}}
    checkpoint=a.out/f'step{reg["steps"]}.pt';torch.save({'model':final,'contract':model.contract(),'pilot_sha256':sha(ancestry_out),'recipe':recipe,'research_only':True},checkpoint)
    (a.out/'report.json').write_text(json.dumps({'recipe':recipe,'history':history,'frozen_parent_bit_exact':True,'checkpoint_sha256':sha(checkpoint),'release_authorized':False},indent=2,allow_nan=False))


if __name__=='__main__':main()
