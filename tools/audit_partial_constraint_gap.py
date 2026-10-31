"""Fixed TRAIN raw/final error comparison on supervised partial-highlight pixels."""
import json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint
from training.aces_reference import reference_view
from training.aces_clipped_smoothing import smooth_clipped
from training.aces_colour_backtrack import constrain_colour
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def main():
    out=Path('G:/rudra_training/partial_constraint_gap_20261025');out.mkdir(exist_ok=False)
    root=Path('G:/rudra_training/partial_temporal1_400_20261025')
    checkpoints={'initial':'G:/rudra_training/aces_guarded_projected600_20261024/step600.pt',
                 'ema':str(root/'step400.pt'),'fixed':'G:/rudra_training/partial_fixed400_20261025/step400.pt'}
    pilots={'still':'G:/datasets/corpora/aces_consistent_exposures_v2_20261024/train/report.json',
            'motion':'G:/datasets/corpora/aces_temporal_context_pairs_20261024/train/report.json'}
    pools={'still':root/'crop_pool.json','motion':root/'motion_pool.json'}
    reg={'checkpoints':{p:sha(p) for p in checkpoints.values()},'pilots':{p:sha(p) for p in pilots.values()},'pools':{str(p):sha(p) for p in pools.values()},
         'selection':'First saved crop from first four distinct TRAIN scenes per domain; full-frame inference, metrics only partial support in saved64px interior.',
         'scope':'Descriptive TRAIN stage comparison, no gradients, tuning or causal claim. Raw, hybrid64 and appearance-constrained predictions; original thresholds untouched.'}
    (out/'registration.json').write_text(json.dumps(reg,indent=2));(out/'auditor.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2);view=reference_view();view.set_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models={k:model_from_checkpoint(torch.load(p,map_location='cpu',weights_only=True)).eval().cuda() for k,p in checkpoints.items()}
    records=[]
    for domain,pilot in pilots.items():
        rows={r['asset_id']:r for r in json.loads(Path(pilot).read_text())['rows']};seen=set()
        for sample in json.loads(pools[domain].read_text()):
            asset=sample['asset_id'] if domain=='still' else sample['assets'][0];row=rows[asset]
            if row['scene_id'] in seen:continue
            if row['split']!='train':raise ValueError('TRAIN only')
            for key in ('sdr','target'):
                if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('input identity')
            source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
            tensor=torch.from_numpy(source).permute(2,0,1)[None].cuda();y,x=sample['origin']
            states={}
            for name,model in models.items():
                with torch.inference_mode():value=model(tensor)
                array=lambda t:t[0].permute(1,2,0).cpu().numpy()
                raw=array(value.acescg);baseline=array(value.baseline);w=value.reconstruction_weight[0,0].cpu().numpy()
                mask=(w>0)&(source.max(-1)>.5)&(source.min(-1)<254.5/255)
                roi=np.zeros_like(mask);roi[y+64:y+128,x+64:x+128]=True;mask &= roi
                if not mask.any():raise ValueError('empty partial support')
                smooth,_=smooth_clipped(raw,source,sigma=64,mode='hybrid')
                assert np.array_equal(raw[mask],smooth[mask])
                final,diagnostic=constrain_colour(source,smooth,baseline,w,view,include_fully_clipped=True)
                changed=mask&np.any(final!=raw,axis=-1)
                metrics={}
                for stage,pred in [('analytic',baseline),('raw',raw),('final',final)]:
                    metrics[stage]={'photo':float(np.square(np.arcsinh(pred[mask])-np.arcsinh(target[mask])).mean()),'grade':{}}
                    for ev in (-3,0,3):
                        error=(view.forward(pred[mask]*2.**ev)-view.forward(target[mask]*2.**ev))*255
                        metrics[stage]['grade'][str(ev)]=float(np.square(error.astype(np.float64)).mean())
                states[name]={'metrics':metrics,'partial_pixels':int(mask.sum()),'changed_partial_pixels':int(changed.sum()),
                              'raw_max_source_code_error':float((np.abs(view.forward(raw[mask])-source[mask])*255).max()),
                              'final_max_source_code_error':float((np.abs(view.forward(final[mask])-source[mask])*255).max()),'diagnostic':diagnostic}
            records.append({'domain':domain,'asset_id':asset,'scene':row['scene_id'],'states':states});seen.add(row['scene_id']);print('audited',domain,len(seen),flush=True)
            if len(seen)==4:break
        if len(seen)!=4:raise ValueError('coverage')
    summary={}
    for state in models:
        rr=[r['states'][state] for r in records]
        summary[state]={'pixels':sum(r['partial_pixels'] for r in rr),'changed_pixels':sum(r['changed_partial_pixels'] for r in rr),
            'equal_image_photo':{stage:float(np.mean([r['metrics'][stage]['photo'] for r in rr])) for stage in ('analytic','raw','final')},
            'equal_image_grade':{ev:{stage:float(np.mean([r['metrics'][stage]['grade'][ev] for r in rr])) for stage in ('analytic','raw','final')} for ev in ('-3','0','3')}}
    reversals={}
    for state in ('ema','fixed'):
        reversals[state]={}
        for ev in ('-3','0','3'):
            reversals[state][ev]=sum(r['states'][state]['metrics']['raw']['grade'][ev]<r['states']['initial']['metrics']['raw']['grade'][ev] and r['states'][state]['metrics']['final']['grade'][ev]>r['states']['initial']['metrics']['final']['grade'][ev] for r in records)
    result={'summary':summary,'raw_improvement_final_regression_images':reversals,'rows':records,'registration_sha256':sha(out/'registration.json'),
            'code_hashes':{p:sha(p) for p in [__file__,'training/aces_clipped_smoothing.py','training/aces_colour_backtrack.py']},'scope':reg['scope'],'release_authorized':False}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'summary':summary,'reversals':reversals},indent=2))


if __name__=='__main__':main()
