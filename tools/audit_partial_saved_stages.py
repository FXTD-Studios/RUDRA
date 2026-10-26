"""Old DEV partial-highlight stage errors from hash-verified saved predictions."""
import json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_reference import reference_view
from training.aces_model import reconstruction_weight
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def main():
    out=Path('G:/rudra_training/partial_constraint_dev_gap_20261025');out.mkdir(exist_ok=False)
    roots={'ema':Path('G:/rudra_training/partial_temporal1_400_20261025'),'fixed':Path('G:/rudra_training/partial_fixed400_20261025')}
    pilot=Path('G:/datasets/corpora/aces_consistent_exposures_v2_20261024/val/report.json')
    reports={};hashes={str(pilot):sha(pilot)}
    for name,root in roots.items():
        reports[name]={}
        for stage,file in [('raw','candidate_raw_report.json'),('final','candidate_report.json')]:
            p=root/'paired_development/old_coherent'/file;hashes[str(p)]=sha(p)
            reports[name][stage]={r['asset_id']:r for r in json.loads(p.read_text())['rows']}
    (out/'registration.json').write_text(json.dumps({'input_hashes':hashes,'scope':'All24 old DEV images, partial highlight support only; reuse frozen outputs; no retraining, tuning or new holdout claim.'},indent=2))
    (out/'auditor.py').write_bytes(Path(__file__).read_bytes());torch.set_num_threads(2);view=reference_view();view.set_threads(4);rows=[]
    for r in json.loads(pilot.read_text())['rows']:
        for key in ('sdr','target'):
            if sha(r[key+'_path'])!=r[key+'_sha256']:raise ValueError('source identity')
        s=canonical_to_source(load_rgb(r['sdr_path'],hdr=False));target=np.load(r['target_path'])
        w=reconstruction_weight(torch.from_numpy(s).permute(2,0,1)[None])[0,0].numpy();mask=(w>0)&(s.max(-1)>.5)&(s.min(-1)<254.5/255)
        if not mask.any():continue
        truths={ev:view.forward(target[mask]*2.**ev) for ev in (-3,0,3)};states={}
        for name in roots:
            preds={}
            for stage in ('raw','final'):
                row=reports[name][stage][r['asset_id']]
                if sha(row['prediction_path'])!=row['prediction_sha256']:raise ValueError('prediction identity')
                preds[stage]=np.load(row['prediction_path'])[mask]
            preds['analytic']=view.inverse(s[mask]);grades={}
            for stage,pred in preds.items():
                grades[stage]={str(ev):float(np.square((view.forward(pred*2.**ev)-truths[ev]).astype(np.float64)*255).mean()) for ev in truths}
            states[name]={'changed_pixels':int(np.any(preds['raw']!=preds['final'],axis=-1).sum()),'grades':grades}
        rows.append({'asset_id':r['asset_id'],'partial_pixels':int(mask.sum()),'states':states});print('audited',len(rows),flush=True)
    summary={name:{'pixels':sum(r['partial_pixels'] for r in rows),'changed_pixels':sum(r['states'][name]['changed_pixels'] for r in rows),
        'equal_image_grade':{ev:{stage:float(np.mean([r['states'][name]['grades'][stage][ev] for r in rows])) for stage in ('analytic','raw','final')} for ev in ('-3','0','3')}} for name in roots}
    (out/'report.json').write_text(json.dumps({'rows':rows,'summary':summary,'registration_sha256':sha(out/'registration.json'),'release_authorized':False},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
