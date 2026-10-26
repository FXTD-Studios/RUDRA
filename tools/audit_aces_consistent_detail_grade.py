"""Registered detail/grade audit against saved, coherently rendered HDR reference."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_erosion
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_aces_learned_detail import common_detail
from tools.summarize_aces_generated import sha
from training.aces_reference import reference_view
from training.aces_model import reconstruction_weight
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_constrained_detail_grade_20261024.json'))
    parser.add_argument('--out',type=Path,default=Path('G:/rudra_training/aces_equal_head_colour_backtrack_v2_20261024/detail_grade.json'))
    args=parser.parse_args();regpath=args.registration
    reg=json.loads(regpath.read_text());pilot=json.loads(Path(reg['pilot']).read_text())
    reports={k:json.loads(Path(reg[k]).read_text()) for k in ('evaluation','parent')}
    for r in reports.values():
        if r['pilot_sha256']!=sha(reg['pilot']):raise ValueError('pilot identity')
    predictions={k:{r['asset_id']:r for r in v['rows']} for k,v in reports.items()}
    for p in predictions.values():
        if set(p)!={r['asset_id'] for r in pilot['rows']}:raise ValueError('incomplete predictions')
    out=args.out
    if out.exists():raise ValueError('new output required')
    torch.set_num_threads(2);view=reference_view();view.set_threads(4);rows=[]
    for row in pilot['rows']:
        if row['licence']!='CC0' or not row['commercial_ok'] or row['split']!='val':raise ValueError('source scope')
        for k in ('sdr','reference','origin'):
            if sha(row[k+'_path'])!=row[k+'_sha256']:raise ValueError('source identity')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));reference=np.load(row['reference_path'])
        candidates={'analytic':view.inverse(source)}
        for k,p in predictions.items():
            pred=p[row['asset_id']]
            if sha(pred['prediction_path'])!=pred['prediction_sha256']:raise ValueError('prediction identity')
            candidates['constrained' if k=='evaluation' else 'parent']=np.load(pred['prediction_path'])
        if any(v.shape!=reference.shape or not np.isfinite(v).all() for v in candidates.values()):raise ValueError('arrays')
        weight=reconstruction_weight(torch.from_numpy(source).permute(2,0,1)[None])[0,0].numpy()
        high=(weight>0)&(source.max(-1)>.5)
        masks={'highlight':binary_erosion(high,iterations=2),'fully_clipped':binary_erosion(source.min(-1)>=254.5/255,iterations=2)}
        detail={k:common_detail(candidates,reference,m) for k,m in masks.items()}
        grades={}
        for ev in (-3,0,3):
            truth=view.forward(reference*2.**ev)
            grades[str(ev)]={k:float(np.mean(((view.forward(v*2.**ev)[high]-truth[high])*255)**2)) if high.any() else None for k,v in candidates.items()}
        rows.append({'asset_id':row['asset_id'],'detail':detail,'highlight_grade_mse':grades});print('audited',len(rows),flush=True)
    summary={};checks={}
    for region in masks:
        valid=[r['detail'][region] for r in rows if r['detail'][region]['methods']['constrained']['edges']>0]
        summary[region]={k:float(np.mean([r['methods'][k]['rmse_ev'] for r in valid])) if valid else None for k in candidates}
        checks[region+'_detail']=bool(valid) and summary[region]['constrained']<=summary[region]['analytic']
    summary['grades']={}
    for ev in ('-3','0','3'):
        valid=[r['highlight_grade_mse'][ev] for r in rows if r['highlight_grade_mse'][ev]['analytic'] is not None]
        summary['grades'][ev]={k:float(np.mean([r[k] for r in valid])) if valid else None for k in candidates}
        checks['grade_'+ev]=bool(valid) and summary['grades'][ev]['constrained']<=summary['grades'][ev]['analytic']
    result={'summary':summary,'checks':checks,'numerical_pass':all(checks.values()),'rows':rows,
            'registration_sha256':sha(regpath),'pilot_sha256':sha(reg['pilot']),
            'evaluation_sha256':sha(reg['evaluation']),'parent_report_sha256':sha(reg['parent']),
            'auditor_sha256':sha(__file__),'scope':reg['scope'],'release_authorized':False}
    out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'summary':summary,'checks':checks},indent=2))


if __name__=='__main__':main()
