"""Evaluate registered colour backtracking using saved development predictions."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_colour_backtrack import constrain_colour
from training.aces_model import reconstruction_weight
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha
from tools.audit_aces2_sources import scores
from tools.compare_aces_development import assess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_colour_backtrack_20261024.json'))
    parser.add_argument('--out',type=Path,default=Path('G:/rudra_training/aces_v1_colour_backtrack_20261024'))
    parser.add_argument('--parent-report',type=Path,default=Path('G:/rudra_training/aces_v1_analytic_shadow_20261024/development24/report.json'))
    parser.add_argument('--pilot',type=Path,default=Path('G:/datasets/corpora/run05_aces_spatial_dev24_v2_20261023/report.json'));args=parser.parse_args()
    torch.set_num_threads(2)
    root=args.out;root.mkdir(exist_ok=False)
    reg=args.registration
    parent_path=args.parent_report
    parent=json.loads(parent_path.read_text())
    pilot_path=args.pilot
    pilot=json.loads(pilot_path.read_text())
    assert parent['pilot_sha256']==sha(pilot_path)
    registration=json.loads(reg.read_text())
    assert parent['checkpoint_sha256']==sha(registration['parent'])
    constraint=constrain_colour
    if registration.get('include_fully_clipped'):
        from functools import partial
        constraint=partial(constrain_colour,include_fully_clipped=True)
    if registration.get('fully_clipped_analytic'):
        from training.aces_reliable_highlights import partial_only
        constraint=partial_only
    if registration.get('fully_clipped_feather_radius'):
        from training.aces_reliable_highlights import feather_full
        from functools import partial
        constraint=partial(feather_full,radius=registration['fully_clipped_feather_radius'])
    predictions={r['asset_id']:r for r in parent['rows']};view=reference_view();view.set_threads(4);rows=[]
    for i,r in enumerate(pilot['rows']):
        old=predictions[r['asset_id']]
        for p,h in ((r['sdr_path'],r['sdr_sha256']),(r['target_path'],r['target_sha256']),(old['prediction_path'],old['prediction_sha256'])):
            assert sha(p)==h
        source=canonical_to_source(load_rgb(r['sdr_path'],hdr=False));baseline=view.inverse(source)
        prediction=np.load(old['prediction_path']);target=np.load(r['target_path'])
        w=reconstruction_weight(torch.from_numpy(source).permute(2,0,1)[None])[0,0].numpy()
        result,diagnostic=constraint(source,prediction,baseline,w,view)
        loss={};shadow=(w>0)&(source.max(-1)<=.5);highlight=(w>0)&~shadow
        for name,array in (('analytic',baseline),('model',result)):
            error=np.mean((np.arcsinh(array)-np.arcsinh(target))**2,-1)
            loss[name]={'all_mse':float(error.mean()),'shadow_mse':float(error[shadow].mean()) if shadow.any() else None,
                        'highlight_mse':float(error[highlight].mean()) if highlight.any() else None}
        from tools.evaluate_aces_temporal import LUMA_AP1
        diagnostic['new_negative_luminance']=int((((result@LUMA_AP1)<0)&((baseline@LUMA_AP1)>=0)).sum())
        path=root/f'{i:03d}_prediction.npy';np.save(path,result)
        codes=view.forward(result)
        rows.append({'asset_id':r['asset_id'],'prediction_path':str(path),'prediction_sha256':sha(path),
                     'roundtrip':scores(codes,source),'target_errors':loss,
                     'shadow_lift_gt1code':int(((codes-source).max(-1)[shadow]*255>1).sum()),'diagnostic':diagnostic})
        print(i+1,diagnostic,flush=True)
    report={'rows':rows,'evaluation_role':parent['evaluation_role'],'parent_report_sha256':sha(parent_path),'registration_sha256':sha(reg),
            'pilot_sha256':sha(pilot_path),'parent_checkpoint_sha256':parent['checkpoint_sha256'],
            'evaluator_sha256':sha(__file__),'constraint_sha256':sha('training/aces_colour_backtrack.py'),
            'fully_clipped_analytic':bool(registration.get('fully_clipped_analytic')),
            'include_fully_clipped':bool(registration.get('include_fully_clipped')),
            'fully_clipped_feather_radius':registration.get('fully_clipped_feather_radius'),
            'reliability_policy_sha256':sha('training/aces_reliable_highlights.py') if registration.get('fully_clipped_analytic') or registration.get('fully_clipped_feather_radius') else None}
    acceptance=assess(report);parent_score=assess(parent)
    acceptance['checks'].update(
        highlight_no_regression_vs_parent=acceptance['errors']['highlight']['model']<=parent_score['errors']['highlight']['model'],
        partial_highlight_appearance=all(r['diagnostic']['remaining_display_errors']==0 for r in rows),
        outside_support_unchanged=all(r['diagnostic']['outside_support_bit_exact'] for r in rows),
        no_new_negative_luminance=all(r['diagnostic']['new_negative_luminance']==0 for r in rows))
    acceptance['numerical_pass']=all(acceptance['checks'].values())
    (root/'report.json').write_text(json.dumps(report,indent=2));(root/'acceptance.json').write_text(json.dumps(acceptance,indent=2))
    print(json.dumps(acceptance,indent=2))


if __name__=='__main__':main()
