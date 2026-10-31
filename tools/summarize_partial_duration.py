"""Compare completed deterministic duration runs without relaxing any gate."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.summarize_aces_generated import sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-root',type=Path)
    ap.add_argument('--fit-root',type=Path)
    ap.add_argument('--steps',type=int,choices=(1600,6400),default=1600)
    ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    base=Path('G:/rudra_training')
    roots={'control400':base/'partial_pcgrad_deterministic_control400_20261025',
           'duration1600':base/'partial_pcgrad_duration1600_20261025'}
    root=a.candidate_root or base/f'partial_pcgrad_duration{a.steps}_20261025'
    roots[f'duration{a.steps}']=root
    prefix=400 if a.steps==1600 else 1600
    identity=json.loads((root/'duration_identity.json').read_text())
    required={f'history_first{prefix}_exact',f'schedule_first{prefix}_exact',
              'crop_pool_exact','motion_pool_exact',f'updates{a.steps}','frozen_parent_exact'}
    assert required <= identity.keys() and all(identity[k] is True for k in required)
    result={'identity':identity,'arms':{},'release_authorized':False,
            'scope':f'Reused DEV; original thresholds. Exact first{prefix} history. Not fresh holdout.'}
    coverage=root/'duration_coverage_counts.json'
    result['coverage']=json.loads(coverage.read_text()) if coverage.exists() else None
    result['coverage_status']='computed' if coverage.exists() else 'not yet computed'
    for name,r in roots.items():
        read=lambda p:json.loads((r/p).read_text())
        train=read('report.json');p=read('paired_development/acceptance.json')
        steps=int(name.removeprefix('control').removeprefix('duration'))
        assert len(train['history'])==steps
        assert sha(r/f'step{steps}.pt')==train['checkpoint_sha256']
        g=read('old_detail_grade.json');v=read('hunyuan48.json');s=read('preservation.json')
        assert g['evaluation_sha256']==sha(r/'paired_development/old_coherent/candidate_report.json')
        assert g['parent_report_sha256']==sha(r/'paired_development/old_coherent/parent_report.json')
        assert v['checkpoint_sha256']==train['checkpoint_sha256']==s['candidate_sha256']
        result['arms'][name]={'checkpoint_sha256':train['checkpoint_sha256'],
            'paired_checks':p['checks'],'H_error_by_dataset':p['H_error_by_dataset'],
            'detail':p['new_detail_gradient_rmse_ev'],'aligned_residual':p['new_aligned_residual_mse'],
            'grades':g['summary'],'grade_checks':g['checks'],'flicker_ev':v['max_transition_mean_ev'],
            'video_checks':v['checks'],'preservation_checks':s['checks'],'grade_input_hashes_verified':True}
    fit_root=a.fit_root or base/f'partial_duration{a.steps}_training_fit_20261025'
    fit=json.loads((fit_root/'report.json').read_text())
    fit_reg=json.loads((fit_root/'registration.json').read_text())
    assert fit['registration_sha256']==sha(fit_root/'registration.json')
    candidate_sha=result['arms'][f'duration{a.steps}']['checkpoint_sha256']
    assert fit_reg['checkpoints'][str(root/f'step{a.steps}.pt')]==candidate_sha
    assert fit['weights_unchanged'] is True and fit['analytic_calibration_matches'] is True
    assert (fit['unique_crops'],fit['frame_exposures'],fit['pixel_exposures'])==(622,1600,982607)
    result['fixed_train_fit']={k:fit['summary'][k] for k in ('analytic','uncapped','candidate')}
    result['train_fit_scope']='Same622 unique crops from original400 supervision, not full extended-run coverage; raw errors, unlike constrained DEV.'
    result['train_fit_weights_unchanged']=fit['weights_unchanged']
    out=a.out or base/f'partial_duration{a.steps}_comparison_20261025'
    out.mkdir(exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
