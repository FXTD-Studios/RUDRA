"""Matched deterministic capped/uncapped PCGrad results, retaining all failures."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--direct',action='store_true');args=parser.parse_args()
    roots={'control':Path('G:/rudra_training/partial_pcgrad_deterministic_control400_20261025'),
           'cap':Path('G:/rudra_training/partial_pcgrad_cap400_20261025')}
    if args.direct:roots={'control':Path('G:/rudra_training/partial_pcgrad_cap400_20261025'),'cap':Path('G:/rudra_training/partial_pcgrad_direct400_20261025')}
    read=lambda root,name:json.loads((root/name).read_text())
    for name in ('crop_pool.json','motion_pool.json','schedule.json'):
        assert (roots['control']/name).read_bytes()==(roots['cap']/name).read_bytes()
    trains={arm:read(root,'report.json') for arm,root in roots.items()}
    assert [r['analytic_ema'] for r in trains['control']['history']]==[r['analytic_ema'] for r in trains['cap']['history']]
    caps=[r['pcgrad']['direct_step' if args.direct else 'update_cap'] for r in trains['cap']['history']]
    assert len(caps)==400 and all(c['applied_l2']<=1e-4 for c in caps)
    summary={'same_pools_schedule_ema':True,'update_cap_verified':True,'arms':{},
             'cap_statistics':{k:{'min':min(c[k] for c in caps),'median':float(np.median([c[k] for c in caps])),
                                   'max':max(c[k] for c in caps)} for k in (('requested_l2','applied_l2') if args.direct else ('proposed_l2','applied_l2','scale','rounding_adjustments'))},
             'scope':'Reused development data; original limits. No holdout or release claim.'}
    if args.direct:
        requested=[r['pcgrad']['update_cap']['applied_l2'] for r in trains['control']['history']]
        assert requested==[c['requested_l2'] for c in caps]
        assert all(c['applied_l2']<=c['requested_l2'] for c in caps)
        summary['direct_movement']={'requested_schedule_exact':True,'zero_directions':sum(c['zero_direction'] for c in caps),
                                    'total_requested':sum(requested),'total_applied':sum(c['applied_l2'] for c in caps),
                                    'min_applied_requested_ratio':min(c['applied_l2']/c['requested_l2'] for c in caps)}
        summary['arm_labels']={'control':'capped_adam','cap':'direct_projected'}
    for arm,root in roots.items():
        p=read(root,'paired_development/acceptance.json');g=read(root,'old_detail_grade.json');v=read(root,'hunyuan48.json');s=read(root,'preservation.json')
        summary['arms'][arm]={'checkpoint_sha256':trains[arm]['checkpoint_sha256'],'paired_checks':p['checks'],
                              'H_error_by_dataset':p['H_error_by_dataset'],'detail':p['new_detail_gradient_rmse_ev'],
                              'aligned_residual':p['new_aligned_residual_mse'],'grades':g['summary'],'grade_checks':g['checks'],
                              'flicker_ev':v['max_transition_mean_ev'],'video_checks':v['checks'],'preservation_checks':s['checks']}
    out=Path('G:/rudra_training/partial_pcgrad_direct_comparison_20261025' if args.direct else 'G:/rudra_training/partial_pcgrad_cap_comparison_20261025');out.mkdir(exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
