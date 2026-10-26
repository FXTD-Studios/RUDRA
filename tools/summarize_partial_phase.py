"""Collect both preregistered crop-phase arms; retain every failed quality gate."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.summarize_aces_generated import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group();group.add_argument('--temporal',action='store_true');group.add_argument('--fixed',action='store_true');group.add_argument('--coverage',action='store_true');group.add_argument('--group-coverage',action='store_true');group.add_argument('--pcgrad',action='store_true');args=parser.parse_args()
    if args.group_coverage:args.coverage=True
    coverage_name='group_coverage' if args.group_coverage else 'coverage'
    arms=('temporal1','fixed') if args.fixed else (('temporal0','temporal1') if args.temporal else ('legacy','aligned'))
    roots={arm:Path(f'G:/rudra_training/partial_{arm}_400_20261025' if args.temporal else f'G:/rudra_training/partial_phase_{arm}400_20261025') for arm in arms}
    if args.fixed:roots={'temporal1':Path('G:/rudra_training/partial_temporal1_400_20261025'),'fixed':Path('G:/rudra_training/partial_fixed400_20261025')}
    if args.pcgrad:
        arms=('temporal1','pcgrad')
        roots={'temporal1':Path('G:/rudra_training/partial_temporal1_400_20261025'),'pcgrad':Path('G:/rudra_training/partial_pcgrad400_20261025')}
    if args.coverage:
        arms=('temporal1',coverage_name)
        roots={'temporal1':Path('G:/rudra_training/partial_temporal1_400_20261025'),coverage_name:Path(f'G:/rudra_training/partial_{coverage_name}400_20261025')}
    paired_motion=args.temporal or args.fixed or args.coverage or args.pcgrad
    read=lambda root,name:json.loads((root/name).read_text())
    pools={arm:read(root,'crop_pool.json') for arm,root in roots.items()}
    centres=lambda rows:[(r['asset_id'],r['centre']) for r in rows]
    assert centres(pools[arms[0]])==centres(pools[arms[1]])
    same_schedule=sha(roots[arms[0]]/'schedule.json')==sha(roots[arms[1]]/'schedule.json')
    if args.coverage:
        assert read(roots[arms[0]],'schedule.json')[1::2]==read(roots[arms[1]],'schedule.json')[1::2]
        assert sha(roots[arms[0]]/'crop_pool.json')==sha(roots[arms[1]]/'crop_pool.json')
    else:assert same_schedule
    if paired_motion:assert sha(roots[arms[0]]/'motion_pool.json')==sha(roots[arms[1]]/'motion_pool.json')
    result={'same_centres':True,'same_schedule':same_schedule,'arms':{},'release_authorized':False,
            'scope':'Exploratory development comparison; original quality thresholds. Outside-full diagnostic is retained, full-support preservation checked independently.'}
    for arm,root in roots.items():
        train=read(root,'report.json');paired=read(root,'paired_development/acceptance.json')
        grade=read(root,'old_detail_grade.json');video=read(root,'hunyuan48.json');pres=read(root,'preservation.json')
        result['arms'][arm]={'checkpoint_sha256':train['checkpoint_sha256'],'train_frames':train['train_frames'],
            'pool':train['pool'],'updates':len(train['history']),'paired_checks':paired['checks'],
            'H_error_by_dataset':paired['H_error_by_dataset'],'detail':paired['new_detail_gradient_rmse_ev'],
            'aligned_residual':paired['new_aligned_residual_mse'],'grade':grade['summary'],'grade_checks':grade['checks'],
            'video_checks':video['checks'],'flicker_ev':video['max_transition_mean_ev'],'preservation':pres['checks']}
        if paired_motion:
            result['arms'][arm]['motion_pool']=len(read(root,'motion_pool.json'))
            result['arms'][arm]['motion_updates']=sum(r['domain']=='motion' for r in train['history'])
    if paired_motion:result['same_motion_pool']=True
    if args.group_coverage:
        pool=read(roots[arms[0]],'motion_pool.json')
        a=read(roots[arms[0]],'schedule.json');b=read(roots[arms[1]],'schedule.json')
        assert all([pool[i]['group'] for i in x]==[pool[i]['group'] for i in y] for x,y in zip(a[::2],b[::2]))
        result['same_motion_group_slots']=True
    if args.coverage:
        result['same_still_schedule']=True
        result['coverage']=read(Path(f'G:/rudra_training/partial_{coverage_name}_schedule_20261025'),'report.json')
    out=Path('G:/rudra_training/partial_fixed_comparison_20261025' if args.fixed else ('G:/rudra_training/partial_temporal_comparison_20261025' if args.temporal else 'G:/rudra_training/partial_phase_comparison_20261025'));out.mkdir(exist_ok=True)
    if args.coverage:out=Path(f'G:/rudra_training/partial_{coverage_name}_comparison_20261025');out.mkdir(exist_ok=True)
    if args.pcgrad:
        history=read(roots['pcgrad'],'report.json')['history'];diagnostics={}
        for row in history:
            data=row['pcgrad']
            for name,norm,projected,actual in zip(data['task_names'],data['weighted_gradient_norms'],data['unit_projected_descent_derivatives'],data['actual_step_first_order_changes']):
                record=diagnostics.setdefault(name,{'steps':0,'projected_adverse':0,'actual_step_adverse':0})
                record['steps']+=1;record['projected_adverse']+=int(projected>1e-6*norm);record['actual_step_adverse']+=int(actual>1e-6*norm*data['actual_adam_step_l2'])
        result['pcgrad_diagnostics']=diagnostics
        result['projection_count']=sum(r['pcgrad']['projections'] for r in history)
        out=Path('G:/rudra_training/partial_pcgrad_comparison_20261025');out.mkdir(exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
