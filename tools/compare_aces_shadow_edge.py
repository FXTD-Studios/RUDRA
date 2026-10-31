"""Registered paired assessment: local shadow branch with image-edge supervision vs the local1200 control."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.compare_aces_development import assess
from tools.compare_aces_shadow_context import shadow_mean
from tools.evaluate_run05_dequant import sha

REPORTS=(('natural','development24/report.json'),('original','original_development24/report.json'),
         ('motion','object_motion384/report.json'),('training','report.json'))
PAIRED_RECIPE_KEYS=('steps','seed','crop','border','lr','wide','base_checkpoint_sha256','training_pilot_sha256')


def candidate_checks(n,o,m,t,baseline):
    if any(x['checkpoint_sha256']!=n['checkpoint_sha256'] for x in (o,m,t)):raise ValueError('candidate identity mismatch')
    if n['pilot_sha256']!=baseline['pilot_sha256'] or o['evaluation_role']!='scene-disjoint development':raise ValueError('development identity mismatch')
    result=assess(n);checks=result['checks']
    checks['shadow_no_coverage_regression']=shadow_mean(n)<=shadow_mean(baseline)
    frames=[f for c in m['clips'] for f in c['frames']]
    checks.update(frozen_predictor_restorer=t['frozen_predictor_and_restorer_bit_exact'],motion384=len(frames)==384,
                  zero_negative_Y=all(f['negative_luminance_pixels']==0 for f in frames),
                  zero_motion_lift=all(f['shadow_lift_gt1code']==0 for f in frames),
                  motion_worst_transition_002=all(c['max_transition_cell_excess_ev'] is not None and c['max_transition_cell_excess_ev']<=.02 for c in m['clips']))
    result['natural_shadow_mse']=shadow_mean(n);result['original_shadow_mse']=shadow_mean(o)
    result['numerical_pass']=all(checks.values());return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','control','baseline','registration','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new report required')
    files={'baseline':a.baseline/'development24/report.json','registration':a.registration}
    for name,root in (('candidate',a.candidate),('control',a.control)):
        for suffix,path in REPORTS:files[name+'_'+suffix]=root/path
    r={k:json.loads(v.read_text()) for k,v in files.items()}
    results={name:candidate_checks(*(r[name+'_'+s] for s,_ in REPORTS),r['baseline']) for name in ('candidate','control')}
    cr=r['candidate_training']['recipe'];kr=r['control_training']['recipe']
    for key in PAIRED_RECIPE_KEYS:
        if cr[key]!=kr[key]:raise ValueError('paired recipe mismatch: '+key)
    if not cr.get('include_image_edges') or kr.get('include_image_edges'):raise ValueError('candidate must be the only edge-supervised run')
    if r['candidate_training']['visits']!=r['control_training']['visits']:raise ValueError('paired sample visits mismatch')
    if r['candidate_original']['pilot_sha256']!=r['control_original']['pilot_sha256'] or r['candidate_motion']['manifest_sha256']!=r['control_motion']['manifest_sha256']:raise ValueError('paired dataset mismatch')
    natural_gain=1-results['candidate']['natural_shadow_mse']/results['control']['natural_shadow_mse']
    paired={'natural_shadow_improves5pct_vs_control':natural_gain>=.05,
            'original_shadow_no_control_regression':results['candidate']['original_shadow_mse']<=results['control']['original_shadow_mse']}
    control={(c['scene_id'],c['variant']):c for c in r['control_motion']['clips']}
    for c in r['candidate_motion']['clips']:
        if c['scene_id']=='original_object_motion_1':
            paired['dark_mean_no_control_regression_'+c['variant']]=c['mean_cell_excess_ev']<=control[(c['scene_id'],c['variant'])]['mean_cell_excess_ev']
    out={'results':results,'paired_checks':paired,'natural_shadow_relative_improvement_vs_control':natural_gain,
         'numerical_pass':results['candidate']['numerical_pass'] and all(paired.values()),'release_authorized':False,
         'source_reports':{k:{'path':str(v),'sha256':sha(v)} for k,v in files.items()},
         'scope':'registered paired shadow development trial; no visual/detail/platform/native/release acceptance'}
    a.out.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in out.items() if k!='source_reports'},indent=2))


if __name__=='__main__':main()
