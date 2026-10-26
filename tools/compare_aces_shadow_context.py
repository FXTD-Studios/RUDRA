"""Registered mixed-data local versus wide shadow development assessment."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.compare_aces_development import assess
from tools.evaluate_run05_dequant import sha


def shadow_mean(report):
    values=[r['target_errors']['model']['shadow_mse'] for r in report['rows'] if r['target_errors']['model']['shadow_mse'] is not None]
    if not values:raise ValueError('shadow observations required')
    return sum(values)/len(values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('local','wide','baseline','registration','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new report required')
    files={'baseline':a.baseline/'development24/report.json','registration':a.registration}
    for name,root in (('local',a.local),('wide',a.wide)):
        for suffix,path in (('natural','development24/report.json'),('original','original_development24/report.json'),('motion','object_motion384/report.json'),('training','report.json')):files[name+'_'+suffix]=root/path
    r={k:json.loads(v.read_text()) for k,v in files.items()};results={}
    for name in ('local','wide'):
        n=r[name+'_natural'];o=r[name+'_original'];m=r[name+'_motion'];t=r[name+'_training']
        if any(x['checkpoint_sha256']!=n['checkpoint_sha256'] for x in (o,m,t)):raise ValueError('candidate identity mismatch')
        if n['pilot_sha256']!=r['baseline']['pilot_sha256'] or o['evaluation_role']!='scene-disjoint development':raise ValueError('development identity mismatch')
        result=assess(n);checks=result['checks'];checks['shadow_no_coverage_regression']=shadow_mean(n)<=shadow_mean(r['baseline'])
        frames=[f for c in m['clips'] for f in c['frames']]
        checks.update(frozen_predictor_restorer=t['frozen_predictor_and_restorer_bit_exact'],motion384=len(frames)==384,
                      zero_negative_Y=all(f['negative_luminance_pixels']==0 for f in frames),zero_motion_lift=all(f['shadow_lift_gt1code']==0 for f in frames),
                      motion_worst_transition_002=all(c['max_transition_cell_excess_ev'] is not None and c['max_transition_cell_excess_ev']<=.02 for c in m['clips']))
        result['original_shadow_mse']=shadow_mean(o);result['numerical_pass']=all(checks.values());results[name]=result
    if r['local_original']['pilot_sha256']!=r['wide_original']['pilot_sha256'] or r['local_motion']['manifest_sha256']!=r['wide_motion']['manifest_sha256']:raise ValueError('paired dataset mismatch')
    lr=r['local_training']['recipe'];wr=r['wide_training']['recipe']
    for key in ('steps','seed','crop','border','lr','base_checkpoint_sha256','training_pilot_sha256'):
        if lr[key]!=wr[key]:raise ValueError('paired recipe mismatch')
    if r['local_training']['visits']!=r['wide_training']['visits']:raise ValueError('paired sample visits mismatch')
    improvement=1-results['wide']['original_shadow_mse']/results['local']['original_shadow_mse']
    paired={'original_shadow_improves10pct':improvement>=.1,'natural_shadow_no_local_regression':shadow_mean(r['wide_natural'])<=shadow_mean(r['local_natural'])}
    control={(c['scene_id'],c['variant']):c for c in r['local_motion']['clips']}
    for c in r['wide_motion']['clips']:
        if c['scene_id']=='original_object_motion_1':paired['dark_mean_no_local_regression_'+c['variant']]=c['mean_cell_excess_ev']<=control[(c['scene_id'],c['variant'])]['mean_cell_excess_ev']
    out={'results':results,'paired_checks':paired,'original_shadow_relative_improvement':improvement,
         'numerical_pass':results['wide']['numerical_pass'] and all(paired.values()),'release_authorized':False,
         'source_reports':{k:{'path':str(v),'sha256':sha(v)} for k,v in files.items()},'scope':'registered shadow development pair; no visual/detail/platform/native/release acceptance'}
    a.out.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in out.items() if k!='source_reports'},indent=2))


if __name__=='__main__':main()
