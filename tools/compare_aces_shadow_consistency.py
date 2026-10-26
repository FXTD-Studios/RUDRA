"""Registered paired shadow fine-tuning assessment; no release promotion."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.compare_aces_development import assess
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('control','treatment','baseline','registration','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    files={'baseline':a.baseline/'development24/report.json','registration':a.registration}
    for name,root in (('control',a.control),('treatment',a.treatment)):
        for suffix,path in (('natural','development24/report.json'),('motion','object_motion384/report.json'),('training','report.json')):
            files[name+'_'+suffix]=root/path
    r={k:json.loads(v.read_text()) for k,v in files.items()};results={}
    for name in ('control','treatment'):
        n=r[name+'_natural'];m=r[name+'_motion'];t=r[name+'_training']
        if n['checkpoint_sha256']!=m['checkpoint_sha256'] or n['checkpoint_sha256']!=t['checkpoint_sha256']:raise ValueError('checkpoint identity mismatch')
        if n['pilot_sha256']!=r['baseline']['pilot_sha256']:raise ValueError('natural dataset mismatch')
        result=assess(n);checks=result['checks'];checks['shadow_no_coverage_regression']=result['errors']['shadow']['model']<=assess(r['baseline'])['errors']['shadow']['model']
        checks['frozen_predictor_restorer']=t['frozen_predictor_and_restorer_bit_exact']
        frames=[f for c in m['clips'] for f in c['frames']]
        checks['motion_all384']=len(frames)==384
        checks['motion_zero_negative_luma']=all(f['negative_luminance_pixels']==0 for f in frames)
        checks['motion_zero_shadow_lift']=all(f['shadow_lift_gt1code']==0 for f in frames)
        checks['motion_worst_transition_002']=all(c['max_transition_cell_excess_ev'] is not None and c['max_transition_cell_excess_ev']<=.02 for c in m['clips'])
        result['numerical_pass']=all(checks.values());results[name]=result
    if r['control_motion']['manifest_sha256']!=r['treatment_motion']['manifest_sha256']:raise ValueError('motion manifest mismatch')
    cr=r['control_training']['recipe'];tr=r['treatment_training']['recipe']
    for key in ('shadow_initialization_sha256','seed','steps','lr','training_pilot_sha256','base_checkpoint_sha256'):
        if cr[key]!=tr[key]:raise ValueError('paired recipe mismatch: '+key)
    if r['control_training']['visits']!=r['treatment_training']['visits']:raise ValueError('sample visit mismatch')
    checks={'natural_shadow_no_control_regression':results['treatment']['errors']['shadow']['model']<=results['control']['errors']['shadow']['model']}
    old={(c['scene_id'],c['variant']):c for c in r['control_motion']['clips']}
    for c in r['treatment_motion']['clips']:
        key=(c['scene_id'],c['variant'])
        if c['scene_id']=='original_object_motion_1':checks['dark_mean_no_control_regression_'+c['variant']]=c['mean_cell_excess_ev']<=old[key]['mean_cell_excess_ev']
    out={'results':results,'paired_checks':checks,'numerical_pass':results['treatment']['numerical_pass'] and all(checks.values()),
         'source_reports':{k:{'path':str(v),'sha256':sha(v)} for k,v in files.items()},'release_authorized':False,
         'scope':'registered paired shadow development checks; no visual/detail/generated-platform/native/release claim'}
    a.out.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in out.items() if k!='source_reports'},indent=2))


if __name__=='__main__':main()
