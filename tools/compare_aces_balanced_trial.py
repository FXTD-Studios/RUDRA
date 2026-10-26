"""Assess the preregistered balanced-restoration trial against fixed controls."""
import argparse,json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.compare_aces_development import assess
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','code-control','ramp-control','registration','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    files={'registration':a.registration,'natural':a.candidate/'development24/report.json',
           'ramps':a.candidate/'grade_ramps/report.json','control_ramps':a.ramp_control/'grade_ramps/report.json'}
    for domain in ('development24','development8'):
        files[domain]=a.candidate/domain/'restoration_codes.json'
        files['control_'+domain]=a.code_control/domain/'restoration_codes.json'
    reports={k:json.loads(v.read_text()) for k,v in files.items()}
    identity=reports['natural']['checkpoint_sha256']
    checks=assess(reports['natural'])['checks']
    for name in ('ramps','development24','development8'):
        if reports[name]['checkpoint_sha256']!=identity:raise ValueError('candidate identity mismatch')
    for domain in ('development24','development8'):
        new=reports[domain];old=reports['control_'+domain]
        if new['pilot_sha256']!=old['pilot_sha256']:raise ValueError('evaluation dataset mismatch')
        x=new['equal_image_mean_code_mse']['model'];y=old['equal_image_mean_code_mse']['model']
        checks[domain+'_code_no_regression']=math.isfinite(x) and math.isfinite(y) and x<=y
    if reports['ramps']['view']!=reports['control_ramps']['view']:raise ValueError('view mismatch')
    current={r['ramp']:r for r in reports['ramps']['rows']}
    control={r['ramp']:r for r in reports['control_ramps']['rows']}
    if set(current)!=set(control):raise ValueError('ramp set mismatch')
    gains={}
    for name in current:
        for ev in ('-3','0','3'):
            x=current[name]['grades'][ev]['methods']['model']['gradient_rmse_code']
            y=control[name]['grades'][ev]['methods']['model']['gradient_rmse_code']
            gain=1-x/y if y>0 else None;key=name+'_'+ev
            gains[key]=gain;checks[key+'_gradient25pct']=gain is not None and math.isfinite(gain) and gain>=.25
    result={'checkpoint_sha256':identity,'checks':checks,'gradient_relative_improvements':gains,
            'numerical_pass':all(checks.values()),'release_authorized':False,
            'scope':'registered development checks only; visual/HDR-detail/temporal/platform/release remain separate',
            'source_reports':{k:{'path':str(v),'sha256':sha(v)} for k,v in files.items()}}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='source_reports'},indent=2))


if __name__=='__main__':main()
