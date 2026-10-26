"""Require natural fidelity and procedural/ramp gains together, without promotion."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.compare_aces_development import assess
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--control-ramps',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    files={'natural':a.candidate/'development24/report.json','natural_codes':a.candidate/'development24/restoration_codes.json',
           'procedural':a.candidate/'development8/report.json','procedural_codes':a.candidate/'development8/restoration_codes.json',
           'ramps':a.candidate/'grade_ramps/report.json','control_ramps':a.control_ramps}
    r={name:json.loads(path.read_text()) for name,path in files.items()}
    checkpoint=r['natural']['checkpoint_sha256']
    if any(r[name]['checkpoint_sha256']!=checkpoint for name in r if name!='control_ramps'):raise ValueError('candidate identity mismatch')
    for name in ('natural','procedural'):
        if r[name]['evaluation_role']!='scene-disjoint development' or r[name]['pilot_sha256']!=r[name+'_codes']['pilot_sha256']:raise ValueError('development identity mismatch')
    checks=assess(r['natural'])['checks'].copy()
    checks['natural_code_no_regression']=r['natural_codes']['relative_improvement']>=0
    checks['procedural_code_improvement25pct']=r['procedural_codes']['relative_improvement']>=.25
    c={x['ramp']:x for x in r['control_ramps']['rows']};t={x['ramp']:x for x in r['ramps']['rows']}
    if set(c)!=set(t) or r['control_ramps']['view']!=r['ramps']['view']:raise ValueError('ramp/view identity mismatch')
    changes={}
    for name in c:
        old=c[name]['grades']['0']['methods']['model']['plateau_fraction'];new=t[name]['grades']['0']['methods']['model']['plateau_fraction']
        changes[name]=1-new/old
        checks[name+'_plateau25pct']=changes[name]>=.25
        for ev in ('-3','0','3'):
            checks[name+'_gradient_no_regression_'+ev]=t[name]['grades'][ev]['methods']['model']['gradient_rmse_code']<=c[name]['grades'][ev]['methods']['model']['gradient_rmse_code']
    result={'checkpoint_sha256':checkpoint,'checks':checks,'plateau_relative_reductions':changes,
            'natural_code_improvement':r['natural_codes']['relative_improvement'],
            'procedural_code_improvement':r['procedural_codes']['relative_improvement'],
            'numerical_pass':all(checks.values()),'source_reports':{name:{'path':str(path),'sha256':sha(path)} for name,path in files.items()},
            'scope':'registered restoration development checks only; no visual, temporal, HDR-detail or release acceptance','release_authorized':False}
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_reports'},indent=2))


if __name__=='__main__':main()
