"""Apply registered development and matched-control shadow-scale gates."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.compare_aces_development import assess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('control','treatment','mechanics','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new report required')
    c=json.loads(a.control.read_text());t=json.loads(a.treatment.read_text());m=json.loads(a.mechanics.read_text())
    if not m['pass'] or c['checkpoint_sha256']!=m['control_sha256'] or t['checkpoint_sha256']!=m['treatment_sha256']:
        raise ValueError('matched checkpoint mechanics required')
    if c['pilot_sha256']!=t['pilot_sha256'] or c['device']!=t['device']:
        raise ValueError('same evaluation data/device required')
    ca=assess(c);ta=assess(t);checks=ta['checks'].copy()
    baseline=ca['errors']['shadow']['model'];new=ta['errors']['shadow']['model']
    improvement=1-new/baseline if baseline and new is not None else None
    checks['shadow_beats_control_5pct']=improvement is not None and improvement>=.05
    for region in ('all','highlight'):
        checks[region+'_no_control_regression']=ta['errors'][region]['model']<=ca['errors'][region]['model']
    result={'control':ca,'treatment':ta,'matched_shadow_improvement':improvement,'checks':checks,
            'numerical_pass':all(checks.values()),'visual_review':'pending','release_authorized':False}
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
