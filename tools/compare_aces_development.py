"""Fixed development gates for the registered ACES spatial trial."""
import argparse,json,math
from pathlib import Path


def assess(report):
    rows=report['rows']
    if not rows or report.get('evaluation_role')!='scene-disjoint development':
        raise ValueError('scene-disjoint development report required')
    def mean_error(region,method):
        values=[r['target_errors'][method][region+'_mse'] for r in rows
                if r['target_errors'][method][region+'_mse'] is not None]
        if not values or not all(math.isfinite(v) and v>=0 for v in values):return None
        return sum(values)/len(values)
    errors={region:{method:mean_error(region,method) for method in ('analytic','model')}
            for region in ('all','highlight','shadow')}
    baseline=errors['all']['analytic'];model=errors['all']['model']
    improvement=1-model/baseline if baseline and model is not None else None
    checks={'target_improvement_5pct':improvement is not None and improvement>=.05,
            'ordinary_R':all(r['roundtrip']['ordinary']['within1_fraction'] is not None and
                              r['roundtrip']['ordinary']['within1_fraction']>=.999 for r in rows),
            'no_shadow_lift':sum(r['shadow_lift_gt1code'] for r in rows)==0}
    for region in ('highlight','shadow'):
        a=errors[region]['analytic'];b=errors[region]['model']
        checks[region+'_no_regression']=a is not None and b is not None and b<=a
    return {'errors':errors,'relative_improvement':improvement,'checks':checks,
            'numerical_pass':all(checks.values()),'visual_review':'pending','release_authorized':False,
            'scope':'development gates only; numerical pass never implies release or visual pass'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    result=assess(json.loads(a.report.read_text()));a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
