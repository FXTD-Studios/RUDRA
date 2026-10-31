"""Apply registered auxiliary-loss trial criteria; never authorizes release."""
import argparse
import json
from pathlib import Path


def compare(control, auxiliary):
    for key in ('manifest_sha256', 'view', 'policy', 'device'):
        if control[key] != auxiliary[key]:
            raise ValueError('comparison identity mismatch: ' + key)
    left = {r['asset_id']: r for r in control['per_pair']}
    right = {r['asset_id']: r for r in auxiliary['per_pair']}
    if len(left) != control['pairs'] or len(right) != auxiliary['pairs'] or left.keys() != right.keys():
        raise ValueError('pair coverage mismatch')
    cl, al = control['label'], auxiliary['label']
    for key, row in left.items():
        other = right[key]
        if any(row[k] != other[k] for k in ('scene_id', 'codec')):
            raise ValueError('stratum mismatch')
        if row['methods']['baseline_bt1886'] != other['methods']['baseline_bt1886']:
            raise ValueError('baseline/reference mismatch')
        c = row['methods'][cl]['reference']['pushes']
        a = other['methods'][al]['reference']['pushes']
        if len(c) != 7 or len(a) != 7:
            raise ValueError('missing exposure pushes')
        for x, y in zip(c, a):
            if x['push'] != y['push'] or x['reference_gentle_pixels'] != y['reference_gentle_pixels']:
                raise ValueError('reference mask mismatch')
            for metric in ('code_rmse', 'gradient_rmse', 'curvature_rmse'):
                if (x[metric] is None) != (y[metric] is None):
                    raise ValueError('measurement coverage mismatch')

    def gradient(report, group):
        pushes = report['summary'][group]['methods'][report['label']]['pushes']
        vals = [pushes[str(p)]['gradient_rmse']['mean'] for p in range(-3, 4)]
        return None if any(v is None for v in vals) else sum(vals) / 7

    c_all, a_all = gradient(control, 'all'), gradient(auxiliary, 'all')
    c_clean, a_clean = gradient(control, 'none'), gradient(auxiliary, 'none')
    con = control['summary']['all']['methods'][cl]['contract']
    aux = auxiliary['summary']['all']['methods'][al]['contract']
    h, ch = aux['h_abs_log2']['mean'], con['h_abs_log2']['mean']
    s, cs = aux['s_visible_lift']['mean'], con['s_visible_lift']['mean']
    checks = {
        'full24': control['pairs'] == auxiliary['pairs'] == 24,
        'gradient_improves_5pct': c_all is not None and a_all is not None and c_all > 0 and a_all <= .95 * c_all,
        'clean_gradient_no_regression': c_clean is not None and a_clean is not None and a_clean <= c_clean,
        'roundtrip_every_pair': all(r['methods'][al]['contract'].get('r_within_1_code', 0) >= .999 for r in right.values()),
        'highlight_guard': h is not None and ch is not None and h <= 1.1 * ch,
        'shadow_guard': s is not None and cs is not None and s <= cs + .01,
    }
    return {'checks': checks, 'quality_trial_pass': all(checks.values()),
            'gradient_mean_over_seven_pushes': {'control': c_all, 'auxiliary': a_all,
                'control_clean': c_clean, 'auxiliary_clean': a_clean},
            'gradient_improvement_fraction': None if c_all is None or c_all <= 0 or a_all is None else 1 - a_all / c_all,
            'aggregation': 'equal-pair means within each exposure, then equal mean of seven exposures; all seven required',
            'release_authorized': False,
            'remaining': 'Verify matched training identities/mechanics; original headroom, temporal, platform, native integration and release gates still apply.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--auxiliary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('refusing to overwrite comparison')
    result = compare(json.loads(args.control.read_text()), json.loads(args.auxiliary.read_text()))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
