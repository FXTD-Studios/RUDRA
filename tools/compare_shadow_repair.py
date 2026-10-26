"""Registered recovery gates against both matched and earlier good controls."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.compare_run05_dequant import compare
from tools.evaluate_run05_dequant import sha


def compare_repair(control, treatment, original):
    paired = compare(control, treatment)
    earlier = compare(original, treatment)  # Also validates view, masks and inputs.
    values = earlier['gradient_mean_over_seven_pushes']
    label = treatment['label']
    shadows = [r['methods'][label]['contract'].get('s_visible_lift')
               for r in treatment['per_pair']]
    shadows = [v for v in shadows if v is not None]
    checks = {**paired['checks'],
              'no_regression_vs_original_overall': values['auxiliary'] is not None and values['control'] is not None and values['auxiliary'] <= values['control'],
              'no_regression_vs_original_clean': values['auxiliary_clean'] is not None and values['control_clean'] is not None and values['auxiliary_clean'] <= values['control_clean'],
              'zero_visible_shadow_lift': bool(shadows) and all(v == 0 for v in shadows)}
    return {'checks': checks, 'numerical_trial_pass': all(checks.values()),
            'paired_comparison': paired, 'original_gradient_comparison': values,
            'visual_review_required': True, 'release_authorized': False,
            'remaining': 'Matched checkpoint verification, shadow-colour reports, fixed image review without colour blobs, and original release gates.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('control', 'treatment', 'original', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('new evidence file required')
    paths = [args.control, args.treatment, args.original]
    result = compare_repair(*(json.loads(path.read_text()) for path in paths))
    result['report_sha256'] = {str(path): sha(path) for path in paths}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
