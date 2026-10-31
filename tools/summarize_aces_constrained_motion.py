"""Verify the registered constrained-motion diagnostic and its complete evidence."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.summarize_aces_generated import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('registration','report','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    reg=json.loads(a.registration.read_text());report=json.loads(a.report.read_text())
    manifest=json.loads(Path(reg['manifest']).read_text())
    rows=[r for c in report['clips'] for r in c['frames']]
    transitions=[t for c in report['clips'] for t in c['transitions']]
    values=[t['mean_abs_excess_ev'] for t in transitions]
    complete=len(report['clips'])==len(manifest['clips']) and len(rows)==reg['frames']
    for got,expected in zip(report['clips'],manifest['clips']):
        complete &= got['scene_id']==expected['scene_id'] and got['variant']==expected.get('variant','unspecified')
        complete &= [r['frame'] for r in got['frames']]==[r['frame'] for r in expected['frames']]
        complete &= [r['frame'] for r in got['transitions']]==[r['frame'] for r in expected['frames'][1:]]
    checks={
        'complete':bool(complete),
        'checkpoint_identity':report['checkpoint_sha256']==reg['checkpoint_sha256']==sha(reg['checkpoint']),
        'manifest_identity':report['manifest_sha256']==reg['manifest_sha256']==sha(reg['manifest']),
        'constraint_enabled':report['colour_backtrack'] is True,
        'prediction_hashes':bool(rows) and all(sha(r['prediction_path'])==r['prediction_sha256'] for r in rows),
        'temporal':bool(values) and all(v is not None and math.isfinite(v) and v<=reg['checks']['max_transition_mean_excess_ev'] for v in values),
        'no_new_negative_luminance':all(r['new_negative_luminance_pixels']==0 for r in rows),
        'no_shadow_lift':all(r['shadow_lift_gt1code']==0 for r in rows),
        'appearance':all(r['colour_constraint']['remaining_display_errors']==0 for r in rows),
        'outside_unchanged':all(r['colour_constraint']['outside_support_bit_exact'] for r in rows),
    }
    result={'checks':checks,'numerical_pass':all(checks.values()),'frames':len(rows),
            'max_transition_mean_excess_ev':max(values) if values and all(v is not None for v in values) else None,
            'registration_sha256':sha(a.registration),'report_sha256':sha(a.report),
            'scope':reg['scope'],'release_authorized':False}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
