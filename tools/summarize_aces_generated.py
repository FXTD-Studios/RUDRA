"""Registered stability summary of the v1 ACES export on generated (AI) clips.

No HDR ground truth exists for these clips, so this measures stability and visible
defects only (configs/experiments/aces_v1_generated_review_20261024.json), never
reconstruction accuracy.
"""
import argparse,json,hashlib
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

LIMITS={'ordinary_within1_min':.999,'max_shadow_lift_code':1.0,'max_transition_excess_ev':.02}


def clip_checks(report,expected_frames=48):
    c=report['checks']
    checks={'frames_processed':c['frames']==expected_frames and len(report['frames'])==expected_frames,
            'exr_readback_bit_exact':c['all_exr_bit_exact_readback'],
            'ordinary_appearance_within1':c['ordinary_within1_min'] is not None and c['ordinary_within1_min']>=LIMITS['ordinary_within1_min'],
            'no_shadow_lift':c['max_shadow_lift_code'] is None or c['max_shadow_lift_code']<=LIMITS['max_shadow_lift_code'],
            'no_new_negative_luminance':c['new_negative_luminance_pixels']==0,
            'temporal_worst_transition_002':c['max_transition_excess_ev'] is not None and c['max_transition_excess_ev']<=LIMITS['max_transition_excess_ev'],
            'video_written':report.get('video') is not None}
    return checks


def summarize(root,registration=None):
    rows=[]
    registration=Path(registration) if registration else Path(__file__).resolve().parents[1]/'configs/experiments/aces_v1_generated_review_20261024.json'
    registered=json.loads(registration.read_text())
    expected_sha=sha(registered['candidate'])
    expected={x['clip']:x for x in registered['expected_inputs']}
    for report_path in sorted(Path(root).glob('*/report.json')):
        r=json.loads(report_path.read_text());f=r['frames']
        source=expected.get(report_path.parent.name)
        frame_count=source['expected_frames'] if source else 48
        checks=clip_checks(r,frame_count)
        checks['registered_candidate']=r['provenance']['checkpoint_sha256']==expected_sha
        if registered.get('colour_backtrack'):
            checks['constraint_enabled']=r['provenance'].get('colour_backtrack') is True
            checks['partial_highlight_appearance']=all(x.get('colour_constraint') is not None and x['colour_constraint']['remaining_display_errors']==0 for x in f)
            checks['outside_support_unchanged']=all(x.get('colour_constraint') is not None and x['colour_constraint']['outside_support_bit_exact'] for x in f)
        checks['registered_input']=source is not None and Path(r['provenance']['input']).resolve()==Path(source['path']).resolve() and sha(source['path'])==source['sha256']
        artifacts=[(x['exr'],x['sha256']) for x in f if x.get('exr')]
        if r.get('video'):artifacts.append((r['video']['path'],r['video']['sha256']))
        checks['saved_artifacts_verified']=len(artifacts)==(frame_count+11)//12+1 and all(Path(p).is_file() and sha(p)==h for p,h in artifacts)
        rows.append({'clip':report_path.parent.name,'frames':r['checks']['frames'],'checks':checks,'pass':all(checks.values()),
                     'ordinary_within1_min':r['checks']['ordinary_within1_min'],'max_shadow_lift_code':r['checks']['max_shadow_lift_code'],
                     'mean_transition_excess_ev':r['checks']['mean_transition_excess_ev'],'max_transition_excess_ev':r['checks']['max_transition_excess_ev'],
                     'reconstructed_fraction_mean':sum(x['reconstructed_fraction'] for x in f)/len(f),
                     'shadow_support_fraction_mean':sum(x['shadow_support_fraction'] for x in f)/len(f),
                     'contact_sheet':str(report_path.parent/'contact_first_frame.png'),
                     'checkpoint_sha256':r['provenance']['checkpoint_sha256']})
    if not rows:raise ValueError('no clip reports found')
    if len({x['checkpoint_sha256'] for x in rows})!=1:raise ValueError('mixed checkpoints')
    return {'clips':rows,'clips_passing':sum(x['pass'] for x in rows),'clips_total':len(rows),
            'failures':{x['clip']:[k for k,v in x['checks'].items() if not v] for x in rows if not x['pass']},
            'expected_clips':len(expected),'suite_complete':{x['clip'] for x in rows}==set(expected),
            'numerical_pass':{x['clip'] for x in rows}==set(expected) and all(x['pass'] for x in rows),'limits':LIMITS,'release_authorized':False,
            'scope':'generated clips, no HDR ground truth: appearance, shadow, luminance sign and frame-to-frame stability only; frame-capped review; not reconstruction accuracy'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--registration',type=Path);a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    out=summarize(a.root,a.registration);a.out.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:out[k] for k in ('clips_passing','clips_total','failures','numerical_pass')},indent=2))
    for x in out['clips']:print(f"{x['clip']:32s} pass={x['pass']} R={x['ordinary_within1_min']:.5f} lift={x['max_shadow_lift_code']} worstEV={x['max_transition_excess_ev']} recon={x['reconstructed_fraction_mean']:.4f}")


if __name__=='__main__':main()
