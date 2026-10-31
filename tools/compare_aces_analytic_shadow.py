"""Registered acceptance for the v1 analytic-shadow candidate (configs/experiments/aces_analytic_shadow_20261024.json)."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.compare_aces_development import assess
from tools.compare_aces_shadow_context import shadow_mean
from tools.evaluate_run05_dequant import sha
from training.aces_model import reconstruction_weight
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def support_of(source):
    x=torch.from_numpy(np.ascontiguousarray(source)).permute(2,0,1)[None].float()
    return ((reconstruction_weight(x)>0)&(x.amax(1,keepdim=True)<=.5))[0,0].numpy()


def outside_support_identical(source,candidate,parent):
    support=support_of(source)
    return bool(np.array_equal(candidate[~support],parent[~support])),int(support.sum())


def motion_checks(motion):
    frames=[f for c in motion['clips'] for f in c['frames']]
    return {'motion384':len(frames)==384,
            'zero_negative_Y':all(f['negative_luminance_pixels']==0 for f in frames),
            'zero_motion_lift':all(f['shadow_lift_gt1code']==0 for f in frames),
            'motion_worst_transition_002':all(c['max_transition_cell_excess_ev'] is not None and c['max_transition_cell_excess_ev']<=.02 for c in motion['clips'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','parent','pilot','registration','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new report required')
    files={'natural':a.candidate/'development24/report.json','original':a.candidate/'original_development24/report.json',
           'motion':a.candidate/'object_motion384/report.json','parent_natural':a.parent/'development24/report.json',
           'parent_motion':a.parent/'object_motion384/report.json','registration':a.registration,'pilot':a.pilot}
    r={k:json.loads(v.read_text()) for k,v in files.items()}
    n,o,m,pn=r['natural'],r['original'],r['motion'],r['parent_natural']
    if not (n['checkpoint_sha256']==o['checkpoint_sha256']==m['checkpoint_sha256']):raise ValueError('candidate identity mismatch')
    if n['pilot_sha256']!=pn['pilot_sha256'] or n['pilot_sha256']!=sha(a.pilot):raise ValueError('development identity mismatch')
    if n['contract'].get('shadow')!='analytic':raise ValueError('candidate must be analytic-shadow')
    rows={row['asset_id']:row for row in r['pilot']['rows']};parent_rows={x['asset_id']:x for x in pn['rows']};isolation=[]
    for x in n['rows']:
        src=rows[x['asset_id']];y=parent_rows[x['asset_id']]
        for path,h in ((src['sdr_path'],src['sdr_sha256']),(x['prediction_path'],x['prediction_sha256']),(y['prediction_path'],y['prediction_sha256'])):
            if sha(path)!=h:raise ValueError('hash mismatch '+str(path))
        same,pixels=outside_support_identical(canonical_to_source(load_rgb(src['sdr_path'],hdr=False)),np.load(x['prediction_path']),np.load(y['prediction_path']))
        isolation.append({'asset_id':x['asset_id'],'shadow_support_pixels':pixels,'outside_support_bit_exact':same})
    result=assess(n);checks=result['checks']
    checks['outside_shadow_bit_exact_to_parent']=all(i['outside_support_bit_exact'] for i in isolation)
    checks['original_dev_zero_shadow_lift']=sum(x['shadow_lift_gt1code'] for x in o['rows'])==0
    # Synthetic shadow scenes can contain no ordinary pixels at all (pixels==0, within1 None): nothing to measure there.
    measured=[x for x in o['rows'] if x['roundtrip']['ordinary']['pixels']>0]
    if not measured:raise ValueError('original-dev has no ordinary pixels to check')
    checks['original_dev_ordinary_R']=all(x['roundtrip']['ordinary']['within1_fraction']>=.999 for x in measured)
    checks.update(motion_checks(m))
    parent_motion=motion_checks(r['parent_motion'])
    out={'checks':checks,'numerical_pass':all(checks.values()),'errors':result['errors'],'relative_improvement':result['relative_improvement'],
         'disclosed_costs':{'natural_shadow_mse':{'candidate':shadow_mean(n),'parent_coverage4800':shadow_mean(pn),
                                                  'relative_change':shadow_mean(n)/shadow_mean(pn)-1,
                                                  'note':'single parent seed; 5-seed study measured +-4% seed spread for trained shadow branches'},
                            'original_dev_shadow_mse':shadow_mean(o),
                            'original_dev_rows_without_ordinary_pixels':len(o['rows'])-len(measured),
                            'original_dev_all_pixel_within1_min':min(x['roundtrip']['all']['within1_fraction'] for x in o['rows'])},
         'parent_motion_checks':parent_motion,'isolation':isolation,'release_authorized':False,
         'source_reports':{k:{'path':str(v),'sha256':sha(v)} for k,v in files.items()},
         'scope':'registered v1 shadow-policy development acceptance; visual, highlight detail, platform, native and release acceptance separate'}
    a.out.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps({k:out[k] for k in ('checks','numerical_pass','disclosed_costs','parent_motion_checks')},indent=2))


if __name__=='__main__':main()
