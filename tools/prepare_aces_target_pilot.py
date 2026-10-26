"""Build explicit development AP1 subsets with source-scene split checks."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.aces_targets import build_target
from pipeline.run05_corpus import load_plate,render_look,exposure_ev,NETWORK_PER_SCENE
from rudra.aces_view import ACES2View
from rudra.delivery.colorspace import convert
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha
from tools.audit_aces2_sources import scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','sources','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--boundary-method',choices=('scalar','poisson_asinh'),default='scalar')
    p.add_argument('--bounded-shadows',action='store_true')
    p.add_argument('--region-shadows',action='store_true')
    p.add_argument('--direct-shadow-reference',action='store_true')
    p.add_argument('--smooth-shadows',action='store_true')
    p.add_argument('--split',choices=('train','val'),default='train')
    p.add_argument('--per-look',type=int,default=2,help='zero selects all rows in split')
    args=p.parse_args()
    if args.out.exists():raise ValueError('new output required')
    if args.per_look<0:raise ValueError('nonnegative per-look required')
    torch.set_num_threads(2);view=ACES2View()
    origins={r['asset_id']:r for r in map(json.loads,args.sources.read_text().splitlines())}
    rows=list(map(json.loads,args.manifest.read_text().splitlines()))
    train_scenes={r['scene_id'] for r in rows if r['split']=='train'}
    val_scenes={r['scene_id'] for r in rows if r['split']=='val'}
    if train_scenes&val_scenes:raise ValueError('source-scene leakage across splits')
    counts={};results=[];args.out.mkdir(parents=True)
    for row in rows:
        family=row['sdr_curve']
        if row['split']!=args.split or row.get('excluded') or (args.per_look and counts.get(family,0)>=args.per_look):continue
        origin=origins[row['parent_asset_id']]
        if not origin['commercial_ok'] or origin['licence_licence']!='CC0':raise ValueError('CC0 sources only')
        if sha(origin['hdr_path'])!=row['parent_hdr_sha256'] or sha(row['sdr_path'])!=row['sdr_sha256']:raise ValueError('source identity mismatch')
        counts[family]=counts.get(family,0)+1
        plate=load_plate(origin);scene=plate/NETWORK_PER_SCENE
        ev=exposure_ev(scene,row['look'])
        if abs(ev-row['render_ev'])>.0001:raise ValueError('render exposure mismatch')
        clean=render_look(scene,row['look'],ev)
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False))
        reference=convert(scene*(2**ev),'rec2020','ap1')
        pair=build_target(source,clean,reference,view,boundary_method=args.boundary_method,bounded_shadows=args.bounded_shadows,region_shadows=args.region_shadows,direct_shadow_reference=args.direct_shadow_reference,smooth_shadows=args.smooth_shadows)
        target=pair['target_acescg'];baseline=pair['restored_acescg'];weight=pair['weight']
        mask=weight>1e-6
        ratio=1+(target-baseline)/((np.abs(baseline)+.001)*np.maximum(weight[...,None],1e-6))
        reachable=(ratio>=1/16)&(ratio<=16)
        path=args.out/f'{len(results):03d}_target_acescg.npy';np.save(path,target)
        results.append({'asset_id':row['asset_id'],'scene_id':row['scene_id'],'family':family,'split':args.split,'codec':row['sdr_codec'],
                        'sdr_path':row['sdr_path'],'sdr_sha256':row['sdr_sha256'],
                        'target_path':str(path),'target_sha256':sha(path),'fits':pair['fits'],
                        'target_min':float(target.min()),'target_max':float(target.max()),
                        'roundtrip':scores(view.forward(target),source),
                        'reconstruction_pixels':int(mask.sum()),
                        'residual_channels_outside_4stop_fraction':float(np.mean(~reachable[mask])) if mask.any() else 0.,
                        'source_cc0_parent_sha256':row['parent_hdr_sha256']})
        print(f'[ACES target pilot] {len(results)}',flush=True)
    report={'view':view.identity,'boundary_method':args.boundary_method,'bounded_shadows':args.bounded_shadows,'region_shadows':args.region_shadows,'direct_shadow_reference':args.direct_shadow_reference,'smooth_shadows':args.smooth_shadows,'source_manifest_sha256':sha(args.manifest),'sources_sha256':sha(args.sources),
            'rows':results,'release_authorized':False,
            'selection':{'split':args.split,'per_look':args.per_look,'scene_count':len({r['scene_id'] for r in results})},
            'builder_code_sha256':{str(path):sha(path) for path in map(Path,('tools/prepare_aces_target_pilot.py','pipeline/aces_targets.py','pipeline/aces_boundary.py','pipeline/aces_shadow_smooth.py','pipeline/aces_shadow_regions.py'))},
            'scope':'development virtual AP1 targets from original scene-disjoint split; validation is previously used development data, not independent release holdout; relative fits, not physical exposure'}
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
