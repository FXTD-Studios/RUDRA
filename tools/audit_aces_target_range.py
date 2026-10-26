"""Check whether saved TRAIN AP1 targets fit proposed residual parameter ranges."""
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
from training.aces_model import compose, CONTRACT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','manifest','sources','out'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError('new report required')
    torch.set_num_threads(2);view=ACES2View()
    pilot=json.loads(args.pilot.read_text())
    if sha(args.manifest)!=pilot['source_manifest_sha256'] or sha(args.sources)!=pilot['sources_sha256']:raise ValueError('identity mismatch')
    rows={r['asset_id']:r for r in map(json.loads,args.manifest.read_text().splitlines())}
    origins={r['asset_id']:r for r in map(json.loads,args.sources.read_text().splitlines())}
    result=[]
    for item in pilot['rows']:
        row=rows[item['asset_id']];plate=load_plate(origins[row['parent_asset_id']]);scene=plate/NETWORK_PER_SCENE
        ev=exposure_ev(scene,row['look']);clean=render_look(scene,row['look'],ev)
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False))
        pair=build_target(source,clean,convert(scene*(2**ev),'rec2020','ap1'),view,
                          boundary_method=pilot.get('boundary_method','scalar'),
                          bounded_shadows=pilot.get('bounded_shadows',False),
                          region_shadows=pilot.get('region_shadows',False),
                          direct_shadow_reference=pilot.get('direct_shadow_reference',False),
                          smooth_shadows=pilot.get('smooth_shadows',False))
        target=np.load(item['target_path']);np.testing.assert_array_equal(target,pair['target_acescg'])
        base=pair['restored_acescg'];scale=(np.abs(base)+.001)*pair['weight'][...,None]
        measured={}
        target_codes=view.forward(target)
        for stops in (4,8,12):
            projected=np.clip(target,base+scale*(2.**(-stops)-1),base+scale*(2.**stops-1))
            error=np.abs(projected-target)
            code=np.abs(view.forward(projected)-target_codes).max(-1)*255
            measured[str(stops)]={'max_ap1_error':float(error.max()),'channels_error_gt1e_5':int((error>1e-5).sum()),
                                  'max_view_code_error':float(code.max()),'pixels_view_error_gt1':int((code>1).sum())}
        # Oracle fields test representability through the actual model compose,
        # not learned quality. Recompute the baseline after finite tanh inversion.
        src=torch.from_numpy(source).permute(2,0,1)[None]
        fields=torch.zeros(1,6,*src.shape[-2:])
        correction=torch.from_numpy(pair['corrected_source']-source).permute(2,0,1)[None]
        fields[:,:3]=torch.atanh((correction/(.49/255)).clamp(-.999999,.999999))
        with torch.no_grad():
            initial=compose(src,fields)
            desired=torch.from_numpy(target).permute(2,0,1)[None]
            denominator=(initial.acescg.abs().amax(1,keepdim=True)+.001)*initial.reconstruction_weight
            fields[:,3:]=torch.where(denominator>0,(desired-initial.acescg)/denominator.clamp_min(1e-30),0)
            recovered=compose(src,fields).acescg[0].permute(1,2,0).numpy()
        exposure={}
        for ev in (-3,0,3):
            error=np.abs(view.forward(recovered*2.**ev)-view.forward(target*2.**ev)).max(-1)*255
            exposure[str(ev)]={'max_code_error':float(error.max()),'pixels_gt1code':int((error>1).sum())}
        oracle={'max_ap1_error':float(np.abs(recovered-target).max()),
                'max_abs_field':float(fields[:,3:].abs().max()),'exposure_errors':exposure}
        result.append({'asset_id':item['asset_id'],'ranges':measured,'signed_linear_oracle':oracle})
    args.out.write_text(json.dumps({'pilot_sha256':sha(args.pilot),'model_contract':CONTRACT,'rows':result,'release_authorized':False},indent=2))
    for stops in ('4','8','12'):
        print(stops,'stops maxAP1',max(r['ranges'][stops]['max_ap1_error'] for r in result),
              'pixels>1code',sum(r['ranges'][stops]['pixels_view_error_gt1'] for r in result))


if __name__=='__main__':main()
