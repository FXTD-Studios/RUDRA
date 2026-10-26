"""Learned detail against original HDR on common valid support, plus exposure views."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.run05_corpus import load_plate,NETWORK_PER_SCENE
from pipeline.aces_targets import LUMA_AP1
from rudra.delivery.colorspace import convert
from rudra.aces_view import ACES2View
from training.aces_model import reconstruction_weight
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha
from tools.audit_aces_reference_detail import gradient_scores


def common_detail(candidates,reference,mask):
    common=mask&(reference@LUMA_AP1>1e-10)
    for value in candidates.values():common&=value@LUMA_AP1>1e-10
    return {'requested_pixels':int(mask.sum()),'common_pixels':int(common.sum()),
            'excluded_pixels':int((mask&~common).sum()),
            'methods':{name:gradient_scores(value,reference,common) for name,value in candidates.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pilot','evaluation','manifest','sources','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    pilot=json.loads(a.pilot.read_text());evaluation=json.loads(a.evaluation.read_text())
    if pilot['source_manifest_sha256']!=sha(a.manifest) or pilot['sources_sha256']!=sha(a.sources):raise ValueError('provenance mismatch')
    if evaluation['pilot_sha256']!=sha(a.pilot):raise ValueError('evaluation identity mismatch')
    manifest={r['asset_id']:r for r in map(json.loads,a.manifest.read_text().splitlines())}
    origins={r['asset_id']:r for r in map(json.loads,a.sources.read_text().splitlines())}
    predictions={r['asset_id']:r for r in evaluation['rows']}
    if set(predictions)!={r['asset_id'] for r in pilot['rows']}:raise ValueError('prediction set mismatch')
    view=ACES2View();torch.set_num_threads(2);rows=[]
    for row in pilot['rows']:
        parent=manifest[row['asset_id']];origin=origins[parent['parent_asset_id']];prediction=predictions[row['asset_id']]
        if not origin['commercial_ok'] or origin['licence_licence']!='CC0':raise ValueError('CC0 origin required')
        for path,expected in ((origin['hdr_path'],parent['parent_hdr_sha256']),(row['sdr_path'],row['sdr_sha256']),
                              (row['target_path'],row['target_sha256']),(prediction['prediction_path'],prediction['prediction_sha256'])):
            if sha(path)!=expected:raise ValueError('content identity mismatch')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False))
        reference=convert(load_plate(origin)/NETWORK_PER_SCENE,'rec2020','ap1')
        candidates={'analytic':view.inverse(source),'target':np.load(row['target_path']),'model':np.load(prediction['prediction_path'])}
        if any(v.shape!=reference.shape or not np.isfinite(v).all() for v in candidates.values()):raise ValueError('invalid arrays')
        weight=reconstruction_weight(torch.from_numpy(source).permute(2,0,1)[None])[0,0].numpy()
        masks={'highlight':binary_erosion((weight>0)&(source.max(-1)>.5),iterations=2),
               'shadow':binary_erosion((weight>0)&(source.max(-1)<=.5),iterations=2),
               'fully_clipped':binary_erosion(source.min(-1)>=254.5/255,iterations=2)}
        detail={name:common_detail(candidates,reference,mask) for name,mask in masks.items()}
        views={}
        for ev in (-3,0,3):
            target_codes=view.forward(candidates['target']*2.**ev)
            views[str(ev)]={}
            for name in ('analytic','model'):
                codes=view.forward(candidates[name]*2.**ev)
                err=np.mean(((codes-target_codes)*255)**2,axis=-1)
                views[str(ev)][name]={region:float(err[mask].mean()) if mask.any() else None for region,mask in masks.items()}
        rows.append({'asset_id':row['asset_id'],'detail':detail,'view_target_mse_code_squared':views})
        print('detail',len(rows),flush=True)
    summary={}
    for region in masks:
        valid=[r['detail'][region] for r in rows if r['detail'][region]['methods']['model']['edges']>0]
        summary[region]={'images_with_edges':len(valid),'excluded_pixels':sum(r['detail'][region]['excluded_pixels'] for r in rows),
             'mean_gradient_rmse_ev':{name:float(np.mean([r['methods'][name]['rmse_ev'] for r in valid])) if valid else None for name in candidates}}
    a.out.write_text(json.dumps({'checkpoint_sha256':evaluation['checkpoint_sha256'],'pilot_sha256':sha(a.pilot),
         'summary':summary,'rows':rows,'scope':'development diagnostic, common positive-luminance support across all methods, two-pixel eroded interiors, scale-invariant HDR gradients; view errors use experimental target, not physical exposure truth; not release grading acceptance'},indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
