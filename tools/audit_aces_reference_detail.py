"""TRAIN target detail against source HDR: scale-invariant log-luminance gradients.

Separate highlight/shadow interiors from transitions. This diagnoses recovery;
it is not an absolute exposure truth or a release gate.
"""
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


def gradient_scores(candidate,reference,mask):
    cy=candidate@LUMA_AP1;ry=reference@LUMA_AP1
    valid=mask&(cy>1e-10)&(ry>1e-10)
    c=np.log2(np.maximum(cy,1e-10));r=np.log2(np.maximum(ry,1e-10))
    cg=[];rg=[]
    for axis in (0,1):
        sl=[slice(None)]*2;sr=sl.copy();sl[axis]=slice(None,-1);sr[axis]=slice(1,None)
        edges=valid[tuple(sl)]&valid[tuple(sr)]
        cg.append(np.diff(c,axis=axis)[edges]);rg.append(np.diff(r,axis=axis)[edges])
    c=np.concatenate(cg).astype(float);r=np.concatenate(rg).astype(float)
    result={'pixels':int(mask.sum()),'valid_pixels':int(valid.sum()),'edges':len(c)}
    if not len(c):return result
    reference_energy=float(np.mean(r*r));candidate_energy=float(np.mean(c*c))
    result.update(rmse_ev=float(np.sqrt(np.mean((c-r)**2))),reference_rms_ev=reference_energy**.5,
                  candidate_rms_ev=candidate_energy**.5,
                  gradient_gain=float(np.dot(c,r)/np.dot(r,r)) if np.dot(r,r)>1e-16 else None,
                  cosine=float(np.dot(c,r)/np.sqrt(np.dot(c,c)*np.dot(r,r)))
                  if np.dot(c,c)*np.dot(r,r)>1e-16 else None)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old','new','manifest','sources','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new report required')
    old=json.loads(a.old.read_text());new=json.loads(a.new.read_text())
    for pilot in (old,new):
        assert pilot['source_manifest_sha256']==sha(a.manifest)
        assert pilot['sources_sha256']==sha(a.sources)
    rows={r['asset_id']:r for r in map(json.loads,a.manifest.read_text().splitlines())}
    origins={r['asset_id']:r for r in map(json.loads,a.sources.read_text().splitlines())}
    view=ACES2View();torch.set_num_threads(2);output=[]
    assert len(old['rows'])==len(new['rows'])
    for o,n in zip(old['rows'],new['rows']):
        assert o['asset_id']==n['asset_id']
        row=rows[o['asset_id']];origin=origins[row['parent_asset_id']]
        assert row['split']=='train' and origin['commercial_ok'] and origin['licence_licence']=='CC0'
        assert sha(origin['hdr_path'])==row['parent_hdr_sha256']
        assert sha(o['sdr_path'])==o['sdr_sha256']==n['sdr_sha256']
        for item in (o,n):assert sha(item['target_path'])==item['target_sha256']
        source=canonical_to_source(load_rgb(o['sdr_path'],hdr=False))
        ref=convert(load_plate(origin)/NETWORK_PER_SCENE,'rec2020','ap1')
        candidates={'analytic':view.inverse(source),'scalar':np.load(o['target_path']),'boundary':np.load(n['target_path'])}
        weight=reconstruction_weight(torch.from_numpy(source).permute(2,0,1)[None])[0,0].numpy()
        top=source.max(-1);support=weight>0
        masks={'highlight_interior':binary_erosion(support&(top>.5),iterations=2),
               'shadow_interior':binary_erosion(support&(top<=.5),iterations=2),
               'fully_clipped_interior':binary_erosion(source.min(-1)>=254.5/255,iterations=2)}
        metrics={region:{name:gradient_scores(value,ref,mask) for name,value in candidates.items()}
                 for region,mask in masks.items()}
        shadow=support&(top<=.5)
        shadow_view={}
        for name,value in candidates.items():
            error=(view.forward(value)-source).max(-1)*255
            shadow_view[name]={'pixels':int(shadow.sum()),'lift_gt1code':int(((error>1)&shadow).sum()),
                               'max_lift_code':float(error[shadow].max()) if shadow.any() else None}
        output.append({'asset_id':o['asset_id'],'metrics':metrics,'shadow_view':shadow_view})
        print('[detail]',len(output),flush=True)
    a.out.write_text(json.dumps({'old_sha256':sha(a.old),'new_sha256':sha(a.new),'rows':output,
        'scope':'TRAIN interior log2 luminance gradient diagnostic; scale invariant, no physical exposure claim'},indent=2,allow_nan=False))


if __name__=='__main__':main()
