"""Export an audited run05 prediction as a research AP1 float32 EXR with a mask.

This preserves the current declared BT1886 research view; it does NOT claim
that an ACES output transform reproduces the SDR, or that scene exposure is known.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.run05_corpus import declared_inverse
from training.contract_checks import roundtrip
from training.sdr2hdr_dataset import load_rgb
from rudra.delivery.aces import write_acescg_exr
from rudra.delivery.exr import read_exr
from rudra.delivery.colorspace import convert


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--asset-id',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise ValueError('new output directory required')
    report=json.loads(args.evaluation.read_text())
    if report['view']!='bt1886' or report['policy']!='observed_clip_v1': raise ValueError('unsupported research view/policy')
    manifest=Path(report['manifest'])
    if sha(manifest)!=report['manifest_sha256'] or sha(report['checkpoint'])!=report['checkpoint_sha256']:
        raise ValueError('evaluation identity changed')
    row=next(r for r in map(json.loads,manifest.read_text().splitlines()) if r['asset_id']==args.asset_id)
    item=next(r for r in report['per_pair'] if r['asset_id']==args.asset_id)
    if sha(row['sdr_path'])!=row['sdr_sha256']: raise ValueError('source identity changed')
    source=load_rgb(row['sdr_path'],hdr=False)
    prediction_path=args.evaluation.parent/item['prediction']
    prediction=np.load(prediction_path)
    if prediction.shape!=source.shape or not np.isfinite(prediction).all() or np.any(prediction<0):
        raise ValueError('invalid evaluated prediction')
    baseline=declared_inverse(source,'bt1886')
    # Include subcode restoration as inferred changes, not only lost-range pixels.
    # Tolerances exclude numerical inverse/interpolation noise, not visible edits.
    tolerance=np.maximum(1e-7,np.abs(baseline)*1e-4)
    mask=np.any(np.abs(prediction-baseline)>tolerance,axis=2).astype(np.float32)
    metadata={'training_use':'research','release_authorized':False,'asset_id':args.asset_id,
        'source_sha256':row['sdr_sha256'],'checkpoint_sha256':report['checkpoint_sha256'],
        'evaluation_sha256':sha(args.evaluation),'prediction_sha256':sha(prediction_path),
        'working_primaries':'ACEScg AP1','storage':'float32','nits_per_unit':203.,
        'declared_research_view':'RUDRA BT1886 picker, not ACES output transform',
        'scene_exposure_recovered':False,
        'mask':'rudra.reconstructed: model changes from analytic inverse, including subcode restoration; not confidence',
        'mask_absolute_tolerance_network_units':1e-7,'mask_relative_tolerance':1e-4,
        'mask_fraction':float(mask.mean())}
    args.out.mkdir(parents=True)
    path=write_acescg_exr(prediction*(10000/203),args.out/'research_master.exr',source_space='rec2020',
        provenance=metadata,half=False,reconstruction_mask=mask)
    decoded,attrs=read_exr(path)
    restored=convert(decoded,'ap1','rec2020')*(203/10000)
    np.testing.assert_allclose(restored,prediction,rtol=2e-5,atol=2e-7)
    np.testing.assert_array_equal(attrs['reconstruction_mask'],mask)
    # The source-derived R mask is independent of the exported change mask.
    metadata['decoded_file_roundtrip']=roundtrip(restored,source,view='bt1886',policy='observed_clip_v1')
    metadata['network_max_abs_file_error']=float(np.max(np.abs(restored-prediction)))
    metadata['exr_sha256']=sha(path)
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__': main()
