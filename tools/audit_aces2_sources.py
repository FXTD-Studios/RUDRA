"""Official ACES2 analytic inverse on real saved SDRs, including FLOAT EXR proof.

No learned reconstruction, dequantization or physical exposure recovery claim.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rudra.aces_view import ACES2View
from rudra.delivery.exr import write_exr, read_exr
from rudra.delivery.colorspace import AP1_CHROMATICITIES
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha


def scores(restored, source):
    error = np.abs(restored-source).max(-1)*255
    top = source.max(-1)
    result = {}
    for label, mask in [('all',np.ones(top.shape,bool)),('ordinary',(top>=5/255)&(top<=254/255))]:
        values = error[mask]
        result[label] = {'pixels': int(values.size), 'within1_fraction': float(np.mean(values<=1)) if values.size else None,
                         'max_code_error': float(values.max()) if values.size else None}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise ValueError('new output required')
    torch.set_num_threads(2)
    view=ACES2View()
    rows=[r for r in map(json.loads,args.manifest.read_text().splitlines()) if r['split']=='val' and not r.get('excluded')]
    for row in rows:
        if sha(row['sdr_path'])!=row['sdr_sha256']: raise ValueError('source identity mismatch')
    args.out.mkdir(parents=True)
    results=[]
    for i,row in enumerate(rows):
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False))
        scene=view.inverse(source)
        result={'asset_id':row['asset_id'],'source_sha256':row['sdr_sha256'],
                'roundtrip':scores(view.forward(scene),source),
                'acescg_min':float(scene.min()),'acescg_max':float(scene.max())}
        if i==0:
            metadata={**view.identity,'scene_exposure_recovered':False,
                      'role':'analytic view inverse only; no learned reconstruction or dequantization',
                      'source_sha256':row['sdr_sha256'],'release_authorized':False}
            path=args.out/'analytic_acescg_float32.exr'
            mask=np.zeros(source.shape[:2],np.float32)
            write_exr(path,scene,half=False,chromaticities=AP1_CHROMATICITIES,
                      attributes={'rudra:provenance':json.dumps(metadata)},reconstruction_mask=mask)
            decoded,attrs=read_exr(path)
            np.testing.assert_array_equal(decoded,scene)
            np.testing.assert_array_equal(attrs['reconstruction_mask'],mask)
            result['exr']={'path':str(path),'sha256':sha(path),'bit_exact_float32':True,
                           'decoded_roundtrip':scores(view.forward(decoded),source)}
        results.append(result)
        print(f'[ACES real sources] {i+1}/{len(rows)}',flush=True)
    report={**view.identity,'manifest_sha256':sha(args.manifest),'pairs':len(results),'per_pair':results,
            'all_pairs_ordinary_999':all(r['roundtrip']['ordinary']['within1_fraction'] is not None and r['roundtrip']['ordinary']['within1_fraction']>=.999 for r in results),
            'release_authorized':False,'scope':'analytic CPU view inversion only; no model/target/native integration or grading-headroom pass'}
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__': main()
