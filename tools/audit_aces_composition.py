"""TRAIN-only stress of explicit AP1 composition, not trained-model quality."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import compose,CONTRACT
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise ValueError('new output required')
    torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(20261023)
    counts={};results=[]
    for row in map(json.loads,args.manifest.read_text().splitlines()):
        codec=row['sdr_codec']
        if row['split']!='train' or row.get('excluded') or counts.get(codec,0)>=2: continue
        counts[codec]=counts.get(codec,0)+1
        if sha(row['sdr_path'])!=row['sdr_sha256']: raise ValueError('source hash mismatch')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False))
        tensor=torch.from_numpy(source.copy()).permute(2,0,1)[None]
        fields=torch.randn((1,6,*source.shape[:2]),generator=generator)*3
        with torch.inference_mode(): output=compose(tensor,fields)
        decoded=reference_view().forward(output.acescg[0].permute(1,2,0).numpy())
        top=source.max(-1);ordinary=(top>=5/255)&(top<=254/255)
        error=np.abs(decoded-source).max(-1)*255
        values=error[ordinary]
        results.append({'asset_id':row['asset_id'],'codec':codec,'pixels':int(ordinary.sum()),
                        'ordinary_within1_fraction':float(np.mean(values<=1)) if values.size else None,
                        'ordinary_max_error_codes':float(values.max()) if values.size else None,
                        'max_source_correction_codes':float((output.restored_source-tensor).abs().max()*255),
                        'finite_acescg':bool(torch.isfinite(output.acescg).all())})
        print(f'[ACES composition] {len(results)}',flush=True)
    report={'contract':CONTRACT,'view':reference_view().identity,'manifest_sha256':sha(args.manifest),
            'scope':'TRAIN first2 per codec, full-resolution random-field stress; not trained quality or headroom',
            'rows':results,'release_authorized':False}
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
