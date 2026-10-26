"""Fixed bounded diffusion pilot on training crops, without training weights."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rudra.interval_dequant import bounded_interval_dequant
from training.banding_diagnostics import reference_gradient_errors
from training.contract_checks import canonical_to_source, load_reference, roundtrip
from training.sdr2hdr_dataset import load_rgb
from pipeline.run05_corpus import canonical, declared_inverse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('use a new report')
    rows, counts = [], {}
    for row in map(json.loads, args.manifest.read_text().splitlines()):
        codec = row['sdr_codec']
        if row['split'] != 'train' or row.get('excluded') or counts.get(codec,0) >= 2:
            continue
        counts[codec] = counts.get(codec,0)+1
        source, target = load_rgb(row['sdr_path'],hdr=False), load_reference(row['hdr_path'])
        h,w = source.shape[:2]
        crop = np.s_[(h-256)//2:(h-256)//2+256,(w-256)//2:(w-256)//2+256]
        source, target = source[crop], target[crop]
        filtered = canonical(bounded_interval_dequant(canonical_to_source(source)))
        methods = {}
        for label, value in [('baseline',source),('interval',filtered)]:
            output = declared_inverse(value,'bt1886')
            methods[label] = {'reference':reference_gradient_errors(output,target),
                'roundtrip':roundtrip(output,source,view='bt1886',policy='observed_clip_v1')}
        rows.append({'asset_id':row['asset_id'],'codec':codec,'methods':methods})
        print(f'[interval pilot] {len(rows)}',flush=True)
    args.out.write_text(json.dumps({'scope':'center256 training crops first2 per codec; no model changes; fixed32 iterations .2 step',
        'manifest':str(args.manifest),'rows':rows},indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
