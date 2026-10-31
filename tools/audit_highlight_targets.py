"""Find target-induced appearance changes in bright but unclipped SDR pixels."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.contract_checks import canonical_to_source, view_forward, load_reference
from training.sdr2hdr_dataset import load_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', default='val')
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('use a new audit path')
    results = []
    for row in map(json.loads, args.manifest.read_text().splitlines()):
        if row.get('split') != args.split:
            continue
        src = canonical_to_source(load_rgb(row['sdr_path'], hdr=False)) * 255
        target = canonical_to_source(view_forward(load_reference(row['hdr_path']), row['declared_view'])) * 255
        bright = (src.max(2) >= .94 * 255) & (src.max(2) < 254.5)
        errors = np.abs(target - src).max(2)
        white = bright & (target.max(2) >= 254.5)
        results.append(dict(asset_id=row['asset_id'], split=row['split'], codec=row.get('sdr_codec'),
            bright_unclipped_pixels=int(bright.sum()), newly_clipped_pixels=int(white.sum()),
            changed_above_one_code=int((bright & (errors > 1)).sum()),
            max_error_codes=float(errors[bright].max()) if bright.any() else None))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(criteria='input max in [0.94*255,254.5); target max>=254.5 is newly clipped',
        rows=results, totals={k:sum(r[k] for r in results) for k in
            ('bright_unclipped_pixels','newly_clipped_pixels','changed_above_one_code')}), indent=2, allow_nan=False))
    print(args.out, flush=True)


if __name__ == '__main__':
    main()
