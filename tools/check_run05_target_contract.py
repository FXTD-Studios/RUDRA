"""Measure whether training targets themselves satisfy the source-preservation rule."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.contract_checks import load_reference, roundtrip
from training.sdr2hdr_dataset import load_rgb


def run(manifest, out):
    if out.exists():
        raise ValueError('use a new output file')
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    result = []
    for row in rows:
        result.append({'asset_id': row['asset_id'], 'codec': row['sdr_codec'],
                       **roundtrip(load_reference(row['hdr_path']), load_rgb(row['sdr_path'], False))})
    groups = {}
    for codec in sorted({r['codec'] for r in result}):
        sample = [r for r in result if r['codec'] == codec and r.get('r_pixels', 0)]
        groups[codec] = {'measured_pairs': len(sample),
                         'passing_pairs': sum(r['r_pass'] for r in sample),
                         'mean_within_1_code': sum(r['r_within_1_code'] for r in sample) / len(sample) if sample else None}
    report = {'summary': groups, 'pairs': result}
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.manifest, a.out)
