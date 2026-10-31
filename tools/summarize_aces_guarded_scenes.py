"""Describe fixed guarded600 DEV failures; no tuning or acceptance changes."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('new output required')
    hashes = {}

    def read(path):
        path = Path(path)
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    pilot_path = Path('G:/datasets/corpora/aces_temporal_context_pairs_20261024/val/report.json')
    pilot = read(pilot_path)
    metadata = {r['asset_id']: r for r in pilot['rows']}
    runs = {}
    for name in ('adam', 'projected'):
        root = Path(f'G:/rudra_training/aces_guarded_{name}600_20261024')
        acceptance = read(root/'paired_development/acceptance.json')
        reports = {method: read(root/f'paired_development/new_motion/{method}_report.json')
                   for method in ('parent', 'candidate_raw', 'candidate')}
        for report in reports.values():
            if report['pilot_sha256'] != hashes[str(pilot_path)]:
                raise ValueError('pilot identity')
            if {r['asset_id'] for r in report['rows']} != set(metadata):
                raise ValueError('incomplete DEV')
        lookup = {method: {r['asset_id']: r for r in report['rows']}
                  for method, report in reports.items()}
        rows = []
        for detail in acceptance['detail_rows']:
            asset = detail['asset_id']
            meta = metadata[asset]
            values = detail['fully_clipped']['methods']
            row = {'asset_id': asset, 'scene': meta['scene_id'],
                   'clipping_fraction': meta.get('desired_fully_clipped_fraction'),
                   'full_detail_rmse_ev': {k: v.get('rmse_ev') for k, v in values.items()},
                   'highlight_mse': {k: v[asset]['target_errors']['model']['highlight_mse']
                                     for k, v in lookup.items()}}
            rows.append(row)
        if len(rows) != len(metadata):
            raise ValueError('detail coverage')

        def aggregate(key):
            result = []
            for group in sorted({str(r[key]) for r in rows}):
                selected = [r for r in rows if str(r[key]) == group]
                item = {'group': group, 'images': len(selected)}
                for metric in ('full_detail_rmse_ev', 'highlight_mse'):
                    methods = selected[0][metric]
                    valid = [r for r in selected if all(v is not None for v in r[metric].values())]
                    item[metric] = {'valid_images': len(valid), 'means': {
                        k: mean(r[metric][k] for r in valid) if valid else None for k in methods}}
                    item[metric]['worse_than_parent'] = sum(r[metric]['candidate'] > r[metric]['parent'] for r in valid)
                result.append(item)
            return result

        runs[name] = {'rows': rows, 'by_scene': aggregate('scene'),
                      'by_clipping_fraction': aggregate('clipping_fraction')}
    result = {'runs': runs, 'input_hashes': hashes,
              'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Previously examined DEV; equal-image means, exclude missing metrics. Parent here is saved RAW parent. No independent holdout or acceptance change.',
              'release_authorized': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: {'by_scene': v['by_scene'], 'by_clipping_fraction': v['by_clipping_fraction']} for k, v in runs.items()}, indent=2))


if __name__ == '__main__':
    main()
