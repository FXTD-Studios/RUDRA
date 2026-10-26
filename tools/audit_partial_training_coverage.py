"""Measure unique and repeated partial supervision in a saved TRAIN schedule."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.aces_model import reconstruction_weight
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def main():
    root = Path('G:/rudra_training/partial_temporal1_400_20261025')
    out = Path('G:/rudra_training/partial_training_coverage_20261025')
    out.mkdir(exist_ok=False)
    pilots = [Path('G:/datasets/corpora/aces_consistent_exposures_v2_20261024/train/report.json'),
              Path('G:/datasets/corpora/aces_temporal_context_pairs_20261024/train/report.json')]
    paths = [root / n for n in ('crop_pool.json', 'motion_pool.json', 'schedule.json')]
    pools = [json.loads(p.read_text()) for p in paths[:2]]
    schedule = json.loads(paths[2].read_text())
    reg = {'input_hashes': {str(p): sha(p) for p in paths + pilots},
           'policy': 'All TRAIN rows; full partial support, reachable 64px-border interior, available pool union, actual scheduled union, repeated pixel exposures. No DEV or model predictions.',
           'crop': 192, 'border': 64, 'release_authorized': False}
    (out / 'registration.json').write_text(json.dumps(reg, indent=2))
    (out / 'auditor.py').write_bytes(Path(__file__).read_bytes())
    available = defaultdict(set)
    scheduled = defaultdict(Counter)
    for motion, pool in enumerate(pools):
        for sample in pool:
            for asset in sample['assets'] if motion else [sample['asset_id']]:
                available[asset].add(tuple(sample['origin']))
    for update, indices in enumerate(schedule):
        motion = update % 2 == 0
        for index in indices:
            sample = pools[int(motion)][index]
            for asset in sample['assets'] if motion else [sample['asset_id']]:
                scheduled[asset][tuple(sample['origin'])] += 1
    torch.set_num_threads(2)
    records = []
    groups = defaultdict(Counter)
    known = set()
    for domain, pilot in zip(('still', 'motion'), pilots):
        for row in json.loads(pilot.read_text())['rows']:
            asset = row['asset_id']
            assert asset not in known and row['split'] == 'train'
            known.add(asset)
            assert sha(row['sdr_path']) == row['sdr_sha256']
            source = canonical_to_source(load_rgb(row['sdr_path'], hdr=False))
            h, w = source.shape[:2]
            assert h >= 192 and w >= 192 and h % 4 == w % 4 == 0
            weight = reconstruction_weight(torch.from_numpy(source).permute(2, 0, 1)[None])[0, 0].numpy()
            partial = (weight > 0) & (source.max(-1) > .5) & (source.min(-1) < 254.5 / 255)
            pool_mask = np.zeros((h, w), dtype=bool)
            frequency = np.zeros((h, w), dtype=np.int32)
            for y, x in available[asset]:
                assert y % 4 == x % 4 == 0 and 0 <= y <= h-192 and 0 <= x <= w-192
                pool_mask[y+64:y+128, x+64:x+128] = True
            for (y, x), count in scheduled[asset].items():
                assert (y, x) in available[asset]
                frequency[y+64:y+128, x+64:x+128] += count
            reachable = np.zeros((h, w), dtype=bool)
            reachable[64:h-64, 64:w-64] = True
            counts = Counter(frames=1, frames_with_partial=int(partial.any()),
                             frames_scheduled=int(bool(scheduled[asset])),
                             partial_pixels=int(partial.sum()),
                             reachable_pixels=int((partial & reachable).sum()),
                             pool_pixels=int((partial & pool_mask).sum()),
                             covered_pixels=int((partial & (frequency > 0)).sum()),
                             pixel_exposures=int(frequency[partial].sum()),
                             frame_exposures=sum(scheduled[asset].values()))
            counts['never_supervised_pixels'] = counts['partial_pixels']-counts['covered_pixels']
            counts['reachable_uncovered_pixels'] = counts['reachable_pixels']-counts['covered_pixels']
            exposure = row.get('exposure_offset', row.get('render_ev'))
            records.append({'asset_id': asset, 'domain': domain, 'scene': row['scene_id'],
                            'exposure': exposure, **counts})
            for key in ('all', domain, f"{domain}/scene/{row['scene_id']}", f'{domain}/exposure/{exposure}'):
                groups[key].update(counts)
        print(domain, 'complete', flush=True)
    assert set(scheduled) <= known
    assert groups['all']['frame_exposures'] == 1600
    calibration = Path('G:/rudra_training/partial_display_fixed_scale_20261025/report.json')
    assert groups['all']['pixel_exposures'] == json.loads(calibration.read_text())['supervised_pixel_exposures']
    summaries = {}
    for key, counts in groups.items():
        summaries[key] = dict(counts)
        for name in ('partial', 'reachable', 'pool'):
            denom = counts[name+'_pixels']
            summaries[key]['covered_fraction_of_'+name] = counts['covered_pixels']/denom if denom else None
    result = {'groups': summaries, 'rows': records, 'registration_sha256': sha(out/'registration.json'),
              'auditor_sha256': sha(__file__), 'calibration_crosscheck_sha256': sha(calibration),
              'limitations': 'Coverage measures exposure, not learning or causal explanation of DEV errors. Target values are not read. Repeated views and motion frames are not independent scenes.'}
    (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: summaries[k] for k in ('all', 'still', 'motion')}, indent=2))


if __name__ == '__main__':
    main()
