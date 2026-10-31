"""Prepare a TRAIN-only maximum-new-pixel greedy motion schedule."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.aces_model import reconstruction_weight
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matched-groups', action='store_true')
    args = parser.parse_args()
    root = Path('G:/rudra_training/partial_temporal1_400_20261025')
    out = Path('G:/rudra_training/partial_group_coverage_schedule_20261025' if args.matched_groups else 'G:/rudra_training/partial_coverage_schedule_20261025')
    out.mkdir(exist_ok=False)
    pilot = Path('G:/datasets/corpora/aces_temporal_context_pairs_20261024/train/report.json')
    paths = [root/n for n in ('crop_pool.json', 'motion_pool.json', 'schedule.json')]
    reg = {'input_hashes': {str(p): sha(p) for p in paths+[pilot]},
           'policy': 'Select 400 distinct motion pairs greedily by newly covered partial pixels; seeded tie priority, then seeded shuffle. Preserve all still batches. No DEV/targets/model errors. Larger supports receive higher priority; coverage is not isolated from sampling distribution.',
           'seed': 20261025, 'release_authorized': False}
    if args.matched_groups:
        reg['policy'] = 'Keep every original motion slot group (scene/exposure) and every still batch. Within each group greedily choose new partial-pixel coverage, break ties by least reuse then seeded priority. No duplicate pair inside a batch. No DEV/target/model inputs.'
    (out/'registration.json').write_text(json.dumps(reg, indent=2))
    (out/'preparer.py').write_bytes(Path(__file__).read_bytes())
    pool = json.loads(paths[1].read_text())
    old = json.loads(paths[2].read_text())
    assert len(old) == 400 and len(pool) >= 400
    needed = {asset for p in pool for asset in p['assets']}
    masks = {}; offset = 0
    torch.set_num_threads(2)
    for row in json.loads(pilot.read_text())['rows']:
        assert row['split'] == 'train'
        if row['asset_id'] not in needed:
            continue
        assert sha(row['sdr_path']) == row['sdr_sha256']
        s = canonical_to_source(load_rgb(row['sdr_path'], hdr=False))
        w = reconstruction_weight(torch.from_numpy(s).permute(2, 0, 1)[None])[0, 0].numpy()
        mask = (w > 0) & (s.max(-1) > .5) & (s.min(-1) < 254.5/255)
        masks[row['asset_id']] = (mask, offset)
        offset += mask.size
    sets = []
    for p in pool:
        pixels = set(); y, x = p['origin']
        for asset in p['assets']:
            mask, start = masks[asset]
            yy, xx = np.nonzero(mask[y+64:y+128, x+64:x+128])
            pixels.update((start+(yy+y+64)*mask.shape[1]+xx+x+64).tolist())
        sets.append(pixels)
    rng = np.random.default_rng(reg['seed'])
    priority = rng.permutation(len(pool)).tolist()
    covered = set(); selected = []; gains = []
    remaining = set(range(len(pool)))
    if args.matched_groups:
        counts = np.zeros(len(pool), dtype=np.int32)
        for batch in old[::2]:
            batch_selected = []
            for original in batch:
                options = [i for i in priority if pool[i]['group'] == pool[original]['group'] and i not in batch_selected]
                best = max(options, key=lambda i: (len(sets[i]-covered), -int(counts[i])))
                gains.append(len(sets[best]-covered)); covered.update(sets[best])
                counts[best] += 1; selected.append(best); batch_selected.append(best)
    else:
        for _ in range(400):
            best = max((i for i in priority if i in remaining), key=lambda i: len(sets[i]-covered))
            gains.append(len(sets[best]-covered)); covered.update(sets[best])
            remaining.remove(best); selected.append(best)
        rng.shuffle(selected)
    schedule = [list(batch) for batch in old]
    for i in range(200):
        schedule[2*i] = selected[2*i:2*i+2]
    old_indices = [i for batch in old[::2] for i in batch]
    old_union = set().union(*(sets[i] for i in old_indices))
    pool_union = set().union(*sets)
    assert all(a == b for a, b in zip(old[1::2], schedule[1::2]))
    if args.matched_groups:
        assert all([pool[i]['group'] for i in a] == [pool[i]['group'] for i in b] for a,b in zip(old[::2],schedule[::2]))
    assert len(covered) > len(old_union)
    (out/'schedule.json').write_text(json.dumps(schedule))
    manifest = {'crop_pool_sha256': sha(paths[0]), 'motion_pool_sha256': sha(paths[1]),
                'schedule_sha256': sha(out/'schedule.json'), 'control_schedule_path': str(paths[2]),
                'control_schedule_sha256': sha(paths[2]), 'steps': 400, 'batch': 4,
                'crop': 192, 'border': 64, 'paired_motion': True,
                'preserve_motion_groups': args.matched_groups,
                'pilot_hashes': {str(pilot): sha(pilot)}, 'registration_sha256': sha(out/'registration.json')}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    report = {'old_motion_union': len(old_union), 'new_motion_union': len(covered),
              'motion_pool_union': len(pool_union), 'old_unique_pairs': len(set(old_indices)),
              'new_unique_pairs': len(set(selected)), 'old_pixel_exposures': sum(len(sets[i]) for i in old_indices),
              'new_pixel_exposures': sum(len(sets[i]) for i in selected), 'selection_gains': gains,
              'manifest_sha256': sha(out/'manifest.json'), 'preparer_sha256': sha(__file__)}
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'selection_gains'}, indent=2))


if __name__ == '__main__':
    main()
