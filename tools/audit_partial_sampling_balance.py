"""Audit TRAIN schedule shifts and reproduce still-pool exclusions exactly."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.aces_model import reconstruction_weight
from training.aces_partial_geometry import aligned_partial_origin
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def main():
    roots = {name: Path('G:/rudra_training')/folder for name, folder in (
        ('control', 'partial_temporal1_400_20261025'), ('coverage', 'partial_coverage400_20261025'))}
    pilots = [Path('G:/datasets/corpora')/p/'train/report.json' for p in (
        'aces_consistent_exposures_v2_20261024', 'aces_temporal_context_pairs_20261024')]
    out = Path('G:/rudra_training/partial_sampling_balance_20261025'); out.mkdir(exist_ok=False)
    inputs = pilots + [r/n for r in roots.values() for n in ('crop_pool.json', 'motion_pool.json', 'schedule.json')]
    reg = {'input_hashes': {str(p): sha(p) for p in inputs},
           'policy': 'TRAIN source masks only; exact seeded still-pool reconstruction and per-scene/exposure scheduled frame/pixel counts. No DEV or targets.',
           'release_authorized': False}
    (out/'registration.json').write_text(json.dumps(reg, indent=2))
    (out/'auditor.py').write_bytes(Path(__file__).read_bytes())
    pools = [json.loads((roots['control']/n).read_text()) for n in ('crop_pool.json', 'motion_pool.json')]
    for root in roots.values():
        for n in ('crop_pool.json', 'motion_pool.json'):
            assert sha(root/n) == sha(roots['control']/n)
    requests = {}; all_requests = defaultdict(set)
    for arm, root in roots.items():
        request = Counter()
        for step, indices in enumerate(json.loads((root/'schedule.json').read_text())):
            motion = step % 2 == 0
            for index in indices:
                p = pools[int(motion)][index]
                for asset in p['assets'] if motion else [p['asset_id']]:
                    request[(asset, *p['origin'])] += 1
                    all_requests[asset].add(tuple(p['origin']))
        requests[arm] = request
    rng = np.random.default_rng(20261025); reproduced = []; exclusions = []
    rows = {}; pixel_counts = {}; torch.set_num_threads(2)
    for domain, pilot in zip(('still', 'motion'), pilots):
        for row in json.loads(pilot.read_text())['rows']:
            asset = row['asset_id']; assert row['split'] == 'train'
            assert sha(row['sdr_path']) == row['sdr_sha256']
            s = canonical_to_source(load_rgb(row['sdr_path'], hdr=False)); h, w = s.shape[:2]
            weight = reconstruction_weight(torch.from_numpy(s).permute(2, 0, 1)[None])[0, 0].numpy()
            part = (weight > 0) & (s.max(-1) > .5) & (s.min(-1) < 254.5/255)
            rows[asset] = {**row, 'domain': domain}
            for y, x in all_requests[asset]:
                pixel_counts[(asset, y, x)] = int(part[y+64:y+128, x+64:x+128].sum())
            if domain != 'still':
                continue
            ys, xs = np.nonzero(part[48:h-48, 48:w-48]); before = len(reproduced)
            attempted = []; reason = None
            if len(ys) < 64 or min(h,w) < 192:
                reason = 'fewer_than64_candidate_pixels' if len(ys) < 64 else 'too_small'
            else:
                for k in rng.choice(len(ys), size=min(4, len(ys)), replace=False):
                    cy, cx = int(ys[k]+48), int(xs[k]+48)
                    y, x = aligned_partial_origin(cy,cx,h,w,192,64)
                    count = int(part[y+64:y+128,x+64:x+128].sum())
                    attempted.append({'centre':[cy,cx], 'origin':[y,x], 'pixels':count})
                    if count:
                        reproduced.append({'asset_id':asset, 'centre':[cy,cx], 'origin':[y,x]})
                if len(reproduced) == before:
                    reason = 'all_selected_crops_empty_after64border'
            exclusions.append({'asset_id':asset, 'scene':row['scene_id'], 'exposure':row['exposure_offset'],
                               'partial_pixels':int(part.sum()), 'candidate48_pixels':len(ys),
                               'reachable64_pixels':int(part[64:h-64,64:w-64].sum()),
                               'accepted_crops':len(reproduced)-before, 'exclusion_reason':reason,
                               'attempted':attempted})
        print(domain, 'audited', flush=True)
    assert reproduced == pools[0]
    summaries = {}
    for arm, request in requests.items():
        groups = defaultdict(Counter)
        for key, count in request.items():
            row = rows[key[0]]; domain = row['domain']
            exposure = row['exposure_offset'] if domain == 'still' else row['desired_fully_clipped_fraction']
            for group in (domain, f"{domain}/scene/{row['scene_id']}", f'{domain}/exposure_stratum/{exposure}'):
                groups[group].update(frames=count, pixels=count*pixel_counts[key])
        summaries[arm] = {k:dict(v) for k,v in groups.items()}
        train = json.loads((roots[arm]/'report.json').read_text())
        assert sum(v['pixels'] for k,v in groups.items() if k in ('still','motion')) == sum(r['pixels'] for r in train['history'])
    excluded = [r for r in exclusions if r['reachable64_pixels'] and not r['accepted_crops']]
    shifts = []
    for key in sorted(set(summaries['control']) | set(summaries['coverage'])):
        if not key.startswith('motion/scene/'):
            continue
        a, b = summaries['control'].get(key,{}), summaries['coverage'].get(key,{})
        shifts.append({'scene':key.split('/scene/')[1], 'control':a, 'coverage':b,
                       'frame_delta':b.get('frames',0)-a.get('frames',0)})
    result = {'groups':summaries, 'still_rows':exclusions, 'excluded_reachable_stills':excluded,
              'motion_scene_shifts':sorted(shifts,key=lambda r:r['frame_delta']),
              'still_pool_exactly_reproduced':True, 'registration_sha256':sha(out/'registration.json'),
              'auditor_sha256':sha(__file__), 'limitation':'Sampling distribution and coverage covary; no causal attribution to a scene or threshold.'}
    (out/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'excluded_reachable_stills':excluded, 'motion_scene_shifts':result['motion_scene_shifts'],
                      'exposure_strata':{a:{k:v for k,v in g.items() if '/exposure_stratum/' in k} for a,g in summaries.items()}},indent=2))


if __name__ == '__main__':main()
