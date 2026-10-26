"""Count registered crop exposures without model inference or training changes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def requests(stills, motion, schedule):
    counts = Counter()
    for step, indices in enumerate(schedule):
        is_motion = step % 2 == 0  # zero-based step 0 is a pair update
        pool = motion if is_motion else stills
        if len(indices) != (2 if is_motion else 4) or len(set(indices)) != len(indices):
            raise ValueError("Expected the registered four-frame alternating recipe")
        for index in indices:
            if type(index) is not int or not 0 <= index < len(pool):
                raise ValueError("Invalid pool index")
            crop = pool[index]
            assets = crop['assets'] if is_motion else [crop['asset_id']]
            if len(assets) != (2 if is_motion else 1):
                raise ValueError("Invalid frame count")
            for asset in assets:
                counts[(asset, *crop['origin'])] += 1
    return counts


def summarize(stills, motion, schedule, prefix):
    if not 0 < prefix < len(schedule):
        raise ValueError("Prefix must be shorter than schedule")
    before = requests(stills, motion, schedule[:prefix])
    total = requests(stills, motion, schedule)
    later = total - before
    new = total.keys() - before.keys()
    return {
        'prefix_steps': prefix, 'total_steps': len(schedule),
        'prefix_unique_crops': len(before), 'total_unique_crops': len(total),
        'additional_unique_crops': len(new),
        'later_frame_exposures': sum(later.values()),
        'later_exposures_on_new_crops': sum(later[key] for key in new),
        'total_frame_exposures': sum(total.values()),
        'later_exposures_on_prefix_crops': sum(later[key] for key in before),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--prefix', type=int, nargs='+', default=[400, 1600])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    files = [args.root / name for name in ('crop_pool.json', 'motion_pool.json', 'schedule.json')]
    stills, motion, schedule = [json.loads(path.read_text()) for path in files]
    report = {
        'inputs_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        'comparisons': [summarize(stills, motion, schedule, prefix) for prefix in args.prefix],
        'scope': 'Registered schedule only; does not prove executed updates, pixel coverage, convergence or quality.',
        'release_authorized': False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
