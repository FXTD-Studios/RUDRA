"""Prepare CC0 development sources with conservative name-family exclusions.

This is a development split screen, not proof of capture/location independence.
Existing validation sources stay fixed. All historically held-out name families
are excluded from new training; source bytes are hashed before conversion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.run05_sources import panorama_key


def family(scene):
    return panorama_key(scene.removeprefix('ph:')).removeprefix('ph:').split('_')[0]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def select(sources, pilot, historical, count=200, dark_count=64):
    valid = [r for r in pilot if r['split'] == 'val']
    held = {family(r['scene_id']) for r in sources + pilot
            if r.get('split') in ('val', 'test')}
    held.update(family(r['scene_id']) for r in historical
                if r.get('split') in ('val', 'test') and
                ('polyhaven' in r.get('source_id', '') or r['scene_id'].startswith(('ph_', 'hl_ph_', 'ph:'))))
    by_scene = {}
    for r in sources:
        if (r['source_id'] != 'polyhaven_moves' or r.get('commercial_ok') is not True
                or r.get('licence_licence') != 'CC0' or r['split'] != 'train'
                or family(r['scene_id']) in held):
            continue
        previous = by_scene.get(r['scene_id'])
        if previous is None or r['asset_id'] < previous['asset_id']:
            by_scene[r['scene_id']] = r
    rank = lambda r: hashlib.sha256(('run05-expansion:' + r['scene_id']).encode()).hexdigest()
    candidates = sorted(by_scene.values(), key=rank)
    dark = lambda r: any(s in r['scene_id'] for s in ('night', 'dark', 'cellar', 'cave', 'interior', 'sunset', 'sunrise'))
    priority = [r for r in candidates if dark(r)][:dark_count]
    ids = {r['scene_id'] for r in priority}
    selected = (priority + [r for r in candidates if r['scene_id'] not in ids])[:count]
    if len(selected) < count:
        raise ValueError(f'only {len(selected)} eligible scenes, requested {count}')
    return selected + valid, dict(train_scenes=len(selected), val_scenes=len({r['scene_id'] for r in valid}),
                                 name_selected_dark_scenes=sum(dark(r) for r in selected),
                                 excluded_name_families=sorted(held), eligible_train_scenes=len(candidates),
                                 independence='development only; name heuristic is not location verification')


def main():
    ap = argparse.ArgumentParser()
    for name in ('sources', 'pilot', 'historical', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    read = lambda p: [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    rows, report = select(read(a.sources), read(a.pilot), read(a.historical))
    if a.out.exists():
        raise ValueError('use a new output directory')
    for r in rows:
        actual = digest(r['hdr_path'])
        if r.get('hdr_sha256') and actual != r['hdr_sha256']:
            raise ValueError(f"source changed: {r['asset_id']}")
        r['hdr_sha256'] = actual
        r['licence_url'] = 'https://polyhaven.com/license'
    report['inputs'] = {str(p): digest(p) for p in (a.sources, a.pilot, a.historical)}
    report['release_authorized'] = False
    a.out.mkdir(parents=True)
    manifest = a.out / 'sources.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    report['sources_sha256'] = digest(manifest)
    (a.out / 'selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
