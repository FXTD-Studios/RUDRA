"""AI fill v5 (22 Nov 2026): how often the trainer draws each clip of a built v5 store, and which dev clips the
dev-loss monitor leaves out. Written after the clip review of 22 Nov (tmp/v5_review/REVIEW_2026-11-22.md),
before any v5 training; the rules below are fixed in research_genfill_v5_20261122.json.

    python tools/genfill_v5_weights.py --clips G:/datasets/corpora/research_genfill_clips_v5_20261122
        [--scenes G:/datasets/sources/genfill_v5/blender/manifest.jsonl] [--out <clips>/sampling_v5.json] [--dry-run]

Each clip's weight = theme factor x edge factor (training/genfill_train.py --clip-weights multiplies it into
its own v4.2 structure weight):
  theme factor  the largest of its themes' factors (THEMES; a clip with no theme 1). The review found sun on
                31 % of the clips and 52 % of the themed ones, and door, practical and thin, where 09V fails,
                behind it; ambientCG tags most of its skies both sun and sky.
  edge factor   1 + EDGE_BOOST where frame 0 has at least EDGE_MIN_PX thin dark pixels (no wider than 6 px,
                3 stops or more below the region's median log luminance) within 8 px outside the mask: a thin
                thing against the bright region, as 09V's antenna and cables. The review found these no more
                often in the thin-themed clips (40-51 %) than in the old material (55 %), so the theme alone
                does not find them.
Left out (weight 0, never in the dev monitor): the Blender scenes in BROKEN_SCENES. Their camera's rotation
wrapped round (an Euler angle crossing +-180 degrees between two keyframes) and Cycles' motion blur swept the
whole turn into the frame: whole frames smeared in a swirl (found 22 Nov 2026 by the sharpness of each frame,
tmp/v5chk/sharp.py: 9 of the 61 motion-blurred scenes, every one thin or glare; fixed in
tools/blender/genfill_v5_scenes.py for later renders).
Dev monitor: a dev clip is left out when its Blender scene is lit by a sky in the other split, or when it comes
from a panorama that lights a training Blender scene (the review found 11 such scenes).
Writes one JSON: the rules, the sha256 of the store's clips.jsonl, the expected share of draws by theme and
source (with the v4.2 structure boost of the registration), and one row per clip."""
import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

THEMES = {'door': 2.0, 'glare': 2.0, 'practical': 1.5, 'thin': 1.5, 'window': 1.25, 'sun': 0.5, 'sky': 0.5}
EDGE_BOOST, EDGE_MIN_PX = 1.0, 30
EDGE_KINDS = ('region', 'highlight', 'sky')
DARK_STOPS, THIN_OPEN_PX, BAND_PX = 3.0, 7, 8
STRUCTURE_BOOST, STRUCTURE_REF = 3.0, 0.10          # the registration's trainer arguments, for the expected shares
# scene -> the frames smeared (a frame's Laplacian variance of log luminance under 0.4 x the scene's median)
BROKEN_SCENES = {'blender:glare_00109': [30, 31, 32, 33], 'blender:thin_00140': [9, 16, 22, 23, 24],
                 'blender:thin_00160': [7, 13, 14, 22], 'blender:glare_00014': [4, 5, 13, 22, 23, 24, 25, 26],
                 'blender:glare_00094': [4, 8, 9, 13, 14, 23, 26, 27, 30], 'blender:thin_00020': [10, 14, 17, 25, 26, 27, 28],
                 'blender:thin_00030': [15, 20, 28], 'blender:glare_00074': [12, 20, 23, 30], 'blender:glare_00024': [2]}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def theme_factor(themes):
    return max((THEMES[t] for t in themes if t in THEMES), default=1.0)


def thin_edge_px(ylog, mask):
    """Frame 0: thin dark pixels (opened away by a 7 px square, DARK_STOPS below the region's median) within
    BAND_PX outside the mask."""
    import cv2
    m = mask.astype(np.uint8)
    if m.sum() < 50:
        return 0
    med = float(np.median(ylog[m > 0]))
    dark = (ylog < med - DARK_STOPS).astype(np.uint8)
    thin = dark & (1 - cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((THIN_OPEN_PX, THIN_OPEN_PX), np.uint8)))
    band = cv2.dilate(m, np.ones((2 * BAND_PX + 1, 2 * BAND_PX + 1), np.uint8)) > 0
    return int(((thin > 0) & band & (m == 0)).sum())


def structure_weight(m):
    s = m.get('structure')
    return 1.0 + STRUCTURE_BOOST * min(float(s) / STRUCTURE_REF, 1.0) if (m.get('kind') == 'region' and s is not None) else 1.0


def crossed_dev(meta, scenes):
    """Scene ids whose dev clips the monitor leaves out: a Blender scene and its sky in different splits."""
    split = {}
    for m in meta:
        split.setdefault(m['scene_id'], m['split'])
    out = set()
    for r in scenes:
        s, k = r['scene_id'], r.get('sky_scene_id')
        if s in split and k in split and split[s] != split[k]:
            out.add(s if split[s] == 'dev' else k)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clips', required=True)
    ap.add_argument('--scenes', default='G:/datasets/sources/genfill_v5/blender/manifest.jsonl')
    ap.add_argument('--out', default='')
    ap.add_argument('--dry-run', action='store_true', help='print the shares, write nothing')
    a = ap.parse_args(argv)
    root = Path(a.clips)
    s = json.loads((root / 'summary.json').read_text())
    if s.get('recipe') != 'v5':
        sys.exit(f'{root} is not a v5 store')
    meta = [json.loads(l) for l in (root / 'clips.jsonl').read_text().splitlines() if l.strip()]
    sp = Path(a.scenes)
    scenes = [json.loads(l) for l in sp.read_text(encoding='utf-8').splitlines() if l.strip()] if sp.exists() else []
    if not scenes and any(m['source'] == 'blender_v5' for m in meta):
        sys.exit(f'no Blender manifest at {sp}: needed for the dev rule')
    drop = crossed_dev(meta, scenes)
    F, (H, W) = s['frames'], s['size']
    n0 = H * W
    cache = {}

    def arr(shard, kind):
        if (shard, kind) not in cache:
            cache[(shard, kind)] = np.load(root / f'{shard}_{kind}.npy', mmap_mode='r')
        return cache[(shard, kind)]

    rows = []
    for m in meta:
        edge = 0
        if m['kind'] in EDGE_KINDS:
            y = np.asarray(arr(m['shard'], 'ylog')[m['index']], np.float32)
            mk = np.unpackbits(np.asarray(arr(m['shard'], 'mask')[m['index']][:n0 // 8]))[:n0].reshape(H, W)
            edge = thin_edge_px(y, mk)
        tf = theme_factor(m.get('themes') or ())
        ef = 1.0 + EDGE_BOOST if edge >= EDGE_MIN_PX else 1.0
        broken = m['scene_id'] in BROKEN_SCENES
        rows.append({'shard': m['shard'], 'index': m['index'], 'split': m['split'], 'asset_id': m['asset_id'],
                     'weight': 0.0 if broken else round(tf * ef, 4), 'theme_factor': tf, 'thin_edge_px': edge,
                     'dev_monitor': not broken and not (m['split'] == 'dev' and m['scene_id'] in drop)})

    def shares(weights):
        tr = [(m, w) for m, w in zip(meta, weights) if m['split'] == 'train']
        tot = sum(w for _, w in tr)
        th, src = defaultdict(float), defaultdict(float)
        for m, w in tr:
            for t in (m.get('themes') or ['(none)']):
                th[t] += w / tot
            src[m['source']] += w / tot
        edge = sum(w for (m, w), r in zip(tr, [r for r in rows if r['split'] == 'train']) if r['thin_edge_px'] >= EDGE_MIN_PX) / tot
        return {'by_theme': {k: round(v, 4) for k, v in sorted(th.items())},
                'by_source': {k: round(v, 4) for k, v in sorted(src.items())}, 'thin_edge': round(edge, 4)}

    sw = [structure_weight(m) for m in meta]
    out = {'id': 'genfill_v5_sampling', 'written_by': 'tools/genfill_v5_weights.py', 'tool_sha256': sha(__file__),
           'clips': str(root), 'clips_jsonl_sha256': sha(root / 'clips.jsonl'),
           'rules': {'themes': THEMES, 'edge_boost': EDGE_BOOST, 'edge_min_px': EDGE_MIN_PX, 'edge_kinds': list(EDGE_KINDS),
                     'dark_stops': DARK_STOPS, 'thin_open_px': THIN_OPEN_PX, 'band_px': BAND_PX,
                     'combine': 'theme factor x edge factor; the trainer multiplies it into its structure weight'},
           'dev_left_out': {'scenes': sorted(drop), 'clips': sum(not r['dev_monitor'] for r in rows)},
           'left_out': {'scenes': sorted(s_ for s_ in BROKEN_SCENES if any(m['scene_id'] == s_ for m in meta)),
                        'clips': sum(r['weight'] == 0 for r in rows), 'why': 'Blender camera rotation wrapped: motion-blur swirl'},
           'counts': {'clips': len(rows), 'thin_edge_clips': sum(r['thin_edge_px'] >= EDGE_MIN_PX for r in rows),
                      'weight_mean_train': round(float(np.mean([r['weight'] for r in rows if r['split'] == 'train'])), 4)},
           'expected_share_of_draws': {'v4_2_structure_only': shares(sw),
                                       'v5': shares([a_ * r['weight'] for a_, r in zip(sw, rows)])},
           'rows': rows}
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
    if not a.dry_run:
        dest = Path(a.out) if a.out else root / 'sampling_v5.json'
        if dest.exists():
            sys.exit(f'{dest} exists: the sampling file is written once (move it aside to rewrite)')
        dest.write_text(json.dumps(out))
        print(f'written {dest} ({sha(dest)})')


if __name__ == '__main__':
    main()
