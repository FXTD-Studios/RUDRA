"""Scale-up training crops for the learned ACES 2.0 partial-highlight route (27 Oct 2026).

Every commercial-clean TRAIN plate in the run05 source registry (FXTD Studios own footage,
Poly Haven CC0 camera moves, CC0/authored highlight and fire renders), minus every scene any
ACES/run05 development, validation or holdout set uses and the fresh FXTD grade set, rendered
through the exact ACES 2.0 SDR 100-nit view, as aces_consistent_exposures_v2 did:

  reference = plate (scene units, AP1) * 2**ev
  source    = rint(clip(view.forward(reference), 0, 1) * 255) / 255        (8-bit codes)
  target    = view.inverse(clean held within 0.49 code of source); reference in highlights;
              view.inverse(source) in shadow support                     (consistent target)

Exposure: FXTD plates at their native camera exposure (1.0 = diffuse white, as shot); every
other plate auto-exposed so its median AP1 luminance is 0.18. Then offsets 0, +2, +4 stops.
A (plate, offset) case is used when the median max code is <= 0.9 and at least 64 partially
clipped pixels (reconstruction weight > 0, max code > 0.5, min code < 254.5/255) lie inside the
crop band. Up to 4 crops per case at aligned origins (border 64), as the trainer's own pool.
Crops whose consistent target misses the source by more than 1 code anywhere are dropped.

Output: shards of N crops (src uint8 codes, target float32 ACEScg), crops.jsonl, summary.json.
Read by tools/train_partial_display.py --crop-store.
"""
import argparse, hashlib, json, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCES = 'G:/datasets/corpora/rudra_run05_sources_20261022/sources.jsonl'
FRESH = 'G:/rudra_training/claude_release_candidate_20261026/fresh_fxtd_grade_v2_20261026/report.json'
EVAL_DIR_PATTERN = re.compile(r'(aces|run05|ambientcg|holdout|temporal|fresh)', re.I)
EVAL_NAME_PATTERN = re.compile(r'(val|dev|holdout|validation|development|quarantine|test|candidate)', re.I)
OFFSETS = (0.0, 2.0, 4.0)
CROP, BORDER, PER_CASE, MAX_SIDE = 192, 64, 4, 1920
SHARD = 256


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def gpath(p, root):
    """G:/... registry paths, optionally re-rooted (tests)."""
    p = str(p).replace('\\', '/')
    return p if not root else root.rstrip('/') + '/' + p.split(':/', 1)[1]


def evaluation_scenes(corpora: Path, fresh: Path):
    """Every scene id a development, validation, holdout or fresh evaluation set touches."""
    out, files = set(), []
    for d in sorted(corpora.iterdir()):
        if not d.is_dir() or not EVAL_DIR_PATTERN.search(d.name):
            continue
        for f in list(d.rglob('report.json')) + list(d.rglob('*.jsonl')):
            rel = str(f.relative_to(d))
            whole = bool(EVAL_NAME_PATTERN.search(d.name)) or bool(EVAL_NAME_PATTERN.search(rel))
            try:
                rows = (json.loads(f.read_text()).get('rows', []) if f.suffix == '.json'
                        else [json.loads(l) for l in f.read_text().splitlines() if l.strip()])
            except Exception:
                continue
            n = 0
            for r in rows:
                if not isinstance(r, dict) or 'scene_id' not in r:
                    continue
                if whole or str(r.get('split', 'train')) != 'train':
                    out.add(r['scene_id']); n += 1
            if n:
                files.append([str(f), n])
    fr = json.loads(fresh.read_text())
    # The scenes whose cases the fresh grade set kept (and evaluated); the others were only
    # screened for highlight coverage, with no model output looked at.
    fresh_scenes = {r['scene_id'] for r in fr['rows'] if r.get('kept')}
    assert len(fresh_scenes) == fr['scenes'], 'fresh scene count'
    out |= fresh_scenes
    return out, files, sorted(fresh_scenes)


def area_downsample(img, max_side):
    import cv2
    h, w = img.shape[:2]
    k = max(h, w) / max_side
    if k > 1:
        img = cv2.resize(img, (int(round(w / k)), int(round(h / k))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    return img[:h // 4 * 4, :w // 4 * 4]


def process(job):
    """One source plate -> its eligible crops (all offsets)."""
    import torch
    from pipeline.run05_corpus import load_plate, NETWORK_PER_SCENE
    from rudra.delivery.colorspace import convert
    from training.aces_model import reconstruction_weight
    from training.aces_reference import reference_view
    from training.aces_partial_geometry import aligned_partial_origin
    torch.set_num_threads(1)
    row, root, seed = job['row'], job['root'], job['seed']
    view = reference_view(); view.set_threads(1)
    rr = dict(row); rr['hdr_path'] = gpath(row['hdr_path'], root)
    plate = load_plate(rr) / NETWORK_PER_SCENE                          # scene units, BT.2020
    plate = area_downsample(np.ascontiguousarray(plate, np.float32), MAX_SIDE)
    plate = convert(plate, 'rec2020', 'ap1').astype(np.float32)
    h, w = plate.shape[:2]
    if min(h, w) < CROP:
        return {'asset_id': row['asset_id'], 'skipped': 'small'}, []
    Y = plate @ np.array([.2722287168, .6740817658, .0536895174], np.float32)
    native = row.get('licence_source') == 'fxtd'
    base = 0.0 if native else float(np.clip(np.log2(0.18 / max(float(np.median(Y[Y > 0])) if (Y > 0).any() else 0.18, 1e-6)), -10, 10))
    rng = np.random.default_rng([seed, int(hashlib.sha256(row['asset_id'].encode()).hexdigest()[:8], 16)])
    crops, cases = [], []
    for off in OFFSETS:
        ev = base + off
        ref = (plate * np.float32(2.0 ** ev)).astype(np.float32)
        clean = view.forward(ref).clip(0, 1)
        src = (np.rint(clean * 255) / 255).astype(np.float32)
        mx = src.max(-1)
        if float(np.median(mx)) > 0.9:
            cases.append({'ev': ev, 'offset': off, 'skipped': 'blown'}); continue
        wgt = reconstruction_weight(torch.from_numpy(src).permute(2, 0, 1)[None])[0, 0].numpy()
        hi = (wgt > 0) & (mx > .5)
        shadow = (wgt > 0) & ~hi
        part = hi & (src.min(-1) < 254.5 / 255)
        ys, xs = np.nonzero(part[CROP // 4:h - CROP // 4, CROP // 4:w - CROP // 4])
        if len(ys) < 64:
            cases.append({'ev': ev, 'offset': off, 'skipped': 'no_partial', 'partial': int(len(ys))}); continue
        corrected = np.clip(clean, src - .49 / 255, src + .49 / 255).clip(0, 1).astype(np.float32)
        tgt = view.inverse(corrected)
        tgt[hi] = ref[hi]
        if shadow.any():
            tgt[shadow] = view.inverse(src[shadow])
        err = np.abs(view.forward(tgt) - src).max(-1) * 255
        kept = 0
        for k in rng.choice(len(ys), size=min(PER_CASE, len(ys)), replace=False):
            cy, cx = ys[k] + CROP // 4, xs[k] + CROP // 4
            y0, x0 = aligned_partial_origin(cy, cx, h, w, CROP, BORDER)
            if not part[y0 + BORDER:y0 + CROP - BORDER, x0 + BORDER:x0 + CROP - BORDER].any():
                continue
            if (err[y0:y0 + CROP, x0:x0 + CROP] > 1).any():
                continue
            crops.append((np.rint(src[y0:y0 + CROP, x0:x0 + CROP] * 255).astype(np.uint8),
                          np.ascontiguousarray(tgt[y0:y0 + CROP, x0:x0 + CROP], np.float32),
                          {'asset_id': row['asset_id'], 'scene_id': row['scene_id'], 'source_id': row['source_id'],
                           'licence': row.get('licence_licence'), 'ev': round(ev, 4), 'offset': off,
                           'centre': [int(cy), int(cx)], 'origin': [int(y0), int(x0)], 'frame_hw': [h, w]}))
            kept += 1
        cases.append({'ev': ev, 'offset': off, 'partial': int(len(ys)), 'crops': kept})
    return {'asset_id': row['asset_id'], 'scene_id': row['scene_id'], 'base_ev': base, 'cases': cases}, crops


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--registration', required=True)
    ap.add_argument('--root', default='', help='re-root G:/ paths (tests)')
    ap.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument('--seed', type=int, default=20261027)
    ap.add_argument('--limit', type=int, default=0, help='first N eligible plates only (smoke test)')
    ap.add_argument('--sources', default=SOURCES, help='source registry (tests)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    src_path = Path(gpath(a.sources, a.root))
    rows = [json.loads(l) for l in src_path.read_text().splitlines() if l.strip()]
    excluded, excl_files, fresh = evaluation_scenes(Path(gpath('G:/datasets/corpora', a.root)), Path(gpath(FRESH, a.root)))
    cand = [r for r in rows if r['split'] == 'train' and r.get('commercial_ok') is True]
    train = [r for r in cand if r['scene_id'] not in excluded]
    val_scenes = {r['scene_id'] for r in rows if r['split'] != 'train'}
    assert not ({r['scene_id'] for r in train} & (val_scenes | excluded)), 'scene leak'
    assert not ({r['scene_id'] for r in train} & set(fresh)), 'fresh holdout leak'
    if a.limit:
        train = train[:a.limit]
    (out / 'exclusions.json').write_text(json.dumps({'excluded_scenes': sorted(excluded), 'fresh_scenes': fresh,
                                                     'files': excl_files, 'dropped_train_rows': len(cand) - len(train) if not a.limit else None}, indent=1))
    print('plates', len(train), 'of', len(cand), 'train rows;', len(excluded), 'scenes excluded', flush=True)
    t0 = time.time(); shard, n_shard, total, plates = [], 0, 0, []
    crops_jsonl = (out / 'crops.jsonl').open('w')

    def flush():
        nonlocal shard, n_shard
        if not shard:
            return
        np.save(out / f'shard_{n_shard:04d}_src.npy', np.stack([c[0] for c in shard]))
        np.save(out / f'shard_{n_shard:04d}_tgt.npy', np.stack([c[1] for c in shard]))
        for i, c in enumerate(shard):
            crops_jsonl.write(json.dumps({**c[2], 'shard': n_shard, 'index': i}) + '\n')
        shard = []; n_shard += 1

    jobs = [{'row': r, 'root': a.root, 'seed': a.seed} for r in train]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (rep, crops) in enumerate(ex.map(process, jobs, chunksize=4)):
            plates.append(rep)
            for c in crops:
                shard.append(c); total += 1
                if len(shard) == SHARD:
                    flush()
            if (i + 1) % 50 == 0:
                print(f'{i + 1}/{len(jobs)} plates, {total} crops, {time.time() - t0:.0f} s', flush=True)
    flush(); crops_jsonl.close()
    (out / 'plates.json').write_text(json.dumps(plates, indent=1))
    meta = [json.loads(l) for l in (out / 'crops.jsonl').read_text().splitlines()]
    by = lambda k: dict(sorted({v: sum(1 for m in meta if m[k] == v) for v in {m[k] for m in meta}}.items()))
    summary = {'crops': total, 'shards': n_shard, 'crop': CROP, 'border': BORDER, 'per_case': PER_CASE,
               'plates_considered': len(train), 'plates_used': len({m['asset_id'] for m in meta}),
               'scenes_used': len({m['scene_id'] for m in meta}), 'crops_by_source': by('source_id'),
               'crops_by_licence': by('licence'), 'crops_by_offset': by('offset'),
               'excluded_scene_count': len(excluded), 'fresh_scene_count': len(fresh),
               'registration_sha256': sha(a.registration), 'builder_sha256': sha(__file__),
               'sources_sha256': sha(src_path), 'seconds': round(time.time() - t0),
               'commercial_clean': all(m['licence'] in ('CC0', 'FXTD Studios own footage',
                                                         'CC0 sources and authored geometry; source receipts retained') for m in meta),
               'release_authorized': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
