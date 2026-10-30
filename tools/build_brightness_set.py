"""Training set for the v3 brightness estimator (rudra/genfill/brightness.py), 30 Oct 2026.

The same commercial-clean TRAIN sources and exclusions as the clip builder (tools/build_genfill_clips.plan):
Poly Haven HDRIs (perspective views) and the FXTD / CC0 plates (a 16:9 window). Single frames at
512 x 288, exposed so that something clips: 'highlight' (aimed at the brightest source, +0..+3 EV),
'broad' (raised until >= 5 % is at code 255: wide, mild clipping) and 'random' (-1..+3 EV).
For every fully clipped component (max code 255, >= 16 px): its FEATURES from the 8-bit codes and the
exact inverse only, and the target b = mean(log2 Y_true - log2 Y_master) over the component.

Output: set.npz (X, y, w = sqrt(area), split, scene ids), rows.jsonl, summary.json.
"""
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_genfill_clips import (AP1_Y, POLYHAVEN, SOURCES, dev_scene, equirect_path, gpath,  # noqa: E402
                                       plan, plate_path, read_exr_rgb, sha)


def frames_of(job):
    from rudra.aces_view import ACES2View
    from rudra.delivery.colorspace import convert
    from rudra.genfill.brightness import FEATURES, component_features, components, frame_features
    import cv2
    cv2.setNumThreads(1)
    h, w = job['h'], job['w']
    rng = np.random.default_rng([job['seed'], int(hashlib.sha256(job['asset_id'].encode()).hexdigest()[:8], 16)])
    view = ACES2View(threads=1)
    if job['kind'] == 'equirect':
        src = np.nan_to_num(np.maximum(convert(read_exr_rgb(job['path']), 'rec709', 'ap1'), 0), posinf=6e4).astype(np.float32)
        native = False
    else:
        from pipeline.run05_corpus import load_plate, NETWORK_PER_SCENE
        row = dict(job['row']); row['hdr_path'] = gpath(row['hdr_path'], job['root'])
        src = convert(load_plate(row) / NETWORK_PER_SCENE, 'rec2020', 'ap1').astype(np.float32)
        k = max(src.shape[:2]) / 1920
        if k > 1:
            src = cv2.resize(src, (int(src.shape[1] / k), int(src.shape[0] / k)), interpolation=cv2.INTER_AREA)
        native = job['row'].get('licence_source') == 'fxtd'
    rows, X, y, wts = [], [], [], []
    for f in range(job['frames_per_source']):
        r = rng.uniform()
        kind = 'highlight' if r < 0.4 else 'broad' if r < 0.75 else 'random'
        if job['kind'] == 'equirect':
            ref = equirect_path(src, rng, 1, h, w, 'random' if kind == 'broad' else kind)[0]
        else:
            ref = plate_path(src, rng, 1, h, w)[0]
        ref = np.maximum(ref, 0).astype(np.float32)
        Y = ref @ AP1_Y
        base = 0.0 if native else float(np.clip(np.log2(0.18 / max(float(np.median(Y[Y > 0])) if (Y > 0).any() else 0.18, 1e-6)), -10, 10))

        def codes_at(o):
            return np.rint(np.clip(view.forward((ref * np.float32(2.0 ** (base + o))).reshape(-1, 3)).reshape(ref.shape), 0, 1) * 255).astype(np.uint8)
        if kind == 'broad':
            off = None
            for o in np.arange(-2.0, 6.01, 0.5):
                if (codes_at(o).max(-1) >= 255).mean() >= 0.05:
                    off = float(o); break
            if off is None:
                continue
        else:
            off = float(rng.choice([0.0, 1.0, 2.0, 3.0]) if kind == 'highlight' else rng.choice([-1.0, 0.0, 1.0, 2.0, 3.0]))
        truth = ref * np.float32(2.0 ** (base + off))
        codes = codes_at(off)
        master = view.inverse((codes.astype(np.float32) / 255).reshape(-1, 3)).reshape(codes.shape).astype(np.float32)
        lab, ids = components(codes)
        if not ids:
            continue
        fr = frame_features(codes, master)
        lt = np.log2(np.maximum(np.minimum(truth, 6e4) @ AP1_Y, 1e-6))
        for c in ids:
            m = lab == c
            feats = component_features(codes, master, m, fr)
            b = float(np.clip((lt[m] - fr['ly'][m]).mean(), 0.0, 12.0))
            X.append(feats); y.append(b); wts.append(float(np.sqrt(m.sum())))
            rows.append({'asset_id': job['asset_id'], 'scene_id': job['scene_id'], 'source': job['source'], 'kind': kind,
                         'ev': round(base + off, 3), 'frame': f, 'pixels': int(m.sum()), 'b': round(b, 4)})
    return rows, X, y, wts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--registration', required=True)
    ap.add_argument('--root', default='')
    ap.add_argument('--sources', default=SOURCES)
    ap.add_argument('--polyhaven', default=POLYHAVEN)
    ap.add_argument('--frames-per-source', type=int, default=6)
    ap.add_argument('--height', type=int, default=288)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--seed', type=int, default=20261030)
    ap.add_argument('--dev-fraction', type=float, default=0.05)
    ap.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--only-existing', action='store_true', help='skip sources whose file is missing (a staged subset)')
    ap.add_argument('--jobs-file', default='', help='the planned jobs as JSON (made with --plan-only where the registry is)')
    ap.add_argument('--plan-only', default='', help='write the planned jobs to this JSON and stop')
    a = ap.parse_args(argv)
    # the clip builder's plan (sources, exclusions, the leak assert), one frame-set job per source
    pa = argparse.Namespace(root=a.root, sources=a.sources, polyhaven=a.polyhaven, frames=1, height=a.height, width=a.width,
                            seed=a.seed, clips_per_hdri=1, clips_per_plate=1, clips_per_fxtd=1, recipe='v2', dev_fraction=a.dev_fraction)
    if a.jobs_file:
        jobs = json.loads(Path(a.jobs_file).read_text())
        for j in jobs:
            j['root'] = a.root
            if j['kind'] == 'equirect':
                j['path'] = gpath(j['path'], a.root)
    else:
        jobs, excl = plan(pa)
        if a.plan_only:
            Path(a.plan_only).write_text(json.dumps(jobs))
            print(json.dumps({'planned': len(jobs)}))
            return
    for j in jobs:
        j['frames_per_source'] = a.frames_per_source
    if a.only_existing:
        jobs = [j for j in jobs if Path(j['path'] if j['kind'] == 'equirect' else gpath(j['row']['hdr_path'], a.root)).exists()]
    if a.limit:
        jobs = jobs[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'sources': len(jobs)}), flush=True)
    t0 = time.time(); rows, X, y, W = [], [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (r, x, yy, ww) in enumerate(ex.map(frames_of, jobs, chunksize=2)):
            rows += r; X += x; y += yy; W += ww
            if (i + 1) % 25 == 0:
                print(f'{i + 1}/{len(jobs)} sources, {len(rows)} components, {time.time() - t0:.0f} s', flush=True)
    split = np.array(['dev' if dev_scene(r['scene_id'], a.dev_fraction) else 'train' for r in rows])
    np.savez(out / 'set.npz', X=np.array(X, np.float32), y=np.array(y, np.float32), w=np.array(W, np.float32), split=split)
    with (out / 'rows.jsonl').open('w') as f:
        for r, s in zip(rows, split):
            f.write(json.dumps({**r, 'split': str(s)}) + '\n')
    from rudra.genfill.brightness import FEATURES
    yv = np.array(y)
    summary = {'components': len(rows), 'sources': len({r['asset_id'] for r in rows}), 'scenes': len({r['scene_id'] for r in rows}),
               'train': int((split == 'train').sum()), 'dev': int((split == 'dev').sum()), 'features': FEATURES,
               'b_mean': float(yv.mean()) if len(yv) else None, 'b_median': float(np.median(yv)) if len(yv) else None,
               'registration_sha256': sha(a.registration), 'builder_sha256': sha(__file__),
               'seconds': round(time.time() - t0), 'release_authorized': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
