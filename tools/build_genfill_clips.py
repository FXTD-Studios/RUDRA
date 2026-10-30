"""Training clips for RUDRA's generative fill (layer 3, Wan2.1-VACE + LoRA), 27 Oct 2026.

Commercial-clean sources only, with the scale builder's exclusions (every development,
validation, holdout and fresh-grade scene; tools/build_aces_scale_crops.evaluation_scenes):

  Poly Haven HDRIs (CC0, G:/datasets/sources/polyhaven/*_2k.exr) whose scene `ph:<name>` is a
      commercial train scene of the run05 registry: perspective camera moves rendered from the
      equirectangular panorama (pan, tilt, zoom, a little handheld wobble), half of them aimed
      at the brightest source in the panorama, a quarter exposed down for crushed shadows.
  The other commercial train plates of the registry (FXTD own footage, CC0/authored renders):
      a moving 16:9 window (pan and zoom) over the plate.

Each clip: F frames (default 33) at W x H (default 512 x 288), scene-linear ACEScg truth,
exposure as the scale builder (FXTD native, others median luminance 0.18) plus an offset, the
exact ACES 2.0 SDR 100-nit view, 8-bit codes (optionally with temporally correlated noise
before the view), the analytic inverse of those codes (the model's condition) and the fill
mask (codes >= tau_high or <= tau_low, tau jittered per clip so the user's thresholds work).
A clip is kept when 0.2 % .. 60 % of its pixels are in the mask.

Two stages, one command:
  render  CPU workers: the clips above (numpy only, OCIO for the view)
  encode  GPU: the Wan VAE (float32) turns truth and condition into latents
          z0 (16 ch), context (inactive 16 + reactive 16 ch); the mask is kept as packed bits
          (its 64-channel latent form is rebuilt in training). Prompt embeddings of the base
          prompt are made once and stored in prompt_cache/.
Output: shards (z0, ctx fp16; mask bits), clips.jsonl, summary.json, a few preview clips
(sdr/truth .npy) for review. --render-only writes the rendered clips as .npz (no VAE).
"""
import argparse, hashlib, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCES = 'G:/datasets/corpora/rudra_run05_sources_20261022/sources.jsonl'
POLYHAVEN = 'G:/datasets/sources/polyhaven'
FRESH = 'G:/rudra_training/claude_release_candidate_20261026/fresh_fxtd_grade_v2_20261026/report.json'
COMMERCIAL = ('CC0', 'FXTD Studios own footage', 'CC0 sources and authored geometry; source receipts retained')
AP1_Y = np.array([.2722287168, .6740817658, .0536895174], np.float32)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def gpath(p, root):
    p = str(p).replace('\\', '/')
    return p if not root else root.rstrip('/') + '/' + p.split(':/', 1)[1]


def read_exr_rgb(path):
    import OpenEXR
    with OpenEXR.File(str(path), separate_channels=True) as f:
        ch = f.channels()
        return np.stack([np.asarray(ch[c].pixels, np.float32) for c in 'RGB'], -1)


def dev_scene(scene_id: str, fraction: float) -> bool:
    """A fixed few percent of the training scenes, held out for the validation loss."""
    return int(hashlib.sha256(('genfill-dev:' + scene_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


# ------------------------------------------------------------------------------- cameras
def _rot(yaw, pitch, roll):
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def render_equirect(pano, yaw, pitch, roll, fov_deg, h, w, ss=2):
    """Perspective view of an equirectangular panorama (radians; +pitch looks up), supersampled."""
    import cv2
    H, W = pano.shape[:2]
    hh, ww = h * ss, w * ss
    f = 0.5 * ww / math.tan(math.radians(fov_deg) / 2)
    xs = (np.arange(ww) + 0.5 - ww / 2) / f
    ys = (np.arange(hh) + 0.5 - hh / 2) / f
    X, Y = np.meshgrid(xs, ys)
    d = np.stack([X, -Y, np.ones_like(X)], -1)
    d = d @ _rot(yaw, -pitch, roll).T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    u = ((lon / (2 * math.pi) + 0.5) * W - 0.5).astype(np.float32)
    v = ((0.5 - lat / math.pi) * H - 0.5).astype(np.float32)
    big = cv2.remap(pano, u % W, np.clip(v, 0, H - 1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


def brightest_direction(pano):
    import cv2
    H, W = pano.shape[:2]
    small = cv2.resize(pano @ np.array([.2126, .7152, .0722], np.float32), (256, 128), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2)
    y, x = np.unravel_index(int(np.argmax(small)), small.shape)
    return ((x + 0.5) / 256 - 0.5) * 2 * math.pi, (0.5 - (y + 0.5) / 128) * math.pi


def equirect_path(pano, rng, F, h, w, kind):
    yaw0, pitch0 = (brightest_direction(pano) if kind == 'highlight'
                    else (rng.uniform(-math.pi, math.pi), math.radians(rng.uniform(-25, 20))))
    fov0 = rng.uniform(55, 85)
    if kind == 'highlight':                                    # the source in frame, not centred
        yaw0 += math.radians(rng.uniform(-0.3, 0.3) * fov0)
        pitch0 += math.radians(rng.uniform(-0.2, 0.2) * fov0 * h / w)
    pitch0 = float(np.clip(pitch0, math.radians(-60), math.radians(60)))
    dyaw, dpitch, dfov = (math.radians(rng.uniform(-0.8, 0.8)), math.radians(rng.uniform(-0.2, 0.2)),
                          rng.uniform(-0.3, 0.3))
    wob = rng.uniform(0, 0.15)                                 # degrees of handheld wobble
    ph = rng.uniform(0, 2 * math.pi, 3)
    frames = []
    for t in range(F):
        j = [math.radians(wob * math.sin(0.37 * t + ph[k]) + wob * 0.5 * math.sin(0.91 * t + 2 * ph[k])) for k in range(3)]
        frames.append(render_equirect(pano, yaw0 + dyaw * t + j[0], pitch0 + dpitch * t + j[1], j[2] * 0.3,
                                      float(np.clip(fov0 + dfov * t, 35, 100)), h, w))
    return np.stack(frames)


def plate_path(plate, rng, F, h, w):
    """A 16:9 window moving (pan, zoom) across a still plate."""
    import cv2
    H, W = plate.shape[:2]
    aspect = w / h
    cw0 = min(W, H * aspect) * rng.uniform(0.45, 0.9)
    zoom = rng.uniform(-0.1, 0.1)
    cx0, cy0 = rng.uniform(cw0 / 2, W - cw0 / 2), rng.uniform(cw0 / aspect / 2, H - cw0 / aspect / 2)
    dx, dy = rng.uniform(-0.1, 0.1) * W / F, rng.uniform(-0.05, 0.05) * H / F
    out = []
    for t in range(F):
        cw = cw0 * (1 + zoom * t / max(1, F - 1))
        ch = cw / aspect
        cx = float(np.clip(cx0 + dx * t, cw / 2, W - cw / 2))
        cy = float(np.clip(cy0 + dy * t, ch / 2, H - ch / 2))
        s = 2 * w / cw
        M = np.array([[s, 0, -s * (cx - cw / 2)], [0, s, -s * (cy - ch / 2)]], np.float32)
        big = cv2.warpAffine(plate, M, (2 * w, 2 * h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out.append(cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA))
    return np.stack(out)


# ------------------------------------------------------------------------------- one job
def make_clip(job):
    """One source -> up to `clips` training clips (dicts of arrays + metadata)."""
    from rudra.aces_view import ACES2View
    from rudra.delivery.colorspace import convert
    from rudra.genfill.masks import MaskSettings, TemporalMask
    import cv2
    cv2.setNumThreads(1)
    F, h, w = job['frames'], job['h'], job['w']
    rng = np.random.default_rng([job['seed'], int(hashlib.sha256(job['asset_id'].encode()).hexdigest()[:8], 16)])
    view = ACES2View(threads=1)
    if job['kind'] == 'equirect':
        pano = convert(read_exr_rgb(job['path']), 'rec709', 'ap1')
        pano = np.nan_to_num(np.maximum(pano, 0), posinf=6e4).astype(np.float32)
        native = False
    else:
        from pipeline.run05_corpus import load_plate, NETWORK_PER_SCENE
        row = dict(job['row']); row['hdr_path'] = gpath(row['hdr_path'], job['root'])
        plate = load_plate(row) / NETWORK_PER_SCENE
        plate = convert(plate, 'rec2020', 'ap1').astype(np.float32)
        k = max(plate.shape[:2]) / 1920
        if k > 1:
            plate = cv2.resize(plate, (int(plate.shape[1] / k), int(plate.shape[0] / k)), interpolation=cv2.INTER_AREA)
        native = row.get('licence_source') == 'fxtd'
    out, tries = [], 0
    while len(out) < job['clips'] and tries < 3 * job['clips']:
        tries += 1
        r = rng.uniform()
        if job.get('recipe', 'v2') == 'v1':
            kind = 'highlight' if r < 0.5 else 'shadow' if r < 0.75 else 'random'
        else:   # v2: 'broad' = large areas only just above the clip (the overshoot case of v1.1)
            kind = 'highlight' if r < 0.4 else 'broad' if r < 0.65 else 'shadow' if r < 0.8 else 'random'
        if job['kind'] == 'equirect':
            ref = equirect_path(pano, rng, F, h, w, 'random' if kind == 'broad' else kind)
        else:
            ref = plate_path(plate, rng, F, h, w)
        ref = np.maximum(ref, 0).astype(np.float32)
        Y = ref[0] @ AP1_Y
        base = 0.0 if native else float(np.clip(np.log2(0.18 / max(float(np.median(Y[Y > 0])) if (Y > 0).any() else 0.18, 1e-6)), -10, 10))
        if kind == 'broad':
            # the first half stop (from -2 up) at which >= 5 % of frame 0 is at code 255: wide, mild clipping
            off = None
            for o in np.arange(-2.0, 6.01, 0.5):
                c0 = np.rint(np.clip(view.forward((ref[0] * np.float32(2.0 ** (base + o))).reshape(-1, 3)), 0, 1) * 255).max(-1)
                if (c0 >= 255).mean() >= 0.05:
                    off = float(o)
                    break
            if off is None:
                continue
        else:
            off = {'highlight': rng.choice([0.0, 1.0, 2.0, 3.0]), 'shadow': rng.choice([-4.0, -3.0, -2.0]),
                   'random': rng.choice([-1.0, 0.0, 1.0, 2.0, 3.0])}[kind]
        ref *= np.float32(2.0 ** (base + off))
        lin = ref
        noise = bool(rng.uniform() < 0.3)
        if noise:                                            # temporally correlated, signal-dependent
            sigma = rng.uniform(0.002, 0.01)
            n = rng.standard_normal(ref.shape).astype(np.float32)
            for t in range(1, F):
                n[t] = 0.5 * n[t - 1] + math.sqrt(1 - 0.25) * n[t]
            lin = np.maximum(ref + sigma * np.sqrt(np.maximum(ref, 1e-4)) * n, 0)
        codes = np.rint(np.clip(view.forward(lin.reshape(-1, 3)).reshape(lin.shape), 0, 1) * 255).astype(np.uint8)
        ms = MaskSettings(tau_high=int(rng.integers(240, 255)), tau_low=int(rng.integers(1, 11)), dilate=2, ema=0.7)
        tm = TemporalMask(ms)
        mask = np.stack([np.logical_or(*tm(c)) for c in codes])
        frac = float(mask.mean())
        if not 0.002 <= frac <= 0.6:
            continue
        cond = view.inverse((codes.astype(np.float32) / 255).reshape(-1, 3)).reshape(codes.shape).astype(np.float32)
        mx = codes.max(-1)
        core = (mx >= 255) | (mx <= 0)                      # v2: where the source knows nothing
        ylog0 = np.log2(np.maximum(np.minimum(ref[0], 6e4) @ AP1_Y, 1e-6)).astype(np.float16)
        out.append({'sdr': codes, 'truth': np.minimum(ref, 6e4).astype(np.float16), 'cond': cond.astype(np.float16),
                    'mask': mask, 'core': core, 'ylog0': ylog0, 'meta': {'asset_id': job['asset_id'], 'scene_id': job['scene_id'], 'source': job['source'],
                                           'licence': job['licence'], 'kind': kind, 'ev': round(base + off, 4),
                                           'offset': off, 'noise': noise, 'tau_high': ms.tau_high, 'tau_low': ms.tau_low,
                                           'mask_fraction': round(frac, 5), 'core_fraction': round(float(core.mean()), 5),
                                           'split': job['split']}})
    return out


# ------------------------------------------------------------------------------- planning
def plan(a):
    from tools.build_aces_scale_crops import evaluation_scenes
    rows = [json.loads(l) for l in Path(gpath(a.sources, a.root)).read_text().splitlines() if l.strip()]
    excluded, excl_files, fresh = evaluation_scenes(Path(gpath('G:/datasets/corpora', a.root)), Path(gpath(FRESH, a.root)))
    non_train = {r['scene_id'] for r in rows if r['split'] != 'train'}
    train_ok = {r['scene_id'] for r in rows if r['split'] == 'train' and r.get('commercial_ok') is True}
    usable = train_ok - non_train - excluded
    jobs = []
    ph_dir = Path(gpath(a.polyhaven, a.root))
    for p in sorted(ph_dir.glob('*_2k.exr')):
        scene = 'ph:' + p.name[:-len('_2k.exr')]
        if scene in usable:
            jobs.append({'kind': 'equirect', 'path': str(p), 'asset_id': 'phhdri:' + p.name, 'scene_id': scene,
                         'source': 'polyhaven_hdri', 'licence': 'CC0', 'clips': a.clips_per_hdri})
    for r in rows:
        if (r['scene_id'] in usable and r['source_id'] != 'polyhaven_moves'
                and r.get('licence_licence') in COMMERCIAL):
            jobs.append({'kind': 'plate', 'row': r, 'asset_id': r['asset_id'], 'scene_id': r['scene_id'],
                         'source': r['source_id'], 'licence': r.get('licence_licence'),
                         'clips': a.clips_per_fxtd if (r.get('licence_source') == 'fxtd' and a.recipe == 'v2') else a.clips_per_plate})
    for j in jobs:
        j.update({'frames': a.frames, 'h': a.height, 'w': a.width, 'seed': a.seed, 'root': a.root, 'recipe': a.recipe,
                  'split': 'dev' if dev_scene(j['scene_id'], a.dev_fraction) else 'train'})
    assert not ({j['scene_id'] for j in jobs} & (excluded | non_train | set(fresh))), 'scene leak'
    return jobs, {'excluded_scenes': sorted(excluded), 'fresh_scenes': fresh, 'files': excl_files}


# --------------------------------------------------------------------------------- encode
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--registration', required=True)
    ap.add_argument('--base', default='', help='Wan2.1-VACE-1.3B-diffusers folder (the encode stage)')
    ap.add_argument('--root', default='', help='re-root G:/ paths (tests)')
    ap.add_argument('--sources', default=SOURCES)
    ap.add_argument('--polyhaven', default=POLYHAVEN)
    ap.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument('--seed', type=int, default=20261027)
    ap.add_argument('--frames', type=int, default=33)
    ap.add_argument('--height', type=int, default=288)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--clips-per-hdri', type=int, default=5)
    ap.add_argument('--clips-per-plate', type=int, default=1)
    ap.add_argument('--clips-per-fxtd', type=int, default=2, help='v2: FXTD own footage weighted up')
    ap.add_argument('--recipe', choices=('v1', 'v2'), default='v2',
                    help="v2 (29 Oct): 'broad' mildly clipped clips, FXTD x2, code-255/0 core masks and frame-0 truth luminance stored")
    ap.add_argument('--dev-fraction', type=float, default=0.03)
    ap.add_argument('--limit', type=int, default=0, help='first N jobs (smoke test)')
    ap.add_argument('--previews', type=int, default=12, help='clips kept as raw sdr/truth for review')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--render-only', action='store_true', help='write rendered clips (.npz), no VAE')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and stop')
    ap.add_argument('--tiny', action='store_true', help=argparse.SUPPRESS)   # tests: a random tiny VAE
    ap.add_argument('--resume', action='store_true',
                    help='continue an interrupted build in --out: sources already in clips.jsonl are skipped')
    ap.add_argument('--in-flight', type=int, default=0, help='rendered clips waiting for the encoder (default 2 x workers)')
    a = ap.parse_args()
    if (a.frames - 1) % 4 or a.height % 16 or a.width % 16:
        sys.exit('frames must be 4k+1, sizes multiples of 16')
    jobs, excl = plan(a)
    if a.limit:
        jobs = jobs[:a.limit]
    by = {}
    for j in jobs:
        by[j['source']] = by.get(j['source'], 0) + 1
    print(json.dumps({'jobs': len(jobs), 'by_source': by, 'dev_jobs': sum(j['split'] == 'dev' for j in jobs),
                      'excluded_scenes': len(excl['excluded_scenes'])}), flush=True)
    if a.dry_run:
        return
    out = Path(a.out)
    done_assets, n_shard0, prev0 = set(), 0, 0
    if a.resume and (out / 'clips.jsonl').exists():
        if (out / 'summary.json').exists():
            sys.exit(f'{out} is complete (summary.json): nothing to resume')
        old = [json.loads(l) for l in (out / 'clips.jsonl').read_text().splitlines() if l.strip()]
        done_assets = {m['asset_id'] for m in old}
        shards = [int(m['shard'].split('_')[1]) for m in old if 'shard' in m]
        n_shard0 = max(shards) + 1 if shards else 0
        prev0 = len(list(out.glob('preview_*.npz')))
        jobs = [j for j in jobs if j['asset_id'] not in done_assets]
        print(json.dumps({'resume': True, 'clips_kept': len(old), 'sources_done': len(done_assets),
                          'sources_left': len(jobs), 'next_shard': n_shard0}), flush=True)
        (out / f'builder_snapshot_resume{n_shard0:04d}.py').write_bytes(Path(__file__).read_bytes())
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (out / 'exclusions.json').write_text(json.dumps(excl, indent=1))
    if a.render_only:
        enc = None
    elif a.tiny:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
        from genfill_tiny import tiny_components
        enc = VaeEncoder.from_vae(tiny_components()[0])
    else:
        enc = VaeEncoder(a.base, a.device, out / 'prompt_cache')
    t0 = time.time()
    meta_f = (out / 'clips.jsonl').open('a' if done_assets else 'w')
    shard, n_shard, total, prev = [], n_shard0, 0, prev0
    SHARD = 32

    def flush():
        nonlocal shard, n_shard
        if not shard:
            return
        for split in ('train', 'dev'):
            part = [c for c in shard if c[3]['split'] == split]
            if not part:
                continue
            stem = out / f'{split}_{n_shard:04d}'
            np.save(f'{stem}_z0.npy', np.stack([c[0] for c in part]))
            np.save(f'{stem}_ctx.npy', np.stack([c[1] for c in part]))
            np.save(f'{stem}_mask.npy', np.stack([c[2] for c in part]))
            if part[0][4] is not None:
                np.save(f'{stem}_core.npy', np.stack([c[4] for c in part]))
                np.save(f'{stem}_ylog.npy', np.stack([c[5] for c in part]))
            for i, c in enumerate(part):
                meta_f.write(json.dumps({**c[3], 'shard': stem.name, 'index': i}) + '\n')
        shard = []; n_shard += 1

    def bounded(ex, fn, items, depth):
        """ex.map in order, but at most `depth` jobs submitted ahead: rendering is faster than the
        GPU encoder, and an unbounded map keeps every finished clip in memory (27 Oct: out of RAM)."""
        from collections import deque
        it, q = iter(items), deque()
        for _ in range(depth):
            try:
                q.append(ex.submit(fn, next(it)))
            except StopIteration:
                break
        while q:
            r = q.popleft().result()
            try:
                q.append(ex.submit(fn, next(it)))
            except StopIteration:
                pass
            yield r

    with ProcessPoolExecutor(a.workers) as ex:
        for i, clips in enumerate(bounded(ex, make_clip, jobs, a.in_flight or 2 * a.workers)):
            for c in clips:
                if a.render_only:
                    np.savez(out / f'clip_{total:06d}.npz', sdr=c['sdr'], truth=c['truth'], cond=c['cond'], mask=c['mask'])
                    meta_f.write(json.dumps({**c['meta'], 'file': f'clip_{total:06d}.npz'}) + '\n')
                else:
                    z0, ctx = enc.encode_clip(c)
                    shard.append((z0, ctx, np.packbits(c['mask'], axis=None), c['meta'],
                                  np.packbits(c['core'], axis=None) if 'core' in c else None, c.get('ylog0')))
                    if prev < a.previews:
                        np.savez_compressed(out / f'preview_{prev:02d}.npz', sdr=c['sdr'], truth=c['truth'], mask=c['mask'])
                        prev += 1
                    if len(shard) == SHARD:
                        flush()
                total += 1
            del clips
            if (i + 1) % 20 == 0:
                print(f'{i + 1}/{len(jobs)} sources, {total} clips, {time.time() - t0:.0f} s', flush=True)
    flush(); meta_f.close()
    meta = [json.loads(l) for l in (out / 'clips.jsonl').read_text().splitlines()]
    count = lambda k: dict(sorted({str(v): sum(1 for m in meta if m[k] == v) for v in {m[k] for m in meta}}.items()))
    summary = {'clips': len(meta), 'recipe': a.recipe, 'has_core': a.recipe == 'v2' and not a.render_only, 'frames': a.frames, 'size': [a.height, a.width], 'sources_used': len({m['asset_id'] for m in meta}),
               'scenes_used': len({m['scene_id'] for m in meta}), 'by_source': count('source'), 'by_kind': count('kind'),
               'by_split': count('split'), 'by_licence': count('licence'),
               'commercial_clean': all(m['licence'] in COMMERCIAL for m in meta),
               'mask_fraction_mean': float(np.mean([m['mask_fraction'] for m in meta])) if meta else 0.0,
               'encoded': not a.render_only, 'base': a.base, 'registration_sha256': sha(a.registration),
               'builder_sha256': sha(__file__), 'seconds': round(time.time() - t0), 'release_authorized': False}
    if not a.render_only:
        from rudra.genfill.model import BASE_PROMPT
        summary['base_prompt'] = BASE_PROMPT
        summary['vae'] = 'float32'
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


class VaeEncoder:
    """The Wan VAE (float32) for the encode stage: truth -> z0; condition -> [inactive, reactive]."""

    def __init__(self, base, device, cache: Path):
        import torch
        from diffusers import AutoencoderKLWan
        from rudra.genfill.model import BASE_PROMPT, PromptBank
        if not base:
            sys.exit('--base (the Wan2.1-VACE-1.3B-diffusers folder) is needed to encode; or --render-only')
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        pb = PromptBank(base, cache, device=device)
        pb.get(BASE_PROMPT)
        pb.release()
        self.vae = AutoencoderKLWan.from_pretrained(base, subfolder='vae', torch_dtype=torch.float32).to(device).eval()
        self._setup(device)

    @classmethod
    def from_vae(cls, vae, device='cpu'):
        self = cls.__new__(cls)
        self.vae = vae.eval()
        self._setup(device)
        return self

    def _setup(self, device):
        import torch
        z = self.vae.config.z_dim
        self.device = device
        self.mean = torch.tensor(self.vae.config.latents_mean, dtype=torch.float32).view(1, z, 1, 1, 1).to(device)
        self.inv_std = (1.0 / torch.tensor(self.vae.config.latents_std, dtype=torch.float32)).view(1, z, 1, 1, 1).to(device)

    def encode_video(self, v: np.ndarray):
        """(F, H, W, 3) in [-1, 1] -> normalised latents (16, f, H/8, W/8) float32 tensor."""
        import torch
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(v, np.float32)).permute(3, 0, 1, 2)[None].to(self.device)
            return ((self.vae.encode(t).latent_dist.mode().float() - self.mean) * self.inv_std)[0]

    def encode_clip(self, clip):
        import torch
        from rudra.genfill import encoding
        m = clip['mask'].astype(np.float32)[..., None]
        z0 = self.encode_video(encoding.encode(clip['truth'].astype(np.float32)))
        c = encoding.encode(clip['cond'].astype(np.float32))
        ctx = torch.cat([self.encode_video(c * (1 - m)), self.encode_video(c * m)], 0)
        return z0.half().cpu().numpy(), ctx.half().cpu().numpy()


if __name__ == '__main__':
    main()
