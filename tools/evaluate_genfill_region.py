"""Gate H of the region re-render (research_genfill_v4_20261109), 10 Nov 2026.

    python tools/evaluate_genfill_region.py --out <new dir> --base <Wan2.1-VACE-1.3B-diffusers> --lora <rudra_genfill.safetensors>
                                            [--finish] [--scenes 24] [--ph-offset 96] [--fxtd-scenes 10]

Truth clips the fill never trained on, made exactly as the v4 clip builder makes its 'region' clips
(tools/build_genfill_clips.region_clip: a display-referred tone curve parks a bright area under
white; the truth is the scene). Sources as tools/evaluate_genfill.py: the shuffled held Poly Haven
list (seed 20261029), entries from --ph-offset (v3's gates used 72-95, so 96 on were never
evaluated: gated), and the fresh FXTD plates (all evaluated before: reported, not gated).

For each clip:
  source   the clip's 8-bit SDR codes, as a folder of PNG stills
  master   the BT.1886 reading of those codes (code ^ 2.4, the model's condition), an EXR folder
  filled   rudra.genfill.run.fill_master with the region re-render on (tau_high the clip's own,
           shadows off, feather 8, steps 30, seed 0), as registered ('raw'); with --finish also
           with the render-time finish (--rerender-hue source and the given colour, level, smooth)
Measured where the source's max code >= tau_high, master and filled against the truth:
  H   PU21 error at -3 EV (Y * 203 nits; the gate) and at -6 EV; mean |log2 Y - log2 Y_truth|
  T   flicker excess: mean |frame-to-frame change of the region's mean log2 Y - the truth's| (EV)
  level  the region's median log2 Y over the truth's (stops): how far off the brightness is
Gate (registered): PU21 at -3 EV better than the BT.1886 condition's, the 95 % bootstrap CI over the
gated clips excluding 0. report.json has one verdict per variant.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_genfill_clips import FRESH, POLYHAVEN, SOURCES, AP1_Y, gpath, read_exr_rgb, region_clip, sha  # noqa: E402
from tools.evaluate_genfill import NITS, bootstrap_ci, write_master_folder  # noqa: E402

SHUFFLE_SEED = 20261029      # the held-scene order every genfill evaluation has used


def eval_sources(a):
    """-> (sources, the number of held Poly Haven scenes). As tools/evaluate_genfill.eval_sources."""
    from tools.build_aces_scale_crops import evaluation_scenes
    rows = [json.loads(l) for l in Path(gpath(a.sources, a.root)).read_text().splitlines() if l.strip()]
    excluded, _, fresh = evaluation_scenes(Path(gpath('G:/datasets/corpora', a.root)), Path(gpath(FRESH, a.root)))
    held = excluded | {r['scene_id'] for r in rows if r['split'] != 'train'}
    ph = []
    for p in sorted(Path(gpath(a.polyhaven, a.root)).glob('*_2k.exr')):
        scene = 'ph:' + p.name[:-len('_2k.exr')]
        if scene in held:
            ph.append({'kind': 'equirect', 'path': str(p), 'scene_id': scene, 'secondary': False})
    np.random.default_rng(SHUFFLE_SEED).shuffle(ph)
    total = len(ph)
    out = ph[a.ph_offset:a.ph_offset + a.scenes]
    fresh_rows = {}
    for r in rows:
        if r['scene_id'] in set(fresh) and r.get('commercial_ok') and r['scene_id'] not in fresh_rows:
            fresh_rows[r['scene_id']] = r
    out += [{'kind': 'plate', 'row': r, 'scene_id': s, 'secondary': True} for s, r in sorted(fresh_rows.items())][:a.fxtd_scenes]
    return out, total


def load_source(src, a):
    """-> (scene-linear ACEScg picture, native). As tools/build_genfill_clips.make_clip."""
    from rudra.delivery.colorspace import convert
    if src['kind'] == 'equirect':
        pano = convert(read_exr_rgb(src['path']), 'rec709', 'ap1')
        return np.nan_to_num(np.maximum(pano, 0), posinf=6e4).astype(np.float32), False
    import cv2
    from pipeline.run05_corpus import load_plate, NETWORK_PER_SCENE
    row = dict(src['row']); row['hdr_path'] = gpath(row['hdr_path'], a.root)
    plate = convert(load_plate(row) / NETWORK_PER_SCENE, 'rec2020', 'ap1').astype(np.float32)
    k = max(plate.shape[:2]) / 1920
    if k > 1:
        plate = cv2.resize(plate, (int(plate.shape[1] / k), int(plate.shape[0] / k)), interpolation=cv2.INTER_AREA)
    return plate, row.get('licence_source') == 'fxtd'


def lum(x):
    return np.maximum(np.asarray(x, np.float32) @ AP1_Y, 1e-6)


def region_metrics(truth, picture, region) -> dict:
    """picture (the master or a fill) against the truth, inside region (F, h, w) bool."""
    from rudra.delivery.bench import pu21_encode
    Yt, Yp = lum(truth), lum(picture)
    m = {'h_log': float(np.mean(np.abs(np.log2(Yp[region]) - np.log2(Yt[region])))),
         'level_stops': float(np.median(np.log2(Yp[region])) - np.median(np.log2(Yt[region])))}
    for ev in (-3, -6):
        g = 2.0 ** ev * NITS
        m[f'pu21_{ev}'] = float(np.mean(np.abs(pu21_encode(Yp[region] * g) - pu21_encode(Yt[region] * g))))
    def series(Y):
        return np.array([np.log2(Y[t][region[t]]).mean() if region[t].any() else np.nan for t in range(Y.shape[0])])
    d = np.abs(np.diff(series(Yp)) - np.diff(series(Yt)))
    m['t_flicker_excess_ev'] = float(np.nanmean(d)) if np.isfinite(d).any() else 0.0
    return m


def gate(rows, variant) -> dict:
    r = [x for x in rows if not x['secondary'] and variant in x]
    d = np.array([x[variant]['pu21_-3'] - x['master']['pu21_-3'] for x in r])
    ci = bootstrap_ci(d)
    mean = lambda k: float(np.mean([x[variant][k] for x in r])) if r else None   # noqa: E731
    return {'clips': int(len(d)), 'pu21_-3_mean_delta': float(d.mean()) if len(d) else None, 'ci95': ci,
            'better_in': int((d < 0).sum()),
            'pu21_-6_mean_delta': float(np.mean([x[variant]['pu21_-6'] - x['master']['pu21_-6'] for x in r])) if r else None,
            'h_log_mean_delta': float(np.mean([x[variant]['h_log'] - x['master']['h_log'] for x in r])) if r else None,
            'level_stops_mean': mean('level_stops'), 't_flicker_excess_mean_ev': mean('t_flicker_excess_ev'),
            'pass': bool(len(d) >= 2 and ci[1] < 0)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--lora', required=True)
    ap.add_argument('--root', default='')
    ap.add_argument('--sources', default=SOURCES)
    ap.add_argument('--polyhaven', default=POLYHAVEN)
    ap.add_argument('--scenes', type=int, default=24)
    ap.add_argument('--ph-offset', type=int, default=96, help='skip the first N shuffled held scenes (0-95 were used by v1-v3)')
    ap.add_argument('--fxtd-scenes', type=int, default=10, help='fresh FXTD plates, reported and not gated')
    ap.add_argument('--finish', action='store_true', help='also run the render-time finish (composite.finish_rerender)')
    ap.add_argument('--no-raw', action='store_true', help='with --finish: only the finish (the registered inference was measured before)')
    ap.add_argument('--colour', type=float, default=0.3)
    ap.add_argument('--level', type=float, default=3.3)
    ap.add_argument('--smooth', type=float, default=0.8)
    ap.add_argument('--dip', type=float, default=0.0, help='with --finish: --rerender-dip (the app uses 0.3 since 12 Nov 2026)')
    ap.add_argument('--still', type=float, default=0.0, help='with --finish: --rerender-still')
    ap.add_argument('--frames', type=int, default=33)
    ap.add_argument('--height', type=int, default=288)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--window', type=int, default=33)
    ap.add_argument('--overlap', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20261110)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--registration', default='configs/experiments/research_genfill_v4_20261109.json')
    a = ap.parse_args(argv)
    import cv2
    import torch
    from rudra.aces_view import ACES2View
    from rudra.genfill.masks import MaskSettings
    from rudra.genfill.model import FillSettings, WanFill
    from rudra.genfill.run import Rerender, fill_master, read_frame
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    srcs, held = eval_sources(a)
    print(json.dumps({'held_polyhaven': held, 'sources': len(srcs), 'gated': sum(not x['secondary'] for x in srcs)}), flush=True)
    view = ACES2View(threads=8)
    dev = a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    model = WanFill.load(a.base, a.lora, device=dev, prompt_cache=Path(a.lora).parent / 'prompt_cache', text_device='cpu')
    s = FillSettings(steps=a.steps, seed=0)
    variants = {} if (a.no_raw and a.finish) else {'raw': lambda src: Rerender(src, 'bt1886')}
    if a.finish:
        variants['finish'] = lambda src: Rerender(src, 'bt1886', 'source', a.colour, a.level, a.smooth, 0, a.dip, a.still)
    rows, t0 = [], time.time()
    for i, src in enumerate(srcs):
        picture, native = load_source(src, a)
        job = {'kind': src['kind'], 'asset_id': src['scene_id'], 'scene_id': src['scene_id'], 'source': 'evaluation',
               'licence': '', 'split': 'evaluation'}
        clip = None
        for attempt in range(6):                             # a view with no bright area to park: another one
            clip = region_clip(job, np.random.default_rng([a.seed, i, attempt]), picture, native, a.frames, a.height, a.width)
            if clip is not None:
                break
        if clip is None:
            print(json.dumps({'scene_id': src['scene_id'], 'skipped': 'no region in 6 views'}), flush=True)
            continue
        codes = clip['sdr']
        truth, cond = clip['truth'].astype(np.float32), clip['cond'].astype(np.float32)
        tau = int(clip['meta']['tau_high'])
        region = codes.max(-1) >= tau
        if not region.any():
            continue
        name = f'{i:03d}_region'
        cdir = out / 'clips' / name
        (cdir / 'source').mkdir(parents=True)
        for t, c in enumerate(codes):
            cv2.imwrite(str(cdir / 'source' / f'clip.{1001 + t:06d}.png'), np.ascontiguousarray(c[..., ::-1]))
        write_master_folder(cdir / 'master', cond)
        ms = MaskSettings(tau_high=tau, tau_low=-1, feather=8)
        m = {'clip': name, 'scene_id': src['scene_id'], 'secondary': bool(src['secondary']), 'curve': clip['meta']['curve'],
             'tau_high': tau, 'region_fraction': float(region.mean()), 'master': region_metrics(truth, cond, region)}
        for v, make in variants.items():
            fdir = cdir / f'filled_{v}'
            fill_master(cdir / 'master', fdir, model, view, s, ms, a.window, a.overlap, a.width * a.height,
                        rerender=make(str(cdir / 'source')))
            filled = np.stack([read_frame(f)['ap1'] for f in sorted(fdir.glob('*.exr'))])
            m[v] = region_metrics(truth, filled, region)
        np.save(cdir / 'truth_f0.npy', truth[0])
        rows.append(m)
        print(json.dumps({k: ({kk: round(vv, 4) for kk, vv in val.items()} if isinstance(val, dict) else
                              round(val, 4) if isinstance(val, float) else val) for k, val in m.items()}), flush=True)
    g = {v: gate(rows, v) for v in variants}
    rep = {'registration': a.registration, 'registration_sha256': sha(a.registration) if Path(a.registration).exists() else None,
           'lora': a.lora, 'lora_sha256': sha(a.lora) if Path(a.lora).exists() else None, 'model': model.identity,
           'held_polyhaven': held, 'ph_offset': a.ph_offset, 'clips': len(rows), 'seconds': round(time.time() - t0),
           'finish': {'colour': a.colour, 'level': a.level, 'smooth': a.smooth, 'dip': a.dip, 'still': a.still} if a.finish else None,
           'gates': g, 'verdict': {v: 'PASS' if g[v]['pass'] else 'FAIL' for v in g}, 'rows': rows,
           'note': 'master = the BT.1886 reading of the codes (the condition); region = source max code >= tau_high',
           'release_authorized': False}
    (out / 'report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({'verdict': rep['verdict'], 'gates': g}, indent=1), flush=True)


if __name__ == '__main__':
    main()
