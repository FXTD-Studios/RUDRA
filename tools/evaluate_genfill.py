"""The registered gates of RUDRA's generative fill (claude_genfill_v1_20261027), 27 Oct 2026.

    python tools/evaluate_genfill.py --out <new dir> --base <Wan2.1-VACE-1.3B-diffusers> --lora <rudra_genfill.safetensors>
                                     [--scenes 24] [--frames 33] [--height 288] [--width 512] [--seed 20261029]

Truth clips the fill never trained on: Poly Haven HDRIs whose scene is in a development,
validation, holdout or fresh-grade set (the scenes the clip builder excludes), rendered as the
builder renders (camera moves, highlight-aimed and shadow-exposed), plus the FXTD plates of the
fresh grade set v2 scenes (a moving window). For each clip:

  master   the exact (analytic) inverse of the 8-bit ACES 2.0 SDR codes (layer 1), written as an
           EXR folder like the app's masters; then the fill runs on it (rudra.genfill.run.fill_master)
  truth    the scene-linear clip

Measured inside the fill regions (the masks the fill used), master vs filled, against truth:
  R   8-bit codes of the SDR view unchanged on every pixel of every frame
  H   mean |log2 Y - log2 Y_truth| in the highlight region; PU21 error at -3 and -6 EV (Y*203 nits)
  S   +3 EV: mean |code difference| of the view in the shadow region
  T   flicker excess: mean |frame-to-frame change of the region's mean log2 Y - the truth's| (EV)
  A   window joins: the frame-to-frame step at a join over the clip's median step
Gates (registered): H improves with a 95 % bootstrap CI over clips excluding 0, and PU21 at -3 and
-6 EV improves; S no worse than +2 %; T <= 0.02 EV; A <= 1.5; R exact. report.json says PASS/FAIL.
"""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_genfill_clips import (FRESH, POLYHAVEN, SOURCES, AP1_Y, equirect_path, gpath, plate_path,  # noqa: E402
                                       read_exr_rgb, sha)

NITS = 203.0


def eval_sources(a):
    from tools.build_aces_scale_crops import evaluation_scenes
    rows = [json.loads(l) for l in Path(gpath(a.sources, a.root)).read_text().splitlines() if l.strip()]
    excluded, _, fresh = evaluation_scenes(Path(gpath('G:/datasets/corpora', a.root)), Path(gpath(FRESH, a.root)))
    held = excluded | {r['scene_id'] for r in rows if r['split'] != 'train'}
    out = []
    for p in sorted(Path(gpath(a.polyhaven, a.root)).glob('*_2k.exr')):
        scene = 'ph:' + p.name[:-len('_2k.exr')]
        if scene in held:
            out.append({'kind': 'equirect', 'path': str(p), 'scene_id': scene})
    rng = np.random.default_rng(a.seed)
    rng.shuffle(out)
    out = out[a.ph_offset:a.ph_offset + a.scenes]          # v1.1: scenes v1 never looked at
    fresh_rows = {}
    pool = set(fresh) if a.fxtd_pool == 'fresh' else {   # 'extra': held FXTD scenes outside the fresh set
        r['scene_id'] for r in rows if r['scene_id'] in held and r.get('licence_source') == 'fxtd'} - set(fresh)
    for r in rows:
        if r['scene_id'] in pool and r.get('commercial_ok') and r['scene_id'] not in fresh_rows:
            fresh_rows[r['scene_id']] = r
    out += [{'kind': 'plate', 'row': r, 'scene_id': s}
            for s, r in sorted(fresh_rows.items())][a.fxtd_offset:a.fxtd_offset + a.fxtd_scenes]
    return out


def render_truth(src, kind, a, rng):
    from rudra.delivery.colorspace import convert
    F, h, w = a.frames, a.height, a.width
    if src['kind'] == 'equirect':
        pano = np.nan_to_num(np.maximum(convert(read_exr_rgb(src['path']), 'rec709', 'ap1'), 0), posinf=6e4).astype(np.float32)
        ref = equirect_path(pano, rng, F, h, w, kind)
        native = False
    else:
        import cv2
        from pipeline.run05_corpus import load_plate, NETWORK_PER_SCENE
        row = dict(src['row']); row['hdr_path'] = gpath(row['hdr_path'], a.root)
        plate = convert(load_plate(row) / NETWORK_PER_SCENE, 'rec2020', 'ap1').astype(np.float32)
        k = max(plate.shape[:2]) / 1920
        if k > 1:
            plate = cv2.resize(plate, (int(plate.shape[1] / k), int(plate.shape[0] / k)), interpolation=cv2.INTER_AREA)
        ref = plate_path(plate, rng, F, h, w)
        native = True
    ref = np.maximum(ref, 0).astype(np.float32)
    Y = ref[0] @ AP1_Y
    base = 0.0 if native else float(np.clip(np.log2(0.18 / max(float(np.median(Y[Y > 0])) if (Y > 0).any() else 0.18, 1e-6)), -10, 10))
    # the first offset (in half stops, from +1 up / -2 down) that gives the clip a region to fill:
    # >= 0.5 % of frame 0 at code >= 255 (v1: 250) for a highlight, <= 4 for a shadow; else the last tried
    from rudra.aces_view import ACES2View
    view = ACES2View(threads=4)
    offs = np.arange(1.0, 8.01, 0.5) if kind == 'highlight' else -np.arange(2.0, 8.01, 0.5)
    for off in offs:
        c = np.rint(np.clip(view.forward((ref[0] * np.float32(2.0 ** (base + off))).reshape(-1, 3)), 0, 1) * 255).max(-1)
        if ((c >= getattr(a, 'hi_target', 255)) if kind == 'highlight' else (c <= 4)).mean() >= 0.005:
            break
    return ref * np.float32(2.0 ** (base + off)), float(base + off)


def write_master_folder(folder: Path, frames_ap1):
    import OpenEXR
    folder.mkdir(parents=True)
    ap1 = (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)
    for i, img in enumerate(frames_ap1):
        ch = {c: np.ascontiguousarray(img[..., j], np.float32) for j, c in enumerate('RGB')}
        ch['rudra.reconstructed'] = np.zeros(img.shape[:2], np.float32)
        OpenEXR.File({'compression': OpenEXR.NO_COMPRESSION, 'type': OpenEXR.scanlineimage, 'chromaticities': ap1},
                     ch).write(str(folder / f'clip.{1001 + i:06d}.exr'))


def lum(x):
    return np.maximum(x @ AP1_Y, 1e-6)


def clip_metrics(truth, master, filled, gen, regions, view, joins):
    from rudra.delivery.bench import pu21_encode
    codes = lambda x: np.rint(np.clip(view.forward(x.reshape(-1, 3)).reshape(x.shape), 0, 1) * 255)
    r_ok = bool((codes(master) == codes(filled)).all())
    hi, lo = regions
    m = {'r_codes_kept': r_ok, 'generated_fraction': float((gen > 0).mean()),
         'highlight_px': int(hi.sum()), 'shadow_px': int(lo.sum())}
    Yt, Ym, Yf = lum(truth), lum(master), lum(filled)
    if hi.any():
        m['h_log_master'] = float(np.mean(np.abs(np.log2(Ym[hi]) - np.log2(Yt[hi]))))
        m['h_log_filled'] = float(np.mean(np.abs(np.log2(Yf[hi]) - np.log2(Yt[hi]))))
        for ev in (-3, -6):
            g = 2.0 ** ev * NITS
            m[f'pu21_{ev}_master'] = float(np.mean(np.abs(pu21_encode(Ym[hi] * g) - pu21_encode(Yt[hi] * g))))
            m[f'pu21_{ev}_filled'] = float(np.mean(np.abs(pu21_encode(Yf[hi] * g) - pu21_encode(Yt[hi] * g))))
    if lo.any():
        g = np.float32(8.0)
        ct, cm, cf = codes(truth * g), codes(master * g), codes(filled * g)
        m['s_plus3_master'] = float(np.mean(np.abs(cm - ct)[lo]))
        m['s_plus3_filled'] = float(np.mean(np.abs(cf - ct)[lo]))
    region = hi | lo
    if region.shape[0] > 1 and region.any():
        def series(Y):
            return np.array([np.log2(Y[t][region[t]]).mean() if region[t].any() else np.nan for t in range(Y.shape[0])])
        st, sf = series(Yt), series(Yf)
        d = np.abs(np.diff(sf) - np.diff(st))
        m['t_flicker_excess_ev'] = float(np.nanmean(d)) if np.isfinite(d).any() else 0.0
        steps = np.array([np.abs(np.log2(Yf[t + 1][region[t + 1]]).mean() - np.log2(Yf[t][region[t]]).mean())
                          if region[t].any() and region[t + 1].any() else np.nan for t in range(Yf.shape[0] - 1)])
        med = np.nanmedian(steps) if np.isfinite(steps).any() else 0.0
        js = [steps[j - 1] for j in joins if 0 < j <= len(steps) and np.isfinite(steps[j - 1])]
        m['a_join_ratio'] = float(max(js) / med) if js and med > 1e-6 else 1.0
    return m


def bootstrap_ci(d, seed=0, n=2000):
    d = np.asarray(d, np.float64)
    if len(d) < 2:
        return [float(d.mean()) if len(d) else float('nan')] * 2
    rng = np.random.default_rng(seed)
    means = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def gates(rows):
    rows = [x for x in rows if not x.get('secondary')]          # reused scenes are reported, never gated
    def pair(k):
        r = [x for x in rows if f'{k}_master' in x]
        return np.array([x[f'{k}_filled'] - x[f'{k}_master'] for x in r]), np.array([x[f'{k}_master'] for x in r])
    g = {}
    g['R'] = {'pass': all(x['r_codes_kept'] for x in rows)}
    dh, _ = pair('h_log')
    ci = bootstrap_ci(dh)
    g['H_log2'] = {'mean_delta': float(dh.mean()) if len(dh) else None, 'ci95': ci, 'clips': int(len(dh)),
                   'pass': bool(len(dh) and ci[1] < 0)}
    for ev in (-3, -6):
        d, _ = pair(f'pu21_{ev}')
        g[f'H_pu21_{ev}'] = {'mean_delta': float(d.mean()) if len(d) else None, 'pass': bool(len(d) and d.mean() < 0)}
    ds, base = pair('s_plus3')
    ci = bootstrap_ci(ds / np.maximum(base, 1e-6)) if len(ds) else [0, 0]
    g['S_plus3'] = {'mean_relative_delta': float((ds / np.maximum(base, 1e-6)).mean()) if len(ds) else None, 'ci95': ci,
                    'pass': bool(not len(ds) or ci[1] <= 0.02)}
    t = [x['t_flicker_excess_ev'] for x in rows if 't_flicker_excess_ev' in x]
    g['T'] = {'mean_ev': float(np.mean(t)) if t else None, 'max_ev': float(np.max(t)) if t else None,
              'pass': bool(not t or np.mean(t) <= 0.02)}
    j = [x['a_join_ratio'] for x in rows if 'a_join_ratio' in x]
    g['A_joins'] = {'max_ratio': float(np.max(j)) if j else None, 'pass': bool(not j or np.max(j) <= 1.5)}
    b = [x for x in rows if 'b_mae_est' in x]
    if b:                                                         # v3: the brightness estimator on its own
        db = np.array([x['b_mae_est'] - x['b_mae_master'] for x in b])
        ci = bootstrap_ci(db)
        over = float(np.mean([x['b_over_rate'] for x in b]))
        g['B_brightness'] = {'mean_delta': float(db.mean()), 'ci95': ci, 'clips': len(b), 'overshoot_rate_1stop': over,
                             'pass': bool(ci[1] < 0 and over <= 0.05)}
    g['all_pass'] = all(v['pass'] for v in g.values() if isinstance(v, dict))
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default='')
    ap.add_argument('--lora', default='')
    ap.add_argument('--root', default='')
    ap.add_argument('--sources', default=SOURCES)
    ap.add_argument('--polyhaven', default=POLYHAVEN)
    ap.add_argument('--scenes', type=int, default=24)
    ap.add_argument('--fxtd-scenes', type=int, default=10)
    ap.add_argument('--ph-offset', type=int, default=72, help='skip the first N shuffled held scenes (v1: 0-23, v1.1: 24-47, v2: 48-71)')
    ap.add_argument('--fxtd-offset', type=int, default=0)
    ap.add_argument('--fxtd-secondary', type=int, default=1,
                    help='v3: every held FXTD scene was evaluated before, so the FXTD clips are reported, not gated (0 = gate them)')
    ap.add_argument('--fxtd-pool', choices=('fresh', 'extra'), default='fresh',
                    help="fresh: the fresh grade set v2 (v1 used 0-7, v1.1 8-17); extra: the other held FXTD scenes (v2)")
    ap.add_argument('--v1-1', action='store_true', help='the v1.1 protocol (scenes 24-47 / fresh 8-17, no gain cap)')
    ap.add_argument('--v2', action='store_true', help='the v2 protocol (scenes 48-71 / the 4 extra FXTD scenes)')
    ap.add_argument('--brightness-model', default='', help='v3: the brightness estimator (.json); empty = the model\'s own brightness')
    ap.add_argument('--brightness-offset', type=float, default=0.0)
    ap.add_argument('--shape-gain', type=float, default=1.0)
    ap.add_argument('--v1', action='store_true', help='the v1 protocol: v1 scenes, v1 masks (region blend, feather 6, shadows 4)')
    ap.add_argument('--frames', type=int, default=33)
    ap.add_argument('--height', type=int, default=288)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--window', type=int, default=33)
    ap.add_argument('--overlap', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--registration', default='configs/experiments/claude_genfill_v3_20261030.json')
    ap.add_argument('--tiny', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    import torch
    from rudra.aces_view import ACES2View
    from rudra.genfill.masks import MaskSettings
    from rudra.genfill.model import FillSettings, WanFill
    from rudra.genfill.run import fill_master, plan_windows, read_frame
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    view = ACES2View(threads=8)
    if a.tiny:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
        from genfill_tiny import tiny_fill
        model = tiny_fill()
    else:
        dev = a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
        lora_dir = Path(a.lora).parent
        model = WanFill.load(a.base, a.lora, device=dev, prompt_cache=lora_dir / 'prompt_cache', text_device='cpu')
    s = FillSettings(steps=a.steps, seed=0)
    a.hi_target = 250 if a.v1 else 255
    if a.v1:
        a.ph_offset, a.fxtd_offset, a.fxtd_scenes, a.fxtd_pool = 0, 0, 8, 'fresh'
        ms = MaskSettings(tau_low=4, feather=6, blend_high=250, blend_low=4, max_gain_stops=0)
    elif a.v1_1:
        a.ph_offset, a.fxtd_offset, a.fxtd_scenes, a.fxtd_pool, a.fxtd_secondary = 24, 8, 10, 'fresh', 0
        ms = MaskSettings(max_gain_stops=0)
    elif a.v2:
        a.ph_offset, a.fxtd_offset, a.fxtd_scenes, a.fxtd_pool, a.fxtd_secondary = 48, 0, 10, 'extra', 0
        ms = MaskSettings()
    else:
        ms = MaskSettings()                                    # v2 defaults: blend at 255 only, shadows off, cap 3 stops
    if a.v1:
        a.fxtd_secondary = 0
    brightness, est = None, None
    if a.brightness_model:
        from rudra.genfill.brightness import BrightnessEstimator, BrightnessSettings
        from rudra.genfill.model import sha256_file
        est = BrightnessEstimator.load(a.brightness_model)
        brightness = BrightnessSettings(est, a.brightness_offset, a.shape_gain,
                                        identity={'estimator': a.brightness_model, 'estimator_sha256': sha256_file(a.brightness_model)})
    srcs = eval_sources(a)
    print(json.dumps({'sources': len(srcs), 'polyhaven': sum(x['kind'] == 'equirect' for x in srcs)}), flush=True)
    rows, t0 = [], time.time()
    joins = [n for (_, n, _) in plan_windows(a.frames, a.window, a.overlap)][1:]
    for i, src in enumerate(srcs):
        for kind in (('highlight', 'shadow') if src['kind'] == 'equirect' else ('highlight',)):
            rng = np.random.default_rng([a.seed, i, 0 if kind == 'highlight' else 1])
            truth, ev = render_truth(src, kind, a, rng)
            codes = np.rint(np.clip(view.forward(truth.reshape(-1, 3)).reshape(truth.shape), 0, 1) * 255).astype(np.float32)
            master = view.inverse((codes / 255).reshape(-1, 3)).reshape(truth.shape).astype(np.float32)
            name = f'{i:03d}_{kind}'
            mdir, fdir = out / 'clips' / name / 'master', out / 'clips' / name / 'filled'
            write_master_folder(mdir, master)
            fill_master(mdir, fdir, model, view, s, ms, a.window, a.overlap, a.width * a.height, brightness=brightness)
            frames = sorted(fdir.glob('*.exr'))
            filled = np.stack([read_frame(f)['ap1'] for f in frames])
            gen = np.stack([read_frame(f)['channels']['rudra.generated'] for f in frames])
            from rudra.genfill.masks import TemporalMask
            mx = codes.max(-1)
            # the registered v1.1 regions: H on the fully clipped pixels (the ones the fill may touch),
            # S on the shadow codes 0-4 (left alone by default: it must not be worse there)
            m = clip_metrics(truth, master, filled, gen, (mx >= 255, mx <= 4), view, joins)
            # v1's dilated regions (250 / 4), reported for comparison only
            tm = TemporalMask(MaskSettings(tau_low=4))
            regs = [tm(c.astype(np.uint8)) for c in codes]
            legacy = clip_metrics(truth, master, filled, gen, (np.stack([r[0] for r in regs]), np.stack([r[1] for r in regs])), view, joins)
            m.update({'v1region_' + k: v for k, v in legacy.items() if k.startswith(('h_', 'pu21', 's_'))})
            if est is not None:                                    # v3: the estimate per clipped component, frame 0
                from rudra.genfill.brightness import component_features, components, frame_features, log2_lum
                c0 = codes[0].astype(np.uint8)
                lab, ids = components(c0)
                if ids:
                    fr0 = frame_features(c0, master[0])
                    lt = log2_lum(np.minimum(truth[0], 6e4))
                    bt = np.array([max(0.0, float((lt[lab == c] - fr0['ly'][lab == c]).mean())) for c in ids])
                    be = est.predict(np.stack([component_features(c0, master[0], lab == c, fr0) for c in ids])) + a.brightness_offset
                    wc = np.sqrt([(lab == c).sum() for c in ids])
                    m.update({'b_mae_est': float(np.average(np.abs(be - bt), weights=wc)),
                              'b_mae_master': float(np.average(bt, weights=wc)),
                              'b_over_rate': float(np.average((be - bt) > 1.0, weights=wc))})
            m.update({'clip': name, 'scene_id': src['scene_id'], 'kind': kind, 'ev': ev,
                      'secondary': bool(a.fxtd_secondary and src['kind'] == 'plate')})
            rows.append(m)
            np.save(out / 'clips' / name / 'truth_f0.npy', truth[0])
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)
    g = gates(rows)
    rep = {'registration': a.registration, 'registration_sha256': sha(a.registration) if Path(a.registration).exists() else None,
           'lora': a.lora, 'lora_sha256': sha(a.lora) if a.lora and Path(a.lora).exists() else None,
           'model': model.identity, 'clips': len(rows), 'seconds': round(time.time() - t0), 'gates': g,
           'verdict': 'PASS' if g['all_pass'] else 'FAIL', 'rows': rows,
           'note': 'master = the exact (analytic) inverse; the learned layer is evaluated separately', 'release_authorized': False}
    (out / 'report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({'verdict': rep['verdict'], 'gates': g}, indent=1), flush=True)


if __name__ == '__main__':
    main()
