"""Fill a RUDRA master: a folder of EXR frames (or one EXR) -> a new folder, same names.

    python -m rudra.genfill.run --master <dir|file.exr> --out <new dir> --base <Wan2.1-VACE-1.3B-diffusers>
                                --lora <rudra_genfill.safetensors> [--tau-high 250] [--tau-low -1]
                                [--blend-high 255] [--blend-low 0] [--feather 0]
                                [--prompt-highlight "..."] [--prompt-shadow "..."] [--seed 0] [--steps 30]
                                [--window 33] [--overlap 8] [--max-area 399360] [--device cuda] [--json]

What it does, per window of frames (33, overlapping 8):
  1. The master's own SDR view (the ACES 2.0 SDR view, 8-bit codes) gives the fill regions:
     codes >= tau_high (clipped) and <= tau_low (crushed), dilated, steadied in time.
  2. The model sees the exact (analytic) inverse of those codes, log-encoded, at a working size
     (<= 832 x 480 area by default), and the region mask; the frames the previous window
     already filled come in as kept frames (mask 0), so windows join without a seam.
  3. The fill is decoded, brought to full size and blended into the master with
     composite.r_safe_blend: no 8-bit code of the master's SDR view changes (the R contract).
Pixels the fill does not touch are copied bit for bit (gamut, half/float as written).
Each frame gains `rudra.generated` (the blend weight: 1 = invented by the fill) and its
`rudra.reconstructed` becomes max(old, generated). genfill.json records everything.

--json: one JSON object per line on stdout (ready / progress / done / error) for the app.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import encoding
from .composite import r_safe_blend
from .masks import MaskSettings, TemporalMask, blend_core, feather

AP0_RED_X = 0.7347
MASK = 'rudra.reconstructed'
GENERATED = 'rudra.generated'


# -------------------------------------------------------------------------------- EXR I/O
def read_frame(path: Path) -> dict:
    import OpenEXR
    with OpenEXR.File(str(path), separate_channels=True) as f:
        header = dict(f.header())
        ch = {k: np.array(c.pixels) for k, c in f.channels().items()}
    for c in 'RGB':
        if c not in ch:
            raise ValueError(f'{path}: no {c} channel')
    chroma = header.get('chromaticities')
    gamut = 'ap0' if chroma is not None and abs(float(chroma[0]) - AP0_RED_X) < 1e-3 else 'ap1'
    rgb = np.stack([ch['R'], ch['G'], ch['B']], -1).astype(np.float32)
    if gamut == 'ap0':
        from rudra.delivery.colorspace import convert
        rgb_ap1 = convert(rgb, 'ap0', 'ap1')
    else:
        rgb_ap1 = rgb
    return {'path': path, 'header': header, 'channels': ch, 'gamut': gamut, 'ap1': rgb_ap1,
            'dtype': ch['R'].dtype}


def write_frame(path: Path, fr: dict, out_ap1: np.ndarray, alpha: np.ndarray, attrs: dict):
    """Writes the frame with the filled pixels replaced; every other pixel bit-identical."""
    import OpenEXR
    ch = {k: v.copy() for k, v in fr['channels'].items()}
    idx = alpha > 0
    if idx.any():
        rgb = out_ap1[idx]
        if fr['gamut'] == 'ap0':
            from rudra.delivery.colorspace import convert
            rgb = convert(rgb, 'ap1', 'ap0')
        for i, c in enumerate('RGB'):
            ch[c][idx] = rgb[:, i].astype(ch[c].dtype)
    old = ch.get(MASK)
    gen = alpha.astype(old.dtype if old is not None else np.float32)
    ch[MASK] = np.maximum(old, gen) if old is not None else gen
    ch[GENERATED] = gen
    header = {k: v for k, v in fr['header'].items() if k not in ('channels', 'dataWindow', 'displayWindow')}
    header['compression'] = OpenEXR.NO_COMPRESSION        # the app's reader takes uncompressed scanlines
    header['type'] = OpenEXR.scanlineimage
    for k, v in attrs.items():
        header[k] = str(v)
    # Written as <name>.part and renamed when whole (28 Oct 2026): a stop or a crash
    # never leaves a short frame under a finished name.
    part = path.with_name(path.name + '.part')
    OpenEXR.File(header, ch).write(str(part))
    os.replace(part, path)


# ------------------------------------------------------------------------------ planning
def plan_windows(n: int, window: int, overlap: int) -> list[tuple[int, int, int]]:
    """Frames 0..n-1 -> [(start, first_new, end)]: the window reads [start, end), writes
    [first_new, end). Every frame is written once; windows are at most `window` frames."""
    if window < 1 or (window - 1) % 4:
        raise ValueError('window must be 4k+1')
    if not 0 <= overlap < window:
        raise ValueError('overlap must be in [0, window)')
    out, written = [], 0
    while written < n:
        start = max(0, written - overlap) if written else 0
        end = min(n, start + window)
        out.append((start, written, end))
        written = end
    return out


def pad_len(k: int) -> int:
    """The smallest 4j+1 >= k (the VAE's frame count)."""
    return k if (k - 1) % 4 == 0 else k + (4 - (k - 1) % 4)


def working_size(h: int, w: int, max_area: int, multiple: int = 16) -> tuple[int, int]:
    """The largest size of the frame's aspect, multiples of `multiple`, within max_area (never upscaled
    past the next multiple)."""
    s = min(1.0, math.sqrt(max_area / float(h * w)))
    return (max(multiple, int(h * s // multiple) * multiple),
            max(multiple, int(w * s // multiple) * multiple))


def list_frames(master: Path, pattern: str = '*.exr') -> list[Path]:
    """The master's EXR frames; `pattern` (fnmatch) picks one render's frames
    in a folder that holds others too (the app's Nuke-style Write, 29 Oct 2026)."""
    import fnmatch
    if master.is_file():
        return [master]
    frames = sorted(p for p in master.iterdir() if p.suffix.lower() == '.exr' and not p.name.startswith('.')
                    and fnmatch.fnmatch(p.name.lower(), pattern.lower()))
    if not frames:
        raise ValueError(f'{master}: no EXR frames')
    return frames


# -------------------------------------------------------------------------------- review
REVIEW_STOPS = 3.0        # the review pictures are 3 stops down, so the invented highlights show
REVIEW_SIDE = 960
TEMPORAL_FLAG = 0.25      # stops of frame-to-frame change the fill adds beyond the master's own


def review_picture(view, ap1: np.ndarray, side: int = REVIEW_SIDE) -> np.ndarray:
    """An 8-bit RGB picture of scene light 3 stops down through the master's SDR view."""
    import cv2
    h, w = ap1.shape[:2]
    s = min(1.0, side / float(max(h, w)))
    small = cv2.resize(ap1, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else ap1
    codes = np.clip(view.forward(small.astype(np.float32) * 2.0 ** -REVIEW_STOPS), 0, 1)
    return np.rint(codes * 255).astype(np.uint8)


def write_review(out: Path, name: str, view, master_ap1: np.ndarray, fill_ap1: np.ndarray, alpha: np.ndarray):
    """review/<frame>.a.png (master), .b.png (fill), .mask.png (rudra.generated), for the app's A/B."""
    import cv2
    d = out / 'review'
    d.mkdir(exist_ok=True)
    stem = Path(name).stem
    a, b = review_picture(view, master_ap1), review_picture(view, fill_ap1)
    m = cv2.resize(np.clip(alpha, 0, 1).astype(np.float32), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    for suffix, img in (('a', a), ('b', b)):
        cv2.imwrite(str(d / f'{stem}.{suffix}.png'), img[:, :, ::-1])
    cv2.imwrite(str(d / f'{stem}.mask.png'), np.rint(m * 255).astype(np.uint8))


def temporal_excess(prev, cur) -> float | None:
    """Stops of change between this frame and the last on the pixels the fill touched, beyond the
    master's own change there (a flicker the fill added). prev/cur: (master_ap1, fill_ap1, alpha), small."""
    if prev is None:
        return None
    sel = (cur[2] > 0) | (prev[2] > 0)
    if not sel.any() or prev[0].shape != cur[0].shape:
        return None
    lum = lambda x: np.log2(np.maximum(x @ np.array([0.2722, 0.6741, 0.0537], np.float32), 1e-4))
    d_fill = np.abs(lum(cur[1]) - lum(prev[1]))[sel].mean()
    d_master = np.abs(lum(cur[0]) - lum(prev[0]))[sel].mean()
    return float(max(0.0, d_fill - d_master))


def small(x: np.ndarray, side: int = 256) -> np.ndarray:
    import cv2
    h, w = x.shape[:2]
    s = min(1.0, side / float(max(h, w)))
    return cv2.resize(x, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else x


# ----------------------------------------------------------------------------------- run
class Stopped(Exception):
    """The app asked to stop (a 'cancel' line on stdin)."""


def watch_stdin(flag: threading.Event):
    """The app's 'cancel' line on stdin sets the flag; the fill stops at the next step."""
    def run():
        for line in sys.stdin:
            if line.strip() == 'cancel':
                flag.set()
                return
    threading.Thread(target=run, daemon=True).start()


class Emitter:
    def __init__(self, json_lines: bool):
        self.json = json_lines

    def __call__(self, **o):
        if self.json:
            sys.stdout.write(json.dumps(o) + '\n'); sys.stdout.flush()
        elif o.get('event') in ('ready', 'done', 'error') or o.get('event') == 'frame':
            print({k: v for k, v in o.items() if k != 'event'} | {'_': o.get('event')}, flush=True)


def fill_master(master: Path, out: Path, model, view, s, ms: MaskSettings, window=33, overlap=8,
                max_area=832 * 480, emit=None, cancel=None, pattern='*.exr', brightness=None) -> dict:
    """brightness: a brightness.BrightnessSettings for the v3 split (shape from the model, brightness
    from the estimator plus the artist's offset); None keeps the model's own brightness (v1-v2)."""
    import cv2
    import torch
    emit = emit or (lambda **o: None)
    frames = list_frames(master, pattern)
    out.mkdir(parents=True, exist_ok=False)
    first = read_frame(frames[0])
    H, W = first['ap1'].shape[:2]
    h, w = working_size(H, W, max_area, model.multiple)
    plan = plan_windows(len(frames), window, overlap)
    tmask = TemporalMask(ms)
    prev_b = None
    masks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    gen_work: dict[int, np.ndarray] = {}            # encoded fill at working size, frames still needed as kept frames
    t0 = time.time()
    stats = {'frames': len(frames), 'windows': len(plan), 'generated_pixels': 0, 'region_pixels': 0,
             'windows_generated': 0, 'working_size': [h, w], 'size': [H, W]}
    per_frame = []          # the review: each frame's generated share and the flicker the fill adds
    prev_small = None
    bid = None if brightness is None else {**brightness.identity, 'offset': brightness.offset,
                                           'shape_gain': brightness.shape_gain, 'ema': brightness.ema}
    attrs = {'rudra:genfill': json.dumps({'model': model.identity, 'settings': asdict(s), 'mask': asdict(ms), 'brightness': bid})}
    for wi, (start, new, end) in enumerate(plan):
        if cancel and cancel():
            stats['cancelled'] = True
            break
        idxs = list(range(start, end))
        frs = {i: (first if i == 0 else read_frame(frames[i])) for i in idxs if i >= new or i not in gen_work}
        codes = {}
        for i in idxs:
            if i in frs:
                if frs[i]['ap1'].shape[:2] != (H, W):
                    raise ValueError(f'{frames[i]}: size differs from the first frame')
                codes[i] = np.rint(np.clip(view.forward(frs[i]['ap1']), 0, 1) * 255).astype(np.uint8)
                if i not in masks:
                    masks[i] = tmask(codes[i])
        need = any(masks[i][0].any() or masks[i][1].any() for i in range(new, end))
        n_pad = pad_len(len(idxs))
        if need:
            vid = np.zeros((n_pad, h, w, 3), np.float32)
            msk = np.zeros((n_pad, h, w), np.float32)
            for k, i in enumerate(idxs + [idxs[-1]] * (n_pad - len(idxs))):
                if i < new and i in gen_work:              # kept frame from the previous window
                    vid[k] = gen_work[i]
                    continue
                analytic = view.inverse(codes[i].astype(np.float32) / 255.0)
                vid[k] = cv2.resize(encoding.encode(analytic).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
                region = (masks[i][0] | masks[i][1]).astype(np.float32)
                msk[k] = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA) > 0
            video = torch.from_numpy(vid).permute(3, 0, 1, 2)[None]
            mt = torch.from_numpy(msk)[None, None]
            regions = {}
            for key, j in (('highlight', 0), ('shadow', 1)):
                r = np.zeros((n_pad, h, w), np.float32)
                for k, i in enumerate(idxs + [idxs[-1]] * (n_pad - len(idxs))):
                    if not (i < new and i in gen_work):
                        r[k] = cv2.resize(masks[i][j].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0
                regions[key] = torch.from_numpy(r)[None, None]
            def step(a, b):
                if cancel and cancel():
                    raise Stopped()
                emit(event='progress', window=wi + 1, windows=len(plan), step=a, steps=b)
            ctx = model.context(video, mt)
            try:
                z = model.denoise(ctx, s, regions, progress=step)
            except Stopped:
                stats['cancelled'] = True
                break
            dec = model.decode_latents(z)[0].permute(1, 2, 3, 0).cpu().numpy()      # (n_pad, h, w, 3) encoded
            stats['windows_generated'] += 1
        for k, i in enumerate(idxs):
            if i < new:
                continue
            fr = frs[i]
            region = masks[i][0] | masks[i][1]
            if need and region.any():
                gen_work[i] = dec[k]
                full = cv2.resize(dec[k], (W, H), interpolation=cv2.INTER_CUBIC)
                fill_lin = encoding.decode(full.astype(np.float32)).astype(np.float32)
                if brightness is not None:                         # v3: shape from the model, brightness from the estimator
                    from .brightness import compose, excess_map
                    b_map, lab = excess_map(codes[i], fr['ap1'], brightness.estimator, brightness.offset)
                    if prev_b is not None and prev_b.shape == b_map.shape and brightness.ema > 0:
                        both = np.isfinite(b_map) & np.isfinite(prev_b)
                        b_map[both] = brightness.ema * prev_b[both] + (1 - brightness.ema) * b_map[both]
                    prev_b = b_map
                    fill_lin = compose(fr['ap1'], fill_lin, b_map, lab, brightness.shape_gain)
                if ms.max_gain_stops > 0:                         # v2: never more than N stops over the master
                    lum = lambda x: np.maximum(x @ np.array([.2722287168, .6740817658, .0536895174], np.float32), 1e-6)
                    lim = lum(fr['ap1']) * np.float32(2.0 ** ms.max_gain_stops)
                    yf = lum(fill_lin)
                    fill_lin = fill_lin * np.where(yf > lim, lim / yf, 1.0)[..., None].astype(np.float32)
                core = blend_core(codes[i], ms) & region          # v1.1: only where the source knows nothing
                weight = feather(core, ms.feather) if ms.feather > 0 else core.astype(np.float32)
                out_ap1, alpha = r_safe_blend(fr['ap1'], fill_lin, weight, view)
            else:
                if need:
                    gen_work[i] = dec[k]
                out_ap1, alpha = fr['ap1'], np.zeros((H, W), np.float32)
            write_frame(out / frames[i].name, fr, out_ap1, alpha, attrs)
            write_review(out, frames[i].name, view, fr['ap1'], out_ap1, alpha)
            cur_small = (small(fr['ap1']), small(out_ap1), small(alpha.astype(np.float32)))
            temporal = temporal_excess(prev_small, cur_small)
            prev_small = cur_small
            gen_frac = float((alpha > 0).mean())
            per_frame.append({'name': frames[i].name, 'generated_fraction': gen_frac, 'temporal_excess_stops': temporal,
                              'flagged': bool(temporal is not None and temporal > TEMPORAL_FLAG)})
            stats['generated_pixels'] += int((alpha > 0).sum())
            stats['region_pixels'] += int(region.sum())
            emit(event='frame', index=i + 1, of=len(frames), file=str(out / frames[i].name),
                 generated_fraction=gen_frac, temporal_excess_stops=temporal, seconds=round(time.time() - t0, 1))
        for i in list(gen_work):                         # keep only what the next window reads
            if i < end - overlap:
                del gen_work[i]
        for i in list(masks):
            if i < start:
                del masks[i]
    stats['seconds'] = round(time.time() - t0, 1)
    stats['flagged_frames'] = sum(1 for f in per_frame if f['flagged'])
    stats['generated_fraction'] = stats['generated_pixels'] / float(max(1, len(frames) * H * W))
    report = {'master': str(master), 'out': str(out), 'model': model.identity, 'settings': asdict(s), 'brightness': bid,
              'mask': asdict(ms), 'window': window, 'overlap': overlap, 'max_area': max_area,
              'view': getattr(view, 'identity', None), 'stats': stats, 'frames': per_frame,
              'review': {'folder': 'review', 'pictures': '<frame>.a.png master, .b.png fill, .mask.png rudra.generated',
                         'exposure_stops': -REVIEW_STOPS, 'temporal_flag_stops': TEMPORAL_FLAG},
              'contract': 'no 8-bit code of the master SDR view changed (r_safe_blend); untouched pixels bit-identical',
              'release_authorized': False}
    (out / 'genfill.json').write_text(json.dumps(report, indent=1, default=str))
    prov = (master if master.is_dir() else master.parent) / 'provenance.json'
    if prov.exists():
        (out / 'master_provenance.json').write_bytes(prov.read_bytes())
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--master', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', required=True, help='Wan2.1-VACE-1.3B-diffusers folder')
    ap.add_argument('--lora', required=True, help='RUDRA genfill LoRA (.safetensors)')
    ap.add_argument('--prompt-cache', default='', help='prompt embeddings folder (default: beside the LoRA)')
    ap.add_argument('--tau-high', type=int, default=MaskSettings.tau_high)
    ap.add_argument('--tau-low', type=int, default=MaskSettings.tau_low)
    ap.add_argument('--prompt-highlight', default='')
    ap.add_argument('--prompt-shadow', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance', type=float, default=1.0)
    ap.add_argument('--window', type=int, default=33)
    ap.add_argument('--overlap', type=int, default=8)
    ap.add_argument('--max-area', type=int, default=832 * 480)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--ocio-threads', type=int, default=8)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--blend-high', type=int, default=MaskSettings.blend_high,
                    help='blend the fill only at max code >= this (v1.1: 255; v1 blended the whole region)')
    ap.add_argument('--blend-low', type=int, default=MaskSettings.blend_low, help='with shadows on: only at max code <= this')
    ap.add_argument('--feather', type=int, default=MaskSettings.feather, help='soft edge in pixels (v1: 6)')
    ap.add_argument('--max-gain-stops', type=float, default=MaskSettings.max_gain_stops,
                    help='v2: the fill at most this many stops brighter than the master (0 = no cap; v1/v1.1 had none)')
    ap.add_argument('--brightness-model', default='auto',
                    help="v3 brightness estimator (.json): 'auto' = brightness.json beside the LoRA if there, 'none' = the model's own brightness (v1-v2)")
    ap.add_argument('--brightness-offset', type=float, default=0.0, help="the artist's offset, stops, added to every clipped area's estimate")
    ap.add_argument('--shape-gain', type=float, default=1.0, help='how strongly the generated shape varies the brightness inside a clipped area')
    ap.add_argument('--glob', default='*.exr', help="the master folder's frames to fill (fnmatch, e.g. 'SH010.*.exr')")
    a = ap.parse_args(argv)
    emit = Emitter(a.json)
    try:
        import torch
        from rudra.aces_view import ACES2View
        from .model import FillSettings, WanFill
        out = Path(a.out)
        if out.exists():
            raise FileExistsError(f'{out} exists: the fill writes a new folder')
        device = a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
        t0 = time.time()
        cache = Path(a.prompt_cache) if a.prompt_cache else Path(a.lora).with_suffix('').parent / 'prompt_cache'
        model = WanFill.load(a.base, a.lora, device=device, prompt_cache=cache, text_device='cpu')  # umT5 (11 GB) never on a 16 GB GPU beside the model
        s = FillSettings(steps=a.steps, seed=a.seed, guidance=a.guidance,
                         prompt_highlight=a.prompt_highlight, prompt_shadow=a.prompt_shadow)
        for text in {s.prompt, s.prompt_highlight, s.prompt_shadow, s.negative} - {''}:
            model.prompts.get(text)
        model.prompts.release()
        view = ACES2View(threads=max(1, min(16, a.ocio_threads)))
        emit(event='ready', device=device, load_seconds=round(time.time() - t0, 1), model=model.identity)
        ms = MaskSettings(tau_high=a.tau_high, tau_low=a.tau_low, blend_high=a.blend_high,
                          blend_low=a.blend_low, feather=a.feather,
                          max_gain_stops=a.max_gain_stops).check()
        stop = threading.Event()
        if a.json:
            watch_stdin(stop)
        brightness = None
        bm = a.brightness_model
        if bm == 'auto':
            cand = Path(a.lora).parent / 'brightness.json'
            bm = str(cand) if cand.exists() else 'none'
        if bm != 'none':
            from .brightness import BrightnessEstimator, BrightnessSettings
            from .model import sha256_file
            brightness = BrightnessSettings(BrightnessEstimator.load(bm), a.brightness_offset, a.shape_gain,
                                            identity={'estimator': bm, 'estimator_sha256': sha256_file(bm)})
        rep = fill_master(Path(a.master), out, model, view, s, ms, a.window, a.overlap, a.max_area, emit,
                          cancel=stop.is_set, pattern=a.glob, brightness=brightness)
        emit(event='done', ok=True, out=str(out), **rep['stats'])
        return 0
    except Exception as e:  # noqa: BLE001 - every failure goes to the app as one line
        emit(event='error', ok=False, error=type(e).__name__, detail=str(e))
        if not a.json:
            raise
        return 2


if __name__ == '__main__':
    sys.exit(main())
