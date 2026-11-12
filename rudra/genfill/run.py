"""Fill a RUDRA master: a folder of EXR frames (or one EXR) -> a new folder, same names.

    python -m rudra.genfill.run --master <dir|file.exr> --out <new dir> --base <Wan2.1-VACE-1.3B-diffusers>
                                --lora <rudra_genfill.safetensors> [--tau-high 250] [--tau-low -1]
                                [--blend-high 255] [--blend-low 0] [--feather 0]
                                [--prompt-highlight "..."] [--prompt-shadow "..."] [--seed 0] [--steps 30]
                                [--window 33] [--overlap 8] [--max-area 399360] [--focus frame|region|areas]
                                [--device auto] [--json]

What it does, per window of frames (33, overlapping 8):
  1. The master's own SDR view (the ACES 2.0 SDR view, 8-bit codes) gives the fill regions:
     codes >= tau_high (clipped) and <= tau_low (crushed), dilated, steadied in time.
  2. The model sees the exact (analytic) inverse of those codes, log-encoded, at a working size
     (<= 832 x 480 area by default), and the region mask; the frames the previous window
     already filled come in as kept frames (mask 0), so windows join without a seam.
     --focus region: the model sees only the box around what the shot fills (one box for the
     whole shot), so the working size is spent there: more detail in a sky that is part of the frame.
     --focus areas (8 Nov 2026): one box per group of clipped areas (a lamp, a window, a sky), each
     drawn on its own at up to the working size, so two lamps at opposite edges of a 4K frame are
     each drawn at full resolution instead of sharing one box nearly as big as the frame.
  3. The fill is decoded, brought to full size and blended into the master with
     composite.r_safe_blend: no 8-bit code of the master's SDR view changes (the R contract).
Pixels the fill does not touch are copied bit for bit (gamut, half/float as written).
Each frame gains `rudra.generated` (the blend weight: 1 = invented by the fill) and its
`rudra.reconstructed` becomes max(old, generated). genfill.json records everything.

--json: one JSON object per line on stdout (ready / progress / done / error) for the app.

--region-rerender (v4, 9 Nov 2026; configs/experiments/research_genfill_v4_20261109.json): the other
contract. The regions come from the SOURCE's own codes (--source: the movie or the folder of stills
the master was made from; the master's ACES round trip is not the source on a BT.1886 master), the
model's condition matches the route (--condition bt1886: code ^ 2.4, the app's default analytic
reading; aces: the ACES 2.0 inverse), the fill is blended plainly inside the whole mask with a
feather (composite.plain_blend): the SDR codes inside the mask are NOT kept, outside it every pixel
is the master's bit for bit. No brightness estimator, no gain cap: the model's own level. For a
LoRA trained for it (v4); with v1-v3 it draws nothing useful (measured 9 Nov).
--rerender-hue source [--rerender-colour <0..1>], --rerender-level <x white>, --rerender-smooth <0..1>
(10 Nov 2026, after the v4 gates on 09V; composite.finish_rerender): the fill takes the source's
colour (with a share of the model's own), its level is <x white> with the model's detail on it,
crossfaded into the master by the
source's code (the master at tau_high, the whole fill at code 253), and it follows the previous frame. With a level the fill acts
only where the source is near white in every channel (composite.neutral_weight): a saturated
highlight is left to the master. All off by default: the model's own colour and level.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import encoding
from .composite import RERENDER_HUES, finish_rerender, neutral_weight, plain_blend, r_safe_blend, white_scale
from .masks import MaskSettings, TemporalMask, blend_core, feather


@dataclass
class Rerender:
    """--region-rerender: where the codes come from and what the model is conditioned on."""
    source: str            # the movie, or the folder of stills, the master was made from
    condition: str = 'bt1886'   # 'bt1886' (code ^ 2.4) or 'aces' (the ACES 2.0 inverse)
    hue: str = 'model'     # 'source': the fill's luminance with the source's colour (composite.finish_rerender)
    colour: float = 0.0    # with hue 'source': the share of the model's own colour kept, 0..1
    level: float = 0.0     # > 0: the level where the source is clipped, in the master's SDR whites; 0 keeps the model's own
    smooth: float = 0.0    # > 0: the fill's log luminance follows the previous frame's (EMA share kept)
    start: int = 0         # the source frame the master's first frame was made from (the app's preview of one frame)
    dip: float = 0.0       # > 0: the fill's fine detail at most this many stops under its surroundings (finish_rerender)
    still: float = 0.0     # > 0: the share of the previous frame kept where the source holds still (finish_rerender)
    CONDITIONS = ('bt1886', 'aces')


class SourceCodes:
    """The source's 8-bit codes per frame index, (H, W, 3) uint8 at the master's size, decoded as the
    engine reads them (tools/measure_flicker: a movie through ffmpeg, stills through OpenCV)."""

    def __init__(self, source: Path, count: int, size, start: int = 0):
        from tools.measure_flicker import decode_frames, decode_movie, FRAME_EXT
        # start (12 Nov 2026): a master that is part of the shot (the app previews the frame at the
        # playhead) begins at that source frame; index i here is still the master's own.
        if start < 0:
            raise ValueError('the source start frame cannot be negative')
        self.source, self.size, self.start = Path(source), tuple(size), int(start)
        self.count = count + self.start
        self.movie = self.source.is_file()
        if not self.movie:
            n = len([p for p in self.source.iterdir() if p.suffix.lower() in FRAME_EXT])
            if n < self.count:
                raise ValueError(f'{source}: {n} frames, the master needs {self.count}')
        self._decode_frames, self._decode_movie = decode_frames, decode_movie
        self._cache = {}

    def __getitem__(self, i: int) -> np.ndarray:
        i = i + self.start
        if i not in self._cache:
            block = 16 if self.movie else 1
            first = (i // block) * block
            frames = (self._decode_movie if self.movie else self._decode_frames)(self.source, first, min(block, self.count - first))
            if not frames:
                raise ValueError(f'{self.source}: frame {i} could not be decoded')
            for k, f in enumerate(frames):
                self._cache[first + k] = self._fit(f)
            for k in [k for k in self._cache if k < first - block]:
                del self._cache[k]
        return self._cache[i]

    def _fit(self, codes01: np.ndarray) -> np.ndarray:
        import cv2
        if codes01.shape[:2] != self.size:
            codes01 = cv2.resize(codes01, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        return np.rint(np.clip(codes01, 0, 1) * 255).astype(np.uint8)


def condition_of(codes: np.ndarray, condition: str, view) -> np.ndarray:
    """The model's condition for 8-bit codes (H, W, 3): ACEScg float32."""
    if condition == 'aces':
        return view.inverse(codes.astype(np.float32) / 255.0)
    from rudra.delivery.colorspace import convert
    return convert(np.power(codes.astype(np.float32) / 255.0, 2.4), 'rec709', 'ap1').astype(np.float32)

AP0_RED_X = 0.7347
MASK = 'rudra.reconstructed'
GENERATED = 'rudra.generated'
# RUDRA 1.1 (5 Nov 2026): rudra.generated split by region; the two always sum to it exactly.
GENERATED_PARTS = ('rudra.generated.highlight', 'rudra.generated.shadow')


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


def write_frame(path: Path, fr: dict, out_ap1: np.ndarray, alpha: np.ndarray, attrs: dict,
                highlight: np.ndarray | None = None):
    """Writes the frame with the filled pixels replaced; every other pixel bit-identical.
    highlight: the frame's highlight fill region (bool); then rudra.generated.highlight / .shadow
    split rudra.generated between the regions (one of them is it, the other 0, per pixel)."""
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
    if highlight is not None:
        hi = np.asarray(highlight, bool)
        ch[GENERATED_PARTS[0]] = np.where(hi, gen, 0).astype(gen.dtype)
        ch[GENERATED_PARTS[1]] = np.where(hi, 0, gen).astype(gen.dtype)
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


FOCUS = ('frame', 'region', 'areas')   # what the model sees: the whole frame, the box around what it fills
                                       # (7 Nov 2026), or one box per group of areas (8 Nov 2026)
FOCUS_MARGIN = 0.15           # context around the box, a share of its own size each side ...
FOCUS_MIN_MARGIN = 48         # ... and at least this many pixels
FOCUS_MIN_SIDE = 256          # a thin strip gets this much of the frame around it, at least
FOCUS_MAX_SHARE = 0.6         # a box this much of the frame or more: the whole frame is as good
FOCUS_COVER = 0.9             # the box takes the largest areas until this share of the shot's mask
FOCUS_MAX_BOXES = 8           # --focus areas: at most this many boxes (closest ones merge first); each is
                              # one more model run per window, so the time grows with the count


def _grow(a: int, b: int, size: int, extra: int) -> tuple[int, int]:
    """[a, b) with context each side (FOCUS_MARGIN of its size, at least FOCUS_MIN_MARGIN + extra),
    at least FOCUS_MIN_SIDE long, shifted to stay inside [0, size)."""
    m = max(FOCUS_MIN_MARGIN + extra, int(np.ceil(FOCUS_MARGIN * (b - a))) + extra)
    a, b = a - m, b + m
    short = FOCUS_MIN_SIDE - (b - a)
    if short > 0:
        a, b = a - short // 2, b + short - short // 2
    if a < 0:
        a, b = 0, b - a
    if b > size:
        a, b = max(0, a - (b - size)), size
    return max(0, a), min(size, b)


def focus_box(union: np.ndarray, extra: int = 0, cover: float = FOCUS_COVER) -> tuple[int, int, int, int] | None:
    """The shot's fill masks, OR-ed over every frame (H, W bool) -> (y0, y1, x0, x1): a box with
    context around the largest connected areas, taken until `cover` of the mask is in them, the same
    box for every frame so the fill moves as one picture. Specks far from them (a lamp, a glint) do
    not stretch it to the whole frame; the fill leaves them to the master (genfill.json says how
    much). None when there is nothing to fill or the box is most of the frame (the whole frame is
    then no worse). `extra`: pixels the blend reaches past the masks (the feather), inside the box."""
    import cv2
    H, W = union.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(union.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    total, got = float(st[1:, 4].sum()), 0.0
    y0, y1, x0, x1 = H, 0, W, 0
    for k in np.argsort(-st[1:, 4], kind='stable') + 1:
        x, y, w, h, a = (int(v) for v in st[k])
        y0, y1, x0, x1 = min(y0, y), max(y1, y + h), min(x0, x), max(x1, x + w)
        got += a
        if got >= cover * total:
            break
    y0, y1 = _grow(y0, y1, H, extra)
    x0, x1 = _grow(x0, x1, W, extra)
    if (y1 - y0) * (x1 - x0) >= FOCUS_MAX_SHARE * H * W:
        return None
    return y0, y1, x0, x1


def _overlap(a, b) -> bool:
    """Two (y0, y1, x0, x1) boxes share a pixel."""
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _merge(a, b):
    """The smallest box holding both."""
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


def _area(b) -> int:
    return (b[1] - b[0]) * (b[3] - b[2])


def focus_boxes(union: np.ndarray, extra: int = 0, cover: float = FOCUS_COVER,
                max_boxes: int = FOCUS_MAX_BOXES) -> list[tuple[int, int, int, int]] | None:
    """--focus areas (8 Nov 2026): the shot's fill masks, OR-ed over every frame -> boxes that do not
    overlap, one per group of connected areas, each with context (as focus_box), the same boxes for
    every frame. The largest areas are taken until `cover` of the mask is in them; boxes that overlap
    merge, and while there are more than `max_boxes` the pair whose merged box grows least merges.
    None when there is nothing to fill or the boxes cover most of the frame (the whole frame is then
    no worse). Each box is drawn on its own at up to the working size."""
    import cv2
    H, W = union.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(union.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    total, got, boxes = float(st[1:, 4].sum()), 0.0, []
    for k in np.argsort(-st[1:, 4], kind='stable') + 1:
        x, y, w, h, a = (int(v) for v in st[k])
        boxes.append(_grow(y, y + h, H, extra) + _grow(x, x + w, W, extra))
        got += a
        if got >= cover * total:
            break
    changed = True
    while changed:                                   # overlapping boxes become one
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _overlap(boxes[i], boxes[j]):
                    boxes[i] = _merge(boxes[i], boxes.pop(j))
                    changed = True
                    break
            if changed:
                break
        if not changed and len(boxes) > max_boxes:   # too many: merge the cheapest pair
            i, j = min(((i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))),
                       key=lambda p: _area(_merge(boxes[p[0]], boxes[p[1]])) - _area(boxes[p[0]]) - _area(boxes[p[1]]))
            boxes[i] = _merge(boxes[i], boxes.pop(j))
            changed = True
    if sum(_area(b) for b in boxes) >= FOCUS_MAX_SHARE * H * W:
        return None
    return sorted(boxes)


def list_frames(master: Path, pattern: str = '*.exr') -> list[Path]:
    """The master's EXR frames; `pattern` (fnmatch) picks one render's frames
    in a folder that holds others too (the app's Nuke-style Write, 29 Oct 2026)."""
    import fnmatch
    if master.is_file():
        return [master]
    frames = sorted(p for p in master.iterdir() if p.suffix.lower() == '.exr' and not p.name.startswith('.')
                    and not p.name.lower().endswith('.mattes.exr')   # RUDRA 1.1: a master's mattes are not frames
                    and fnmatch.fnmatch(p.name.lower(), pattern.lower()))
    if not frames:
        raise ValueError(f'{master}: no EXR frames')
    return frames


# -------------------------------------------------------------------------------- review
REVIEW_STOPS = 3.0        # the review pictures are 3 stops down, so the invented highlights show
REVIEW_SIDE = 960
TEMPORAL_FLAG = 0.25      # stops of frame-to-frame change the fill adds beyond the master's own
FALLBACKS = ('none', 'spikes')   # RUDRA 1.1: what a flickering fill frame falls back to (fill_master)


def _over(temporal) -> bool:
    return bool(temporal is not None and temporal > TEMPORAL_FLAG)


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
                max_area=832 * 480, emit=None, cancel=None, pattern='*.exr', brightness=None,
                region=None, fallback: str = 'none', focus: str = 'frame', rerender: Rerender | None = None) -> dict:
    """brightness: a brightness.BrightnessSettings for the v3 split (shape from the model, brightness
    from the estimator plus the artist's offset); None keeps the model's own brightness (v1-v2).
    region: a rudra.delivery.region.ReconstructionRegion (RUDRA 1.1). On, the fill (and its prompts)
    acts only inside it: the model's fill regions are cut to it and the blend weight is scaled by it,
    so every pixel outside is the master's, bit for bit. None or off: unchanged.
    fallback (RUDRA 1.1 priority 5): 'spikes' writes the master's own frame (no fill) for a frame whose
    fill jumps more than TEMPORAL_FLAG stops beyond the master's own change both INTO it and OUT of it
    (a one-frame flicker; a fill appearing or ending jumps on one side only and is kept). 'none' keeps
    every fill (v3). Either way a fill that is not finite falls back. Each frame is written once the
    next shows whether it spiked; every fallback is in genfill.json ('fell_back').
    focus (7 Nov 2026): 'frame' (v3) gives the model the whole frame at max_area; 'region' gives it
    only the box around everything the shot fills (focus_box), so the same max_area spends its
    pixels where the fill is: a sky in a door a third of the frame wide is drawn at about twice the
    detail. One box for the whole shot (a first pass over the masks), so the fill stays one moving
    picture; outside the box nothing is generated and every pixel stays the master's.
    'areas' (8 Nov 2026): one box per group of areas (focus_boxes), each drawn on its own, so far-apart
    highlights in a 4K frame are each drawn at up to the working size; one model run per box and window."""
    if fallback not in FALLBACKS:
        raise ValueError(f'fallback must be one of {", ".join(FALLBACKS)}')
    if focus not in FOCUS:
        raise ValueError(f'focus must be one of {", ".join(FOCUS)}')
    from rudra.delivery.region import fill_region_weight
    recon_region = region      # `region` is reused below for each frame's fill mask
    import cv2
    import torch
    emit = emit or (lambda **o: None)
    frames = list_frames(master, pattern)
    out.mkdir(parents=True, exist_ok=False)
    first = read_frame(frames[0])
    H, W = first['ap1'].shape[:2]
    region_on = recon_region is not None and recon_region.enabled
    # v4: the source's codes, the matching condition, the whole mask blended plainly, the model's own level.
    source_codes = None
    if rerender is not None:
        if rerender.condition not in Rerender.CONDITIONS:
            raise ValueError(f'condition must be one of {", ".join(Rerender.CONDITIONS)}')
        source_codes = SourceCodes(Path(rerender.source), len(frames), (H, W), rerender.start)
        ms = MaskSettings(**{**asdict(ms), 'blend_high': ms.tau_high, 'blend_low': ms.tau_low, 'max_gain_stops': 0.0})
        brightness = None
        if rerender.hue not in RERENDER_HUES:
            raise ValueError(f'hue must be one of {", ".join(RERENDER_HUES)}')
        if rerender.level < 0 or not 0.0 <= rerender.smooth < 1.0 or not 0.0 <= rerender.colour <= 1.0:
            raise ValueError('level must be >= 0, smooth in [0, 1) and colour in [0, 1]')
    finish_state: dict = {}                          # v4 finish: the previous frame's fill
    finish_white = 1.0
    if rerender is not None and rerender.level > 0:  # the shot's SDR white: up to 9 frames along it
        picks = sorted({int(k) for k in np.linspace(0, len(frames) - 1, min(9, len(frames)))})
        seen = [white_scale((first if k == 0 else read_frame(frames[k]))['ap1'], source_codes[k]) for k in picks]
        seen = [x for x in seen if x]
        finish_white = float(np.median(seen)) if seen else 1.0
    codes_of = (lambda i, fr: source_codes[i]) if source_codes is not None else \
        (lambda i, fr: np.rint(np.clip(view.forward(fr['ap1']), 0, 1) * 255).astype(np.uint8))
    box, boxes, inside, outside_share = None, None, None, 0.0
    if focus in ('region', 'areas'):      # a first pass: the masks the fill will use, OR-ed over the shot
        pre, union = TemporalMask(ms), np.zeros((H, W), bool)
        for k, f in enumerate(frames):
            if cancel and cancel():
                break
            fr0 = first if k == 0 else read_frame(f)
            c0 = codes_of(k, fr0)
            hi, lo = pre(c0)
            m0 = hi | lo
            if region_on:
                m0 &= fill_region_weight(c0, recon_region) > 0
            union |= m0
            emit(event='focus', index=k + 1, of=len(frames))
        if focus == 'region':
            box = focus_box(union, extra=ms.feather)
            boxes = [box] if box else None
        else:
            boxes = focus_boxes(union, extra=ms.feather)
        if boxes:
            inside = np.zeros((H, W), bool)
            for b in boxes:
                inside[b[0]:b[1], b[2]:b[3]] = True
            outside_share = float((union & ~inside).sum()) / max(1.0, float(union.sum()))
    # One lane per box the model draws (the whole frame when there is none); each has its own working
    # size and its own kept frames from the previous window.
    lanes = []
    for b in (boxes or [(0, H, 0, W)]):
        lh, lw = working_size(b[1] - b[0], b[3] - b[2], max_area, model.multiple)
        lanes.append({'box': b, 'crop': (slice(b[0], b[1]), slice(b[2], b[3])), 'h': lh, 'w': lw, 'gen_work': {}})
    y0, y1, x0, x1 = lanes[0]['box']
    h, w = lanes[0]['h'], lanes[0]['w']
    plan = plan_windows(len(frames), window, overlap)
    tmask = TemporalMask(ms)
    prev_b = None
    masks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rweights: dict[int, np.ndarray] = {}             # the reconstruction region's weight, per frame (when on)
    t0 = time.time()
    pending = None                                   # RUDRA 1.1: the frame waiting for the next one

    def commit(e, spike_reason):
        """Write one frame: its fill, or the master's own frame when it falls back."""
        fr, out_ap1, alpha = e['fr'], e['out'], e['alpha']
        reason = e['reason'] or spike_reason
        if reason and alpha.any():
            out_ap1, alpha = fr['ap1'], np.zeros_like(alpha)
        name = frames[e['i']].name
        write_frame(out / name, fr, out_ap1, alpha, attrs, highlight=e['highlight'])
        write_review(out, name, view, fr['ap1'], out_ap1, alpha)
        gen_frac = float((alpha > 0).mean())
        temporal = e['temporal']
        per_frame.append({'name': name, 'generated_fraction': gen_frac, 'temporal_excess_stops': temporal,
                          'flagged': _over(temporal), 'fell_back': reason})
        stats['generated_pixels'] += int((alpha > 0).sum())
        stats['region_pixels'] += e['region_px']
        emit(event='frame', index=e['i'] + 1, of=len(frames), file=str(out / name), generated_fraction=gen_frac,
             temporal_excess_stops=temporal, fell_back=reason, seconds=round(time.time() - t0, 1))

    stats = {'frames': len(frames), 'windows': len(plan), 'generated_pixels': 0, 'region_pixels': 0,
             'windows_generated': 0, 'working_size': [h, w], 'size': [H, W],
             'focus': {'mode': focus, 'box': [y0, y1, x0, x1] if box else None,
                       'mask_outside_box': round(outside_share, 4),   # left to the master (specks far away)
                       # the model's pixels per frame pixel, against the whole frame at the same max_area
                       'detail_gain': round((w / float(x1 - x0)) / (working_size(H, W, max_area, model.multiple)[1] / float(W)), 3)}}
    if rerender is not None and rerender.level > 0:
        stats['rerender_white'] = round(finish_white, 4)
    if focus == 'areas':
        base_gain = working_size(H, W, max_area, model.multiple)[1] / float(W)
        stats['focus']['box'] = None
        stats['focus']['boxes'] = [{'box': list(L['box']), 'working_size': [L['h'], L['w']],
                                    'detail_gain': round((L['w'] / float(L['box'][3] - L['box'][2])) / base_gain, 3)}
                                   for L in lanes] if boxes else []
        stats['focus']['detail_gain'] = min(b['detail_gain'] for b in stats['focus']['boxes']) if boxes else 1.0
    per_frame = []          # the review: each frame's generated share and the flicker the fill adds
    prev_small = None
    bid = None if brightness is None else {**brightness.identity, 'offset': brightness.offset,
                                           'shape_gain': brightness.shape_gain, 'ema': brightness.ema}
    contract = ('region re-render (v4): inside the mask the SDR codes are NOT kept (plain_blend); outside it every pixel is the master\'s, bit-identical'
                if rerender is not None else 'no 8-bit code of the master SDR view changed (r_safe_blend); untouched pixels bit-identical')
    attrs = {'rudra:genfill': json.dumps({'model': model.identity, 'settings': asdict(s), 'mask': asdict(ms), 'brightness': bid,
                                          **({'region': asdict(recon_region)} if region_on else {}),
                                          **({'rerender': asdict(rerender), 'contract': contract} if rerender is not None else {})})}
    for wi, (start, new, end) in enumerate(plan):
        if cancel and cancel():
            stats['cancelled'] = True
            break
        idxs = list(range(start, end))
        frs = {i: (first if i == 0 else read_frame(frames[i])) for i in idxs
               if i >= new or any(i not in L['gen_work'] for L in lanes)}
        codes = {}
        for i in idxs:
            if i in frs:
                if frs[i]['ap1'].shape[:2] != (H, W):
                    raise ValueError(f'{frames[i]}: size differs from the first frame')
                codes[i] = codes_of(i, frs[i])
                if i not in masks:
                    hi, lo = tmask(codes[i])
                    if inside is not None:                # --focus region: nothing is drawn outside the box
                        hi, lo = hi & inside, lo & inside
                    if region_on:
                        rw = fill_region_weight(codes[i], recon_region)
                        rweights[i] = rw
                        hi, lo = hi & (rw > 0), lo & (rw > 0)
                    masks[i] = (hi, lo)
        n_pad = pad_len(len(idxs))
        padded = idxs + [idxs[-1]] * (n_pad - len(idxs))
        decs = {}                                        # lane -> decoded fill (n_pad, h, w, 3), encoded
        for li, L in enumerate(lanes):
            lcrop, lh, lw, lgen = L['crop'], L['h'], L['w'], L['gen_work']
            if not any(masks[i][0][lcrop].any() or masks[i][1][lcrop].any() for i in range(new, end)):
                continue
            vid = np.zeros((n_pad, lh, lw, 3), np.float32)
            msk = np.zeros((n_pad, lh, lw), np.float32)
            for k, i in enumerate(padded):
                if i < new and i in lgen:                  # kept frame from the previous window
                    vid[k] = lgen[i]
                    continue
                analytic = condition_of(codes[i], rerender.condition if rerender else 'aces', view)
                vid[k] = cv2.resize(encoding.encode(analytic[lcrop]).astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
                region = (masks[i][0] | masks[i][1])[lcrop].astype(np.float32)
                msk[k] = cv2.resize(region, (lw, lh), interpolation=cv2.INTER_AREA) > 0
            video = torch.from_numpy(vid).permute(3, 0, 1, 2)[None]
            mt = torch.from_numpy(msk)[None, None]
            regions = {}
            for key, j in (('highlight', 0), ('shadow', 1)):
                r = np.zeros((n_pad, lh, lw), np.float32)
                for k, i in enumerate(padded):
                    if not (i < new and i in lgen):
                        r[k] = cv2.resize(masks[i][j][lcrop].astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA) > 0
                regions[key] = torch.from_numpy(r)[None, None]

            def step(a, b, li=li):
                if cancel and cancel():
                    raise Stopped()
                # one bar per window: every lane's steps in turn
                emit(event='progress', window=wi + 1, windows=len(plan), step=li * b + a, steps=b * len(lanes))
            ctx = model.context(video, mt)
            try:
                z = model.denoise(ctx, s, regions, progress=step)
            except Stopped:
                stats['cancelled'] = True
                break
            decs[li] = model.decode_latents(z)[0].permute(1, 2, 3, 0).cpu().numpy()
            stats['windows_generated'] += 1
        if stats.get('cancelled'):
            break
        need = bool(decs)
        for k, i in enumerate(idxs):
            if i < new:
                continue
            fr = frs[i]
            region = masks[i][0] | masks[i][1]
            if need and region.any():
                fill_lin = fr['ap1'].astype(np.float32).copy()   # outside every box the master itself
                for li, dec in decs.items():
                    L = lanes[li]
                    L['gen_work'][i] = dec[k]
                    (a0, a1, b0, b1), lcrop = L['box'], L['crop']
                    part = cv2.resize(dec[k], (b1 - b0, a1 - a0), interpolation=cv2.INTER_CUBIC)
                    fill_lin[lcrop] = encoding.decode(part.astype(np.float32)).astype(np.float32)
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
                if region_on:                                     # RUDRA 1.1: only inside the region
                    weight = (weight * rweights[i]).astype(np.float32)
                if rerender is not None:                          # v4: the whole mask, the codes inside it not kept
                    # The dilated, feathered mask reaches dark pixels beside the region (a figure against
                    # the sky): the weight follows the source's codes, a ramp of 8 below tau_high.
                    top = codes[i].max(-1).astype(np.float32)
                    ramp = np.clip((top - (ms.tau_high - 8)) / 8.0, 0.0, 1.0)
                    w_plain = (weight * ramp * ramp * (3 - 2 * ramp)).astype(np.float32)
                    if rerender.level > 0:                     # only where the source is near white in every channel
                        w_plain = (w_plain * neutral_weight(codes[i])).astype(np.float32)
                    if rerender.hue != 'model' or rerender.level > 0 or rerender.smooth > 0 or rerender.still > 0:
                        fill_lin = finish_rerender(fill_lin, fr['ap1'], condition_of(codes[i], 'bt1886', view), codes[i], w_plain,
                                                   rerender.hue, rerender.colour, rerender.level, rerender.smooth,
                                                   finish_white, ms.tau_high, finish_state, dip=rerender.dip, still=rerender.still)
                    out_ap1, alpha = plain_blend(fr['ap1'], fill_lin, w_plain)
                else:
                    out_ap1, alpha = r_safe_blend(fr['ap1'], fill_lin, weight, view)
            else:
                for li, dec in decs.items():
                    lanes[li]['gen_work'][i] = dec[k]
                out_ap1, alpha = fr['ap1'], np.zeros((H, W), np.float32)
            reason = None if np.isfinite(out_ap1).all() else 'non_finite'
            if reason:                                     # RUDRA 1.1: never write a broken fill
                out_ap1, alpha = fr['ap1'], np.zeros((H, W), np.float32)
            cur_small = (small(fr['ap1']), small(out_ap1), small(alpha.astype(np.float32)))
            temporal = temporal_excess(prev_small, cur_small)   # measured along the fills as made
            prev_small = cur_small
            entry = {'i': i, 'fr': fr, 'out': out_ap1, 'alpha': alpha, 'highlight': masks[i][0],
                     'region_px': int(region.sum()), 'temporal': temporal, 'reason': reason}
            if pending is not None:
                spike = (fallback == 'spikes' and pending['alpha'].any() and _over(pending['temporal'])
                         and _over(temporal))
                commit(pending, 'flicker' if spike else None)
            pending = entry
        for L in lanes:                                  # keep only what the next window reads
            for i in list(L['gen_work']):
                if i < end - overlap:
                    del L['gen_work'][i]
        for i in list(masks):
            if i < start:
                del masks[i]
                rweights.pop(i, None)
    if pending is not None:                              # the last frame: nothing after it to spike against
        commit(pending, None)
    stats['seconds'] = round(time.time() - t0, 1)
    stats['flagged_frames'] = sum(1 for f in per_frame if f['flagged'])
    stats['fallback_frames'] = sum(1 for f in per_frame if f['fell_back'])
    stats['generated_fraction'] = stats['generated_pixels'] / float(max(1, len(frames) * H * W))
    report = {'master': str(master), 'out': str(out), 'model': model.identity, 'settings': asdict(s), 'brightness': bid,
              'mask': asdict(ms), 'region': asdict(recon_region) if region_on else None, 'fallback': fallback, 'window': window, 'overlap': overlap, 'max_area': max_area,
              'view': getattr(view, 'identity', None), 'stats': stats, 'frames': per_frame,
              'review': {'folder': 'review', 'pictures': '<frame>.a.png master, .b.png fill, .mask.png rudra.generated',
                         'exposure_stops': -REVIEW_STOPS, 'temporal_flag_stops': TEMPORAL_FLAG},
              'contract': contract, 'rerender': asdict(rerender) if rerender is not None else None,
              'release_authorized': False}
    (out / 'genfill.json').write_text(json.dumps(report, indent=1, default=str))
    prov = (master if master.is_dir() else master.parent) / 'provenance.json'
    if prov.exists():
        (out / 'master_provenance.json').write_bytes(prov.read_bytes())
    return report


def checked_prompt_strength(v) -> float:
    """--prompt-strength / a request's prompt_strength: 0..2, finite."""
    x = float(v)
    if not (math.isfinite(x) and 0.0 <= x <= 2.0):
        raise ValueError(f'prompt_strength must be 0 to 2 (got {v})')
    return x


def checked_shape_gain(v) -> float:
    """--shape-gain / a request's shape_gain: 0..2, finite (RUDRA 1.1 exposes it as Detail strength)."""
    x = float(v)
    if not (math.isfinite(x) and 0.0 <= x <= 2.0):
        raise ValueError(f'shape_gain must be 0 to 2 (got {v})')
    return x


def parse_region(text: str):
    """--region's JSON -> an enabled ReconstructionRegion, or None for ''."""
    from rudra.delivery.region import region_from_dict
    if not text or not text.strip():
        return None
    return region_from_dict(json.loads(text))


def request_region(r: dict, a):
    """A serve request's region: its own 'region' object (null = off) when it names one, else --region's."""
    from rudra.delivery.region import region_from_dict
    if 'region' in r:
        return region_from_dict(r['region']) if r['region'] else None
    return parse_region(a.region)


def pick_device(requested: str = 'auto'):
    """The device and the transformer's dtype (31 Oct 2026). CUDA in bfloat16 as before; on a Mac
    Apple's MPS in float32 (bfloat16 kernels are not complete there); else the CPU in float32.
    A device asked for that is not there falls back the same way."""
    import torch
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, 'mps', None)) and torch.backends.mps.is_available()
    if requested in ('auto', 'cuda') and cuda:
        return 'cuda', torch.bfloat16
    if requested in ('auto', 'cuda', 'mps') and mps:
        return 'mps', torch.float32
    return 'cpu', torch.float32


def brightness_settings(a, offset: float, shape_gain: float | None = None):
    """The v3 brightness split (brightness.json beside the LoRA, or --brightness-model), or None."""
    bm = a.brightness_model
    if bm == 'auto':
        cand = Path(a.lora).parent / 'brightness.json'
        bm = str(cand) if cand.exists() else 'none'
    if bm == 'none':
        return None
    from .brightness import BrightnessEstimator, BrightnessSettings
    from .model import sha256_file
    return BrightnessSettings(BrightnessEstimator.load(bm), offset, a.shape_gain if shape_gain is None else shape_gain,
                              identity={'estimator': bm, 'estimator_sha256': sha256_file(bm)})


def prepare_prompts(bank, texts, emit) -> None:
    """The prompts' embeddings, before the fill (6 Nov 2026). A prompt not in the cache loads umT5
    (about 11 GB, on the CPU) once to embed it: a minute or more with nothing on screen, so the app
    is told first ({"event": "embedding", "prompts": [...]}) and can say why it waits."""
    texts = [t for t in dict.fromkeys(texts) if t]
    new = [t for t in texts if not bank.has(t)]
    if new:
        emit(event='embedding', prompts=new, text=f'embedding {len(new)} new prompt(s): once, a minute or more')
    for t in texts:
        bank.get(t)
    bank.release()


def serve(model, view, a, emit, FillSettings) -> int:
    """The app's AI fill preview (4 Nov 2026): the model stays loaded, so a setting changed and
    previewed again costs the fill, not the 1-2 minutes of loading. One JSON object per stdin line:
        {"id": 3, "master": <EXR folder>, "out": <new folder>, "glob": "*.exr",
         "tau_high": 250, "tau_low": -1, "seed": 0, "steps": 30, "brightness_offset": 0.0,
         "prompt_highlight": "", "prompt_shadow": "", "window": 33, "overlap": 8,
         "prompt_strength": 1.0, "shape_gain": 1.0, "region": {...} or null, "fallback": "none",
         "focus": "frame" | "region" | "areas"}
    (any field left out takes the command line's value). Every event of a request carries its
    id; it ends with {"event": "done", "ok": true, "id": ..., ...} or an "error". 'cancel'
    stops the request running at its next step; 'quit' or the end of stdin ends the server."""
    import queue
    requests: queue.Queue = queue.Queue()
    stop = threading.Event()

    def reader():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == 'cancel':
                stop.set()
                continue
            if line == 'quit':
                break
            try:
                requests.put(json.loads(line))
            except ValueError as e:
                emit(event='error', ok=False, error='ValueError', detail=f'not a JSON request: {e}')
        requests.put(None)
    threading.Thread(target=reader, daemon=True).start()
    emit(event='serving')
    while True:
        r = requests.get()
        if r is None:
            return 0
        stop.clear()
        rid = r.get('id')
        tagged = lambda **o: emit(id=rid, **o)   # noqa: E731
        try:
            s = FillSettings(steps=int(r.get('steps', a.steps)), seed=int(r.get('seed', a.seed)),
                             guidance=float(r.get('guidance', a.guidance)),
                             prompt_highlight=str(r.get('prompt_highlight', a.prompt_highlight)),
                             prompt_shadow=str(r.get('prompt_shadow', a.prompt_shadow)),
                             prompt_strength=checked_prompt_strength(r.get('prompt_strength', a.prompt_strength)))
            prepare_prompts(model.prompts, [s.prompt, s.prompt_highlight, s.prompt_shadow, s.negative], tagged)
            ms = MaskSettings(tau_high=int(r.get('tau_high', a.tau_high)), tau_low=int(r.get('tau_low', a.tau_low)),
                              blend_high=a.blend_high, blend_low=a.blend_low, feather=a.feather,
                              max_gain_stops=a.max_gain_stops).check()
            out = Path(r['out'])
            if out.exists():
                raise FileExistsError(f'{out} exists: the fill writes a new folder')
            rep = fill_master(Path(r['master']), out, model, view, s, ms, int(r.get('window', a.window)),
                              int(r.get('overlap', a.overlap)), a.max_area, tagged, cancel=stop.is_set,
                              pattern=str(r.get('glob', '*.exr')),
                              brightness=brightness_settings(a, float(r.get('brightness_offset', a.brightness_offset)),
                                                             checked_shape_gain(r.get('shape_gain', a.shape_gain))),
                              region=request_region(r, a), fallback=str(r.get('fallback', a.fallback)),
                              focus=str(r.get('focus', a.focus)),
                              rerender=Rerender(str(r.get('source', a.source)), str(r.get('condition', a.condition)),
                                                str(r.get('rerender_hue', a.rerender_hue)), float(r.get('rerender_colour', a.rerender_colour)),
                                                float(r.get('rerender_level', a.rerender_level)),
                                                float(r.get('rerender_smooth', a.rerender_smooth)),
                                                int(r.get('source_start', a.source_start)),
                                                float(r.get('rerender_dip', a.rerender_dip)),
                                                float(r.get('rerender_still', a.rerender_still)))
                              if r.get('region_rerender', a.region_rerender) and r.get('source', a.source) else None)
            tagged(event='done', ok=True, out=str(out), **rep['stats'])
        except Exception as e:  # noqa: BLE001 - one failed request does not end the server
            tagged(event='error', ok=False, error=type(e).__name__, detail=str(e))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--master', default='', help='required unless --serve')
    ap.add_argument('--out', default='', help='required unless --serve')
    ap.add_argument('--serve', action='store_true',
                    help='load the model once, then fill one JSON request per stdin line (the app\'s preview); '
                         "'cancel' stops the request running, 'quit' or end of input ends")
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
    ap.add_argument('--device', default='auto', help='auto (CUDA, else Apple MPS, else CPU), cuda, mps or cpu')
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
    ap.add_argument('--prompt-strength', type=float, default=1.0,
                    help='RUDRA 1.1: how strongly a region prompt steers its region (0..2; 1 = as v3 was gated)')
    ap.add_argument('--fallback', default='none', choices=FALLBACKS,
                    help="RUDRA 1.1: 'spikes' writes the master's own frame where the fill flickers for one frame")
    ap.add_argument('--focus', default='frame', choices=FOCUS,
                    help="'region': the model draws only the box around what the shot fills, at the working size (more detail); "
                         "'areas': one box per group of areas, each at up to the working size (far-apart highlights in 4K)")
    ap.add_argument('--region', default='',
                    help='RUDRA 1.1: the reconstruction region as JSON (native reconstruction_region_json); '
                         'the fill acts only inside it')
    ap.add_argument('--glob', default='*.exr', help="the master folder's frames to fill (fnmatch, e.g. 'SH010.*.exr')")
    ap.add_argument('--region-rerender', action='store_true',
                    help='v4: re-render the whole mask from the source\'s codes; the SDR codes inside it are NOT kept (needs --source)')
    ap.add_argument('--source', default='', help='with --region-rerender: the movie or folder of stills the master was made from')
    ap.add_argument('--source-start', type=int, default=0,
                    help='with --region-rerender: the source frame the first master frame was made from (default 0: the whole shot)')
    ap.add_argument('--condition', default='bt1886', choices=Rerender.CONDITIONS,
                    help="with --region-rerender: what the model is conditioned on, the route's inverse (bt1886: code ^ 2.4; aces)")
    ap.add_argument('--rerender-hue', default='model', choices=RERENDER_HUES,
                    help="with --region-rerender: 'source' keeps the fill's luminance and takes the source's colour")
    ap.add_argument('--rerender-colour', type=float, default=0.0,
                    help="with --rerender-hue source: the share of the model's own colour kept, 0..1 (e.g. 0.3)")
    ap.add_argument('--rerender-level', type=float, default=0.0,
                    help="with --region-rerender: the level where the source is clipped, in SDR whites (e.g. 3): the model's "
                         "detail on it, crossfaded into the master by the source's code; 0 keeps the model's own level")
    ap.add_argument('--rerender-smooth', type=float, default=0.0,
                    help="with --region-rerender: how much of the previous frame's fill luminance is kept, 0..1 (e.g. 0.8); 0 is off")
    ap.add_argument('--rerender-dip', type=float, default=0.0,
                    help="with --rerender-level: the fill's fine detail at most this many stops under its surroundings "
                         "(e.g. 0.3: no dark smudges beside a silhouette); 0 is off, the fill as gated")
    ap.add_argument('--rerender-still', type=float, default=0.0,
                    help="with --region-rerender: the share of the previous frame's fill kept where the source holds still "
                         "(within 2 codes), in place of --rerender-smooth there (e.g. 0.9); 0 is off")
    a = ap.parse_args(argv)
    if not a.serve and (not a.master or not a.out):
        ap.error('--master and --out are required (or --serve)')
    if a.region_rerender and not a.source and not a.serve:
        ap.error('--region-rerender needs --source')
    emit = Emitter(a.json or a.serve)
    try:
        import torch
        from rudra.aces_view import ACES2View
        from .model import FillSettings, WanFill
        out = Path(a.out) if a.out else None
        if out is not None and not a.serve and out.exists():
            raise FileExistsError(f'{out} exists: the fill writes a new folder')
        device, dtype = pick_device(a.device)
        t0 = time.time()
        cache = Path(a.prompt_cache) if a.prompt_cache else Path(a.lora).with_suffix('').parent / 'prompt_cache'
        model = WanFill.load(a.base, a.lora, device=device, dtype=dtype, prompt_cache=cache, text_device='cpu')  # umT5 (11 GB) never on a 16 GB GPU beside the model
        checked_shape_gain(a.shape_gain)
        region = parse_region(a.region)                  # refused before the model loads
        s = FillSettings(steps=a.steps, seed=a.seed, guidance=a.guidance,
                         prompt_highlight=a.prompt_highlight, prompt_shadow=a.prompt_shadow,
                         prompt_strength=checked_prompt_strength(a.prompt_strength))
        prepare_prompts(model.prompts, [s.prompt, s.prompt_highlight, s.prompt_shadow, s.negative], emit)
        view = ACES2View(threads=max(1, min(16, a.ocio_threads)))
        emit(event='ready', device=device, load_seconds=round(time.time() - t0, 1), model=model.identity)
        if a.serve:
            return serve(model, view, a, emit, FillSettings)
        ms = MaskSettings(tau_high=a.tau_high, tau_low=a.tau_low, blend_high=a.blend_high,
                          blend_low=a.blend_low, feather=a.feather,
                          max_gain_stops=a.max_gain_stops).check()
        stop = threading.Event()
        if a.json:
            watch_stdin(stop)
        brightness = brightness_settings(a, a.brightness_offset)
        rep = fill_master(Path(a.master), out, model, view, s, ms, a.window, a.overlap, a.max_area, emit,
                          cancel=stop.is_set, pattern=a.glob, brightness=brightness, region=region,
                          fallback=a.fallback, focus=a.focus,
                          rerender=Rerender(a.source, a.condition, a.rerender_hue, a.rerender_colour, a.rerender_level, a.rerender_smooth,
                                            a.source_start, a.rerender_dip, a.rerender_still)
                          if a.region_rerender else None)
        emit(event='done', ok=True, out=str(out), **rep['stats'])
        return 0
    except Exception as e:  # noqa: BLE001 - every failure goes to the app as one line
        emit(event='error', ok=False, error=type(e).__name__, detail=str(e))
        if not a.json:
            raise
        return 2


if __name__ == '__main__':
    sys.exit(main())
