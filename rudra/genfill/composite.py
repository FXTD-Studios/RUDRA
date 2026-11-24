"""The round-trip-safe composite.

out = master + alpha * (fill - master), per pixel, with alpha <= the feathered mask weight and
alpha chosen so that the master's SDR view keeps every 8-bit code:

    rint(255 * clip(view(out))) == rint(255 * clip(view(master)))   on every pixel

The R contract (1 code to the source on 99.9 % of mask-0 pixels) therefore holds for the filled
master exactly as it held for the master: in a clipped highlight any brighter value still views
as 255, so the fill is free there; in a crushed shadow any value below the code's upper edge
still views as 0; a fill that would change a visible code is pulled back towards the master by
bisection (the largest tested alpha that keeps the codes; alpha = 0 always does).

`view` is any object with forward(acescg (N, 3) float32) -> display RGB, e.g. rudra.aces_view.ACES2View.
"""
from __future__ import annotations

import numpy as np


def _codes(view, rgb: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(view.forward(np.ascontiguousarray(rgb, np.float32)), 0.0, 1.0) * 255.0).astype(np.int16)


def plain_blend(master: np.ndarray, fill: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """out = master + weight * (fill - master): the region re-render (v4, 9 Nov 2026). Inside the
    mask the SDR codes are NOT kept; outside it (weight 0) every pixel is the master's, bit for bit."""
    if master.shape != fill.shape or master.shape[:2] != weight.shape:
        raise ValueError('master, fill and weight must agree in size')
    alpha = np.clip(weight.astype(np.float32), 0.0, 1.0)
    out = np.array(master, np.float32, copy=True)
    idx = np.nonzero(alpha > 0)
    if len(idx[0]):
        out[idx] = master[idx] + alpha[idx][:, None] * (fill[idx].astype(np.float32) - master[idx])
    return out, alpha


LUMA_AP1 = np.array([.2722287168, .6740817658, .0536895174], np.float32)
RERENDER_HUES = ('model', 'source')
RERENDER_FULL_CODE = 253        # the source's max code at which the fill sits at the whole level
RERENDER_HUE_SIGMA = 8.0        # pixels: the source's colour is taken from its neighbourhood, not per pixel
RERENDER_DETAIL_SIGMA = 64.0    # pixels at 1080 lines: above this scale the level is the source's (or the model's, within the ceiling)
# How far over `level` the model's own level may sit. 0 since 11 Nov 2026: the model's level is not steady
# (Porch: the region's median fell from 5.5 to 3.3 x white over the first 8 frames with 2 stops allowed, and
# held 2.8-3.2 with none), and with the crossfade by code the allowance bought nothing on the held-out clips
# (PU21 at -3 EV -28.1 with it, -27.6 without) or on 09V (no pixel moved).
RERENDER_CEILING_STOPS = 0.0
RERENDER_FRINGE_PX = 12.0       # pixels at 1080 lines: this close to a darker neighbour the fill is the interior's
RERENDER_REACH_PX = 6.0         # pixels at 1080 lines: beside something dark a code takes its brightest neighbour's within this
RERENDER_DARK_CODES = 40        # "something dark": a neighbour this many codes under tau_high (fully so at 80)


RERENDER_RISE_PX = 150.0        # pixels at 1080 lines: with `rise`, the depth into the full-code core over which the lift approaches it


RERENDER_CALM_STOPS = (0.1, 0.3) # stops: with `calm`, a still pixel's fill that would move this much starts to be held; fully from the second
RERENDER_STEADY_MAX = 15       # 1.3: the longest median `steady` takes
RERENDER_RELEASE_PX = 7        # 1.3 research: hold 'release' lets go of a carried dark line narrower than this
RERENDER_RELEASE_STOPS = 0.15  # ... that its closing lifts by more than this
RERENDER_THIN_PX = 20.0        # 1.3: hold 'thin' holds a clipped shape at most this far (px at 1080) from its edge at its widest
RERENDER_THIN_REACH_PX = 40    # ... its widest point looked for this far around (px at 1080)
RERENDER_HOLD_CODE = 250       # 1.3: with `calm`, a still pixel the source clips at this code is held whole (the T gate's clipped)
RERENDER_STILL_TOP = 3.0       # 1.3: with `blend_still`, the crossfade's code (blurred) within this many codes of where its hold began
RERENDER_NEUTRAL = (0.78, 0.88) # the source's min over max code: below the first the region is left to the master, whole from the second


def neutral_weight(codes: np.ndarray) -> np.ndarray:
    """(H, W) in [0, 1]: how near white the source is, from its codes (H, W, 3). The region re-render
    is for an area the source parks near white in every channel: nothing is known there. A saturated
    highlight (a sunset: red at 255, blue at 120) still carries its colour and its gradient, and a fill
    at a level of its own draws a contour across it (11 Nov 2026, the 07V still: the mode whitened the
    sky above a hard edge). There the weight goes to 0 and the master stays."""
    c = codes.astype(np.float32)
    lo, hi = RERENDER_NEUTRAL
    x = np.clip((c.min(-1) / np.maximum(c.max(-1), 1.0) - lo) / (hi - lo), 0.0, 1.0)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


RERENDER_DETAIL_WINDOW = 25        # px at 1080 lines: the box over which a region pixel's neighbours are counted
RERENDER_DETAIL_COVER = (0.45, 0.75)   # the share of that box in the region: below the first the master stays, whole from the second
RERENDER_DETAIL_MIN_AREA = 5e-4    # a region area under this share of the frame keeps the master (a lamp, a glint)


def detail_keep_weight(codes: np.ndarray, tau_high: int) -> np.ndarray:
    """(H, W) in [0, 1]: 1.2.7 (multi-role review of 1.2.6), --rerender-keep-detail. The region re-render
    redraws every pixel at code tau_high or above. Real objects that sit there keep their picture with this
    on: a thin or broken-up part of the region (a floor grating's bright bars, a railing) by how little of
    the box around it is region, and an area under RERENDER_DETAIL_MIN_AREA of the frame (a practical lamp)
    entirely. core/src/fill_region.cpp ai_fill_v4_weight draws the same rule over the viewer."""
    import cv2
    h, w = codes.shape[:2]
    region = (codes.max(-1) >= tau_high).astype(np.uint8)
    k = max(3, int(round(RERENDER_DETAIL_WINDOW * h / 1080.0)) | 1)
    cover = cv2.blur(region.astype(np.float32), (k, k), borderType=cv2.BORDER_REPLICATE)
    lo, hi = RERENDER_DETAIL_COVER
    x = np.clip((cover - lo) / (hi - lo), 0.0, 1.0)
    weight = x * x * (3.0 - 2.0 * x) * region
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < RERENDER_DETAIL_MIN_AREA * h * w
    small[0] = True                                   # the background label
    weight[small[labels]] = 0.0
    return weight.astype(np.float32)


def white_scale(master: np.ndarray, codes: np.ndarray) -> float | None:
    """The master's SDR white on one frame: the median, over mid-tone pixels (every channel between
    codes 50 and 180), of the master's luminance over the source's display light (code ^ 2.4), as
    tools/compare_hdr_renders.py reads its 'scale'. None when the frame has too few mid-tones."""
    mid = np.all((codes > 50) & (codes < 180), -1)
    if int(mid.sum()) < 500:
        return None
    light = np.power(codes[mid].astype(np.float32) / 255.0, 2.4) @ LUMA_AP1
    return float(np.median((master[mid] @ LUMA_AP1) / np.maximum(light, 1e-6)))


def finish_rerender(fill: np.ndarray, master: np.ndarray, source_lin: np.ndarray, codes: np.ndarray, weight: np.ndarray,
                    hue: str = 'model', colour: float = 0.0, level: float = 0.0, smooth: float = 0.0,
                    white: float = 1.0, tau_high: int = 240, state: dict | None = None, dip: float = 0.0,
                    still: float = 0.0, rise: float = 0.0, calm: float = 0.0, hold: bool | str = False,
                    steady: int = 0, cap: bool = False, blend_still: float = 0.0, hold_calm: float = 0.0) -> np.ndarray:
    """The region re-render's fill, finished before the blend (10 Nov 2026, after the v4 gates on 09V:
    the v4 LoRA drew the structure, but too blue, at 1.2 x white where 2.5-4 was asked, its level
    drifting 0.85-1.72 x white along the shot with jumps at the window seams, and brighter where the
    source is darker).

    fill, master, source_lin (the source's display light, code ^ 2.4) (H, W, 3) ACEScg; codes (H, W, 3)
    8-bit; weight (H, W) the blend weight. Every step is off at its default, and then the fill is
    returned as is:
      hue 'source'  the fill keeps its luminance and takes the source's colour (the source's display
                    light blurred RERENDER_HUE_SIGMA px, so near-clip noise does not tint it), mixed
                    with `colour` (0..1) of the model's own
      level > 0     the detail from the model, the level `level` x `white` (the master's SDR white,
                    white_scale): the fill keeps what is finer than RERENDER_DETAIL_SIGMA; above that
                    scale its own level is kept only where brighter and only RERENDER_CEILING_STOPS
                    over (0: not at all; see the constant).
                    That levelled fill is then crossfaded into the MASTER by the source's code, log
                    luminance and colour alike: the master itself at tau_high, the whole fill at
                    RERENDER_FULL_CODE. So the lift starts at nothing where the region starts and a
                    brighter source stays brighter (11 Nov, the Porch shot: a softbox's hot spots,
                    a smooth gradient through tau_high, came out as ovals with an edge when the
                    model's level applied whole from the region's rim).
                    Two edge rules go with it: within RERENDER_FRINGE_PX of a darker neighbour the
                    fill is taken from the interior (the model's fill is soft and dark beside a
                    silhouette), and beside something dark a code takes its brightest neighbour's
                    within RERENDER_REACH_PX (an anti-aliased edge is a mix of figure and sky, not a
                    darker sky; in a gradient there is nothing dark and the codes stand)
      dip > 0       with a level: the model's fine detail goes at most `dip` stops under its surroundings
                    (a soft floor from half of it). 12 Nov 2026, 09V in the app: dark smudges 0.6 to 2.4
                    stops deep beside the figures, a ghost of the silhouette; clouds are 0.05 to 0.12 stops,
                    and nothing in a region parked at white is that much darker than what is round it.
                    Brighter detail is not touched. Off (0) is the fill as gated on 11 Nov
      smooth > 0    the fill's log luminance follows the previous frame's (an EMA, this share kept)
                    wherever both frames are filled; `state` (a dict the caller keeps for the shot)
                    carries it. A newly filled pixel starts from its own value.
      still > 0     where the source holds still (every channel within 2 codes of the previous frame)
                    this share is kept instead of `smooth`: nothing in the source says the light there
                    changed, so the fill should not either. 12 Nov 2026, water.mp4: the static clipped
                    pixels moved 0.19 EV (p95) with smooth 0.8 alone, 0.06 with still 0.9; the
                    analytic master's own were 0.5 to 2.3 EV. Where the source moves, `smooth` stands.
                    It also averages the model's detail away (09V's structure halved): off in the app.
      calm > 0      where the source holds still (as `still`) AND the fill would move by more than
                    RERENDER_CALM_STOPS[0] from the previous frame's, this share is kept, fully so from
                    RERENDER_CALM_STOPS[1]; under it `smooth` stands. 15 Nov 2026: `still` holds every
                    still pixel and so averages a steady picture away over a shot (09V); the flicker
                    it was made for is the model's detail jumping on still pixels inside moving
                    highlights (water: the raw fill moves 1.5 EV p95 there, the door's 0.08). Only a
                    jump is held. Measured on the v4 raw fills with this finish: water frames 30-35,
                    static clipped p95 0.115 -> 0.063 EV, structure 0.112 -> 0.131; 09V frames 40-47,
                    structure 0.0957 both ways, 98.4 % of the region within 0.01 stops of the fill
                    without it. Off (0) is the fill as gated.
      rise > 0      with a level: where the source sits at RERENDER_FULL_CODE the level keeps climbing,
                    up to `rise` stops over `level`, with the depth into that full-code core
                    (rise * (1 - exp(-depth / RERENDER_RISE_PX)), the depth blurred so the core's
                    medial axis leaves no crease). 13 Nov 2026, testwindow.mp4 (sky glare): with the
                    model's level held at the floor (RERENDER_CEILING_STOPS 0) the core was a flat
                    plate at exactly 3.30 x white from 250 px in, a mask-shaped disc at -3 EV where
                    glare keeps rising towards its source. In a clipped core the only evidence left
                    is how far the clip reaches; a wider clip is taken as a brighter one. Fine detail
                    is untouched (it adds no structure, so it does not move the S gate). Off (0) is
                    the fill as gated on 11 Nov.
      steady > 1    1.3 (23 Nov 2026, AI fill v5): where the source clips (RERENDER_HOLD_CODE) and holds
                    still (2 codes), the fill's own log luminance is the median of its last `steady`
                    frames (as many as the pixel has held still), before `smooth` / `calm`. v5 redraws a
                    thin LED strip a little differently every frame (Porch: static clipped p95 0.229 EV,
                    limit 0.10) beside the dark, where the 1.2.8 trust lets `calm` hold nothing; holding
                    regardless (v5b) carried 09V's drifting antenna's ghosts for frames. A median keeps
                    nothing of its own: it only ever chooses among the model's own recent values (where the
                    pixel has held still for all of them), so a copy the model drew in fewer than half of
                    them is dropped and none can outlast the
                    model drawing it by more than steady // 2 frames. Kept in state: the last frames'
                    values (the ring), never an output.
                    0 or 1: off.
      cap           with a level (1.3, 24 Nov 2026, the learned route): in the crossfade the master's share is
                    never brighter than the fill's own value there. The learned ACES 2.0 master rebuilds a clipped
                    area itself, brighter than the fill's level and unsteady near the clip (Porch: 7.8 x white, its
                    still clipped pixels moving 0.75 EV p95; water 24 x white, 0.67 EV), so its share of a region
                    pixel came through as a rim brighter than the core and as flicker (the T gate on 01V, 02V,
                    water; L 4.07 on Porch). The analytic master sits near white there, under the fill's level,
                    where the cap changes nothing; the app sets it on the learned route only. Off: as gated.
      blend_still   with a level and `state` (1.3, 24 Nov 2026): while the code the crossfade follows stays within
                    RERENDER_STILL_TOP of where it was when the hold began, the crossfade's share of the fill (s)
                    keeps this share of the previous frame's. s follows the source's code, and on a still shot a
                    code near the clip moves 1-2 codes from frame to frame (the codec's noise): at code 251 that
                    moves s by up to 0.12, and with the analytic master about 1.7 stops under the fill's level the
                    pixel moved up to 0.2 stops (01V on the analytic route: T 0.164, 98 % of the worst changes
                    within 4 px of a clipped edge; with 0.95, 0.079). 0: as gated.
      hold_calm     with `hold` (1.3, 24 Nov 2026): the share a held pixel keeps for a jump, instead of `calm`.
                    v5 redraws the inside of a thin clipped shape a little differently every frame (03V: T 0.17
                    with the mix settled, 0.094 with 0.95 here); `calm` itself stays as gated, so a wide clipped
                    sky's clouds are finished as before. 0: `calm`."""
    if hue not in RERENDER_HUES:
        raise ValueError(f'hue must be one of {", ".join(RERENDER_HUES)}')
    if (not 0.0 <= smooth < 1.0 or not 0.0 <= colour <= 1.0 or level < 0 or dip < 0 or rise < 0 or not 0.0 <= still < 1.0
            or not 0.0 <= calm < 1.0 or not 0 <= int(steady) <= RERENDER_STEADY_MAX or not 0.0 <= blend_still < 1.0
            or not 0.0 <= hold_calm < 1.0):
        raise ValueError('smooth, still and calm must be in [0, 1), colour in [0, 1], level, dip and rise >= 0, '
                         f'steady 0..{RERENDER_STEADY_MAX}')
    steady = int(steady)
    if hue == 'model' and level <= 0 and smooth <= 0 and still <= 0 and calm <= 0 and steady <= 1:
        return fill
    import cv2
    on = weight > 0
    if not on.any():
        return fill
    y = np.maximum(fill @ LUMA_AP1, 1e-6).astype(np.float32)
    chroma = fill / y[..., None]
    log_y = np.log2(y)
    px = fill.shape[0] / 1080.0
    top_code = codes.max(-1).astype(np.float32)
    if level > 0:                                       # the interior's fill beside a darker neighbour
        r = max(1.0, RERENDER_FRINGE_PX * px)
        d = np.clip(cv2.distanceTransform((top_code >= tau_high - 8).astype(np.uint8), cv2.DIST_L2, 3) / r, 0.0, 1.0)
        conf = (d * d * (3.0 - 2.0 * d) * on).astype(np.float32)
        near_conf = cv2.GaussianBlur(conf, (0, 0), r)
        known = near_conf > 1e-3
        inner_y = cv2.GaussianBlur(log_y * conf, (0, 0), r) / np.maximum(near_conf, 1e-3)
        inner_c = cv2.GaussianBlur(chroma * conf[..., None], (0, 0), r) / np.maximum(near_conf, 1e-3)[..., None]
        log_y = np.where(known, conf * log_y + (1.0 - conf) * inner_y, log_y).astype(np.float32)
        chroma = np.where(known[..., None], conf[..., None] * chroma + (1.0 - conf[..., None]) * inner_c, chroma).astype(np.float32)
    if hue == 'source':
        near = cv2.GaussianBlur(np.ascontiguousarray(source_lin, np.float32), (0, 0), RERENDER_HUE_SIGMA)
        chroma = (1.0 - colour) * near / np.maximum(near @ LUMA_AP1, 1e-6)[..., None] + colour * chroma
    if level > 0:
        sigma = RERENDER_DETAIL_SIGMA * px
        inside = on.astype(np.float32)                  # the model's own neighbourhood: filled pixels only
        low = cv2.GaussianBlur(log_y * inside, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(inside, (0, 0), sigma), 1e-3)
        k = int(round(RERENDER_REACH_PX * px))
        if k > 0:
            el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            e = np.clip((tau_high - RERENDER_DARK_CODES - cv2.erode(top_code, el)) / float(RERENDER_DARK_CODES), 0.0, 1.0)
            top_code = top_code + e * e * (3.0 - 2.0 * e) * (cv2.dilate(top_code, el) - top_code)
        top = cv2.GaussianBlur(top_code, (0, 0), 2.0)
        s = np.clip((top - tau_high) / max(1.0, float(RERENDER_FULL_CODE - tau_high)), 0.0, 1.0)
        s = s * s * (3.0 - 2.0 * s)
        if blend_still > 0 and state is not None:       # 1.3: the mix settles while its code holds still
            # Held while `top`, the code s is made from (blurred, and lifted beside something dark), stays within
            # RERENDER_STILL_TOP of where the hold began: the codec's noise moves it about there and back. A
            # thin dark object drifting under a pixel a frame (09V's antenna) moves it as little from one frame
            # to the next, but on and on, so it soon leaves the band and the mix follows again. Holding on the
            # frame-to-frame change instead (the pixel's own codes, or `top`) left faint copies of the antenna
            # beside it (24 Nov 2026 replays, tmp/v5_review/09V_*_antenna3x.jpg).
            ps, ref, pon = state.get('s'), state.get('top_ref'), state.get('on')
            if ps is not None and ps.shape == s.shape and ref is not None and ref.shape == top.shape and pon is not None:
                held = (np.abs(top - ref) <= np.float32(RERENDER_STILL_TOP)) & on & pon
                s = np.where(held, np.float32(blend_still) * ps + np.float32(1.0 - blend_still) * s, s)
                ref = np.where(held, ref, top)
            else:
                ref = top
            s = s.astype(np.float32)
            state['s'], state['top_ref'] = s, ref.astype(np.float32)
        floor = np.float32(np.log2(level * white))
        detail = log_y - low
        if dip > 0:
            half = np.float32(0.5 * dip)
            detail = np.where(detail < -half, -half - half * np.tanh((-detail - half) / half), detail).astype(np.float32)
        whole = np.clip(low, floor, floor + RERENDER_CEILING_STOPS) + detail
        if rise > 0:                                    # the full-code core keeps climbing with its depth
            L = RERENDER_RISE_PX * px
            core = ((s >= 0.999) & on).astype(np.uint8)
            if core.any():
                depth = cv2.GaussianBlur(cv2.distanceTransform(core, cv2.DIST_L2, 5).astype(np.float32), (0, 0), 0.5 * L)
                whole = whole + np.float32(rise) * (1.0 - np.exp(-depth / np.float32(L))) * s
        y_m = np.maximum(master @ LUMA_AP1, 1e-6).astype(np.float32)
        log_m = np.log2(y_m).astype(np.float32)
        whole = whole.astype(np.float32)
        log_y = ((1.0 - s) * (np.minimum(log_m, whole) if cap else log_m) + s * whole).astype(np.float32)
        chroma = ((1.0 - s[..., None]) * (master / y_m[..., None]) + s[..., None] * chroma).astype(np.float32)
    if (smooth > 0 or still > 0 or calm > 0 or steady > 1) and state is not None:
        # 1.2.8 (QC of v4.3 on 09V, 21 Nov 2026): what is carried from frame to frame is the fill's own
        # log luminance (`whole` with a level), never the master's share of a region-edge pixel. The
        # carried value had (1 - s) of the master in it, so a dark thin object moving along the edge (09V's
        # antenna, 2-3 px against the sky) left its old positions behind at 0.8, 0.64, ... of its depth: a
        # fan of faint antennas. The master's share is now this frame's, mixed in after the smoothing.
        own = whole if level > 0 else log_y
        # ... and only as much as the previous frame's pixel was the fill's own sky: where the region's edge
        # mixed in the master (s < 1) or the pixel sat within RERENDER_FRINGE_PX of something dark (conf < 1),
        # the model's value there also drew the antenna, so it is carried that much less (the GPU check of
        # 21 Nov 2026 left a faint fan with the first fix alone).
        trust = (s * conf).astype(np.float32) if level > 0 else on.astype(np.float32)
        if steady > 1:                                  # 1.3: the median of the model's own recent values
            own = steady_median(own, codes, on, state, steady)
        prev = state.get('log_y')
        if prev is not None and prev.shape == own.shape:
            both = on & state['on']
            keep = np.full(log_y.shape, np.float32(smooth))
            pc = state.get('codes')
            if (still > 0 or calm > 0) and pc is not None and pc.shape == codes.shape:
                held = np.abs(codes.astype(np.int16) - pc.astype(np.int16)).max(-1) <= 2
                if still > 0:
                    keep[held] = np.float32(still)
                if calm > 0:                            # only a jump is held; a steady picture keeps `smooth`
                    lo, hi = RERENDER_CALM_STOPS
                    t = np.clip((np.abs(own - prev) - lo) / (hi - lo), 0.0, 1.0)
                    t = t * t * (3.0 - 2.0 * t)
                    jump = np.float32(smooth) + (np.float32(calm) - np.float32(smooth)) * t
                    keep = np.where(held, np.maximum(keep, jump), keep).astype(np.float32)
            pt = state.get('trust')
            if pt is not None and pt.shape == keep.shape:
                scaled = (keep * pt).astype(np.float32)
                if hold and calm > 0 and pc is not None and pc.shape == codes.shape:
                    # 1.3 research (23 Nov 2026, v5b; `hold`, off by default and in the app): a pixel the source
                    # clips and holds still is held whole, trust or not. v5 redraws Porch's thin LED strip a little
                    # differently every frame (static clipped p95 0.229 EV, v4.3 0.058) and the strip lies within
                    # RERENDER_FRINGE_PX of the dark ceiling, where the trust above holds nothing; with the hold
                    # 0.080. But the sky beside 09V's drifting antenna is clipped and still too: there the hold
                    # keeps the model's own antennas for frames (v5b: 150-300 px a frame up to 0.87 stops darker
                    # than v5, 4-23 px from it). Not usable as it is.
                    clipped = held & (codes.max(-1) >= RERENDER_HOLD_CODE) & (pc.max(-1) >= RERENDER_HOLD_CODE)
                    mode = hold if isinstance(hold, str) else 'all'
                    if mode == 'dips':
                        # only a dip is held: a pixel getting brighter is never held back, so a dark copy the
                        # model drew cannot stay once it moves on (the v5b ghosts were dark)
                        clipped &= own < prev
                    elif mode == 'release':
                        # held, except where what would be carried is a thin dark line in the last frame (a copy
                        # of something thin): its grey closing over RERENDER_RELEASE_PX lifts it more than
                        # RERENDER_RELEASE_STOPS
                        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RERENDER_RELEASE_PX, RERENDER_RELEASE_PX))
                        clipped &= (cv2.morphologyEx(prev, cv2.MORPH_CLOSE, k) - prev) <= RERENDER_RELEASE_STOPS
                    elif mode == 'thin':
                        # held only inside a thin clipped shape (a strip light, a tube, a small lamp): where the
                        # widest point of the clipped area within RERENDER_THIN_REACH_PX is at most RERENDER_THIN_PX
                        # from its edge. A wide clipped sky (09V's door, beside the drifting antenna) keeps v5's own
                        # finish, whose trust already holds its interior.
                        top_c = (codes.max(-1) >= RERENDER_HOLD_CODE).astype(np.uint8)
                        dt = cv2.distanceTransform(top_c, cv2.DIST_L2, 3)
                        reach = max(3, int(round(RERENDER_THIN_REACH_PX * px)) | 1)
                        widest = cv2.dilate(dt, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * reach + 1, 2 * reach + 1)))
                        clipped &= widest <= np.float32(RERENDER_THIN_PX * px)
                    elif mode != 'all':
                        raise ValueError(f'hold must be off, all, dips, release or thin, not {hold!r}')
                    if hold_calm > 0:                   # 1.3: the held pixels' own share for a jump
                        jump = np.float32(smooth) + (np.float32(hold_calm) - np.float32(smooth)) * t
                    scaled = np.where(clipped, np.maximum(scaled, jump), scaled).astype(np.float32)
                keep = scaled
            own = own.copy()
            own[both] = keep[both] * prev[both] + (1.0 - keep[both]) * own[both]
            log_y = ((1.0 - s) * (np.minimum(log_m, own) if cap else log_m) + s * own).astype(np.float32) if level > 0 else own
        state['log_y'], state['on'], state['codes'], state['trust'] = own, on, codes.copy(), trust
    out = np.array(fill, np.float32, copy=True)
    out[on] = (np.exp2(log_y)[..., None] * chroma)[on]
    return out


def steady_median(own: np.ndarray, codes: np.ndarray, on: np.ndarray, state: dict, n: int) -> np.ndarray:
    """finish_rerender's `steady` (1.3): own (H, W) the fill's own log luminance this frame -> the same, where the
    source clips (RERENDER_HOLD_CODE) and has held still (2 codes) for all of the last n frames, the median of the
    n values the fill had there. state['steady_ring'] keeps the last n - 1 frames' own values as they came in
    (never an output), state['steady_run'] how many frames each pixel has held, state['steady_codes'] last codes."""
    top = codes.max(-1)
    pc = state.get('steady_codes')
    run = state.get('steady_run')
    if pc is not None and pc.shape == codes.shape:
        held = ((np.abs(codes.astype(np.int16) - pc.astype(np.int16)).max(-1) <= 2) & (top >= RERENDER_HOLD_CODE)
                & (pc.max(-1) >= RERENDER_HOLD_CODE))
    else:
        held = np.zeros(top.shape, bool)
    run = np.where(held, (run if run is not None and run.shape == top.shape else np.zeros(top.shape, np.int16)) + 1,
                   0).astype(np.int16)
    ring = [r for r in state.get('steady_ring', []) if r.shape == own.shape]
    state['steady_ring'] = (ring + [own.astype(np.float32, copy=True)])[-(n - 1):] if n > 1 else []
    state['steady_run'], state['steady_codes'] = run, codes.copy()
    if not ring:
        return own
    if len(ring) < n - 1:
        return own
    # Only where the source has held still for the whole window (a partial 2- or 3-frame median keeps a copy
    # drawn just after something passed). A value is in the output only while the model drew it in most of
    # the last n frames: a copy drawn in fewer than n // 2 + 1 of them never shows, and every copy is gone
    # n // 2 frames after the model stops drawing it.
    sel = (run >= n - 1) & on
    out = own.copy()
    if sel.any():
        out[sel] = np.median(np.stack([f[sel] for f in ring + [own]]), axis=0)
    return out.astype(own.dtype)


def r_safe_blend(master: np.ndarray, fill: np.ndarray, weight: np.ndarray, view,
                 iterations: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """master, fill (H, W, 3) ACEScg; weight (H, W) in [0, 1] -> (out (H, W, 3), alpha (H, W))."""
    if master.shape != fill.shape or master.shape[:2] != weight.shape:
        raise ValueError('master, fill and weight must agree in size')
    out = np.array(master, np.float32, copy=True)
    alpha = np.zeros(weight.shape, np.float32)
    idx = np.nonzero(weight > 0)
    if not len(idx[0]):
        return out, alpha
    m = master[idx].astype(np.float32)
    d = fill[idx].astype(np.float32) - m
    w = np.clip(weight[idx].astype(np.float32), 0.0, 1.0)
    ref = _codes(view, m)

    def ok(a):
        return (_codes(view, m + a[:, None] * d) == ref).all(-1)

    good = ok(w)
    best = np.where(good, w, 0.0).astype(np.float32)
    bad = np.nonzero(~good)[0]
    if len(bad):
        lo = np.zeros(len(bad), np.float32)
        hi = w[bad].copy()
        mb, db, rb = m[bad], d[bad], ref[bad]
        for _ in range(iterations):
            mid = (lo + hi) * 0.5
            g = (_codes(view, mb + mid[:, None] * db) == rb).all(-1)
            lo = np.where(g, mid, lo)
            hi = np.where(g, hi, mid)
        best[bad] = lo
    alpha[idx] = best
    out[idx] = m + best[:, None] * d
    return out, alpha
