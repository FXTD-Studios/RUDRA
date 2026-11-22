"""Window joins (1.2.9): does an AI fill change where one window hands over to the next?

The fill is generated in windows of frames (rudra.genfill.run.plan_windows: 33 frames, 8 shared). Where two
windows share frames the later one is crossfaded in, and from the window's `new` frame on it is the later
window's alone. A window that drew its own sky, horizon or lamp shows as a step across that span
(09V, 1.2.8: the horizon and the sky's character change at frames 25, 50 and 75; the lamp pumps from 75).

join_report() takes any per-frame series (the region's level, its texture, a lamp's peak, a horizon row)
and, for each join, compares the change across the handover span (the frame before the crossfade to the
first frame of the new window alone) with the change over spans of the same length elsewhere in the shot:

    ratio = |v[new] - v[start - 1]| / median over the other spans |v[j + L] - v[j]|

About 1 means the join is as smooth as the shot's own motion; 3 and more is a step a viewer sees.
region_series() measures the usual series on a filled folder: the generated region's level and texture
(log2 luminance) and, with boxes, each box's median and p99.

    python -m tools.window_joins <fill folder> [--window 33 --overlap 8 --full-last] [--box lamp=x0,y0,x1,y1] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

AP1_Y = np.array([.2722287168, .6740817658, .0536895174], np.float32)


def plan_joins(n: int, window: int = 33, overlap: int = 8, full_last: bool = False) -> list[tuple[int, int]]:
    """(start, new) of every window after the first: frames start..new-1 are crossfaded, new is the later
    window's alone. full_last: as the fill planned it with --rerender-full-last."""
    from rudra.genfill.run import plan_windows
    return [(s, nw) for s, nw, _ in plan_windows(n, window, overlap, full_last)[1:]]


def join_steps(values, joins: list[tuple[int, int]]) -> list[dict]:
    """For each join, the step across its handover span against the shot's own change over spans of the
    same length that touch no handover. NaN values (no region in that frame) are left out."""
    v = np.asarray(values, np.float64)
    n = len(v)
    busy = np.zeros(n, bool)
    for s, nw in joins:
        busy[max(0, s - 1):min(n, nw + 1)] = True
    out = []
    for s, nw in joins:
        a, b = s - 1, nw
        if a < 0 or b >= n or not (np.isfinite(v[a]) and np.isfinite(v[b])):
            out.append({'start': s, 'new': nw, 'step': None, 'typical': None, 'ratio': None})
            continue
        span = b - a
        ref = [abs(v[j + span] - v[j]) for j in range(n - span)
               if not busy[j:j + span + 1].any() and np.isfinite(v[j]) and np.isfinite(v[j + span])]
        step = float(abs(v[b] - v[a]))
        typ = float(np.median(ref)) if ref else None
        ratio = None if typ is None else float(step / max(typ, 1e-6))
        out.append({'start': s, 'new': nw, 'step': round(step, 5), 'typical': None if typ is None else round(typ, 5),
                    'ratio': None if ratio is None else round(ratio, 2)})
    return out


def join_report(series: dict, n: int, window: int = 33, overlap: int = 8, full_last: bool = False) -> dict:
    """{name: per-frame values} -> {name: {'joins': [...], 'max_ratio': worst join}}."""
    joins = plan_joins(n, window, overlap, full_last)
    rep = {}
    for name, vals in series.items():
        steps = join_steps(vals, joins)
        rs = [x['ratio'] for x in steps if x['ratio'] is not None]
        rep[name] = {'joins': steps, 'max_ratio': max(rs) if rs else None}
    return rep


def _texture(logy: np.ndarray, inside: np.ndarray) -> float:
    import cv2
    hp = logy - cv2.GaussianBlur(logy, (0, 0), 3)
    return float(np.std(hp[inside])) if inside.any() else float('nan')


def frame_measures(rgb_ap1: np.ndarray, generated: np.ndarray | None, boxes: dict | None = None) -> dict:
    """One frame: the generated region's mean log2 Y ('level') and fine texture ('texture'), and for each
    box its median and p99 Y ('<box>50', '<box>99')."""
    y = np.maximum(rgb_ap1 @ AP1_Y, 1e-6).astype(np.float32)
    ly = np.log2(y)
    m = {}
    if generated is not None:
        inside = generated > 0.5
        m['level'] = float(ly[inside].mean()) if inside.any() else float('nan')
        m['texture'] = _texture(ly, inside)
    for name, (x0, y0, x1, y1) in (boxes or {}).items():
        part = y[y0:y1, x0:x1]
        m[name + '50'] = float(np.median(part))
        m[name + '99'] = float(np.percentile(part, 99))
    return m


def region_series(fill_dir: Path, boxes: dict | None = None, pattern: str = '*.exr') -> dict:
    from rudra.genfill.run import read_frame
    series: dict = {}
    for p in sorted(Path(fill_dir).glob(pattern)):
        fr = read_frame(p)
        gen = fr['channels'].get('rudra.generated')
        for k, v in frame_measures(fr['ap1'], gen, boxes).items():
            series.setdefault(k, []).append(v)
    return series


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('fill')
    ap.add_argument('--window', type=int, default=33)
    ap.add_argument('--overlap', type=int, default=8)
    ap.add_argument('--full-last', action='store_true', help='the fill ran with --rerender-full-last')
    ap.add_argument('--box', action='append', default=[], help='name=x0,y0,x1,y1 (pixels)')
    ap.add_argument('--json', default='')
    a = ap.parse_args(argv)
    boxes = {}
    for b in a.box:
        name, xs = b.split('=')
        boxes[name] = tuple(int(x) for x in xs.split(','))
    series = region_series(Path(a.fill), boxes)
    n = len(next(iter(series.values()))) if series else 0
    rep = {'fill': a.fill, 'frames': n, 'window': a.window, 'overlap': a.overlap,
           'joins': join_report(series, n, a.window, a.overlap, a.full_last), 'series': series}
    for k, r in rep['joins'].items():
        print(f"{k:10s} worst join x{r['max_ratio']}  " + '  '.join(f"{j['new']}: x{j['ratio']}" for j in r['joins']))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
