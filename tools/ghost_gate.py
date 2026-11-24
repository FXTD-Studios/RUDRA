#!/usr/bin/env python3
"""Ghost report (RUDRA 1.3, 23 Nov 2026): how a fill differs from a reference fill of the same LoRA beside things the
source moves, where v5b's finish drew its antenna ghosts.

The first try, a gate on thin dark structure inside the clipped, still area (closing over 7-11 px, 0.1-0.3 stops),
did not tell v5b from v5 on 09V (v4.2 532, v5 670, v5b 667 px a frame): the sky's own texture swamps the ghosts. So
this reports, per frame read, the pixels the source clips (>= 250) within BAND_PX of anything that moved (more than
MOVE_CODES over the last 8 frames), and how many of them the fill has more than DIFF_STOPS darker / lighter than the
reference. v5b against v5 on 09V: darker 15-319, lighter 6-380 a frame. Reported with the crops; not a gate.

    python tools/ghost_gate.py --frames <stills> --fill <EXR folder> --ref <EXR folder> [--every 8] [--json out.json]"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tmp' / '09V_test' / 'v4_gates'))
CLIP_CODE, MOVE_CODES, BAND_PX, DIFF_STOPS, LOOKBACK = 250, 8, 25, 0.2, 7
LUMA = np.array([.2722287168, .6740817658, .0536895174], np.float32)


def codes(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1]
    return im.astype(np.float32) * (255.0 / (65535.0 if im.dtype == np.uint16 else 255.0))


def measure(frames_dir, fill_dir, ref_dir, every=8):
    from shot_gates import read_exr
    pngs = sorted(Path(frames_dir).glob('*.png'))
    n = min(len(pngs), len(list(Path(fill_dir).glob('*.exr'))), len(list(Path(ref_dir).glob('*.exr'))))
    fills, refs = sorted(Path(fill_dir).glob('*.exr')), sorted(Path(ref_dir).glob('*.exr'))
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND_PX + 1,) * 2)
    y = lambda p: np.log2(np.maximum(read_exr(p)[0] @ LUMA, 1e-6))
    per = []
    for i in range(LOOKBACK + 1, n, every):
        cs = [codes(pngs[j]) for j in range(i - LOOKBACK, i + 1)]
        moved = np.zeros(cs[0].shape[:2], bool)
        for a, b in zip(cs, cs[1:]):
            moved |= np.abs(b - a).max(-1) > MOVE_CODES
        band = (cs[-1].max(-1) >= CLIP_CODE) & (cv2.dilate(moved.astype(np.uint8), el) > 0) & ~moved
        d = y(fills[i]) - y(refs[i])
        per.append({'frame': i, 'band_px': int(band.sum()), 'darker': int((band & (d < -DIFF_STOPS)).sum()),
                    'lighter': int((band & (d > DIFF_STOPS)).sum())})
    return {'fill': str(fill_dir), 'ref': str(ref_dir), 'frames_read': len(per),
            'darker_mean': round(float(np.mean([p['darker'] for p in per])), 1),
            'darker_max': max(p['darker'] for p in per), 'lighter_mean': round(float(np.mean([p['lighter'] for p in per])), 1),
            'per_frame': per}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--fill', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--every', type=int, default=8)
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    r = measure(a.frames, a.fill, a.ref, a.every)
    print(json.dumps({k: v for k, v in r.items() if k != 'per_frame'}))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
