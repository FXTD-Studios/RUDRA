"""Restated gate for partial-highlight training (configs/experiments/claude_partial_tolerance_20261025.json).

A candidate that trains only highlight_context must leave fully clipped pixels exactly as its parent
produces them: raw candidate output == raw parent output inside fully clipped support, bit-exact,
on every listed frame. Also confirms the outside-highlight region is unchanged.

  python tools/check_full_support_unchanged.py --candidate <ft.pt> --parent <guarded.pt> --frames <png...> --out report.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.rudra_aces_export import load_model, read_still
from training.aces_model import reconstruction_weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', required=True); ap.add_argument('--parent', required=True)
    ap.add_argument('--frames', nargs='+', required=True); ap.add_argument('--out', default='')
    a = ap.parse_args()
    cand, _ = load_model(a.candidate, 'cpu'); par, _ = load_model(a.parent, 'cpu')
    cand.eval(); par.eval()
    rows = []
    for f in a.frames:
        src = read_still(Path(f))
        x = torch.from_numpy(np.ascontiguousarray(src)).permute(2, 0, 1)[None]
        with torch.inference_mode():
            c = cand(x).acescg[0].permute(1, 2, 0).numpy(); p = par(x).acescg[0].permute(1, 2, 0).numpy()
            w = reconstruction_weight(x)[0, 0].numpy()
        high = (w > 0) & (src.max(-1) > .5)
        full = (w > 0) & (src.min(-1) >= np.float32(254.5 / 255))
        partial = high & ~full
        rows.append({'frame': Path(f).name, 'full_pixels': int(full.sum()), 'partial_pixels': int(partial.sum()),
                     'full_support_bit_exact': bool(np.array_equal(c[full], p[full])),
                     'outside_highlight_bit_exact': bool(np.array_equal(c[~high], p[~high])),
                     'partial_changed_pixels': int((c[partial] != p[partial]).any(-1).sum())})
        print(rows[-1], flush=True)
    ok = all(r['full_support_bit_exact'] and r['outside_highlight_bit_exact'] for r in rows)
    rep = {'pass': ok, 'candidate': a.candidate, 'parent': a.parent, 'rows': rows}
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    print('full_support_unchanged pass', ok)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
