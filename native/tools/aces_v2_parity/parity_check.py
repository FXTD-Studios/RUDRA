"""C++ ACES v2 stages vs the Python reference, stage by stage, on real frames.

  python native/tools/aces_v2_parity/parity_check.py --exe <aces_v2_parity(.exe)> \
      --checkpoint <v9 .pt> --g1 <package>/rudra_g1.ts --out report.json <frames...>
Pass: every stage bit-exact, or within --tol (relative) with 0 display-code difference.
"""
import argparse, sys, json, subprocess, tempfile, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.export_native_v2 import G1, staged_reference
from tools.rudra_aces_export import load_model, read_still, CLIPPED_SMOOTHING
from training.aces_clipped_smoothing import smooth_clipped
from training.aces_colour_backtrack import constrain_colour
from training.aces_reference import inverse_aces2_reference, reference_view
from training.aces_model import reconstruction_weight

ap = argparse.ArgumentParser()
ap.add_argument('--exe', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--g1', required=True)
ap.add_argument('--out', default=''); ap.add_argument('--forbid-new-negative-y', action='store_true'); ap.add_argument('--tol', type=float, default=1e-5); ap.add_argument('frames', nargs='+')
a = ap.parse_args()
ck, ts, frames = a.checkpoint, a.g1, a.frames
model, _ = load_model(ck, 'cpu'); model.eval(); g1 = G1(model).eval()
def rd(p):
    with open(p, 'rb') as f:
        h, w, c = np.frombuffer(f.read(12), np.int32); return np.frombuffer(f.read(), np.float32).reshape(h, w, c)
rows = []
for fr in frames:
    src = read_still(Path(fr)) if not fr.endswith('.npy') else np.load(fr).astype(np.float32)
    d = Path(tempfile.mkdtemp())
    with open(d / 'in.f32', 'wb') as f:
        f.write(np.array(src.shape[:2], np.int32).tobytes()); f.write(np.ascontiguousarray(src, np.float32).tobytes())
    t = subprocess.run([a.exe, str(d / 'in.f32'), ts, str(d)] + (['--forbid-new-negative-y'] if a.forbid_new_negative_y else []), capture_output=True, text=True)
    if t.returncode: print(t.stderr); sys.exit(1)
    cpp = json.loads(t.stdout.strip().splitlines()[-1])
    x = torch.from_numpy(np.ascontiguousarray(src)).permute(2, 0, 1)[None]
    with torch.inference_mode():
        pre = model(x).acescg[0].permute(1, 2, 0).numpy()
        base = inverse_aces2_reference(x)[0].permute(1, 2, 0).numpy()
        w = reconstruction_weight(x)[0, 0].numpy()
    sm, _ = smooth_clipped(pre, src, **CLIPPED_SMOOTHING['hybrid64'])
    fin, diag = constrain_colour(src, sm, base, w, reference_view(), include_fully_clipped=True, forbid_new_negative_y=a.forbid_new_negative_y)
    def cmp(a, b):
        d = np.abs(a.astype(np.float64) - b); rel = d / (np.abs(b) + 1e-3)
        return {'max_abs': float(d.max()), 'max_rel': float(rel.max()), 'bit_exact_fraction': float((a == b).all(-1).mean() if a.ndim == 3 else (a == b).mean())}
    disp = lambda a: reference_view().forward(a)
    r = {'frame': Path(fr).name, 'shape': list(src.shape), 'cpp': cpp,
         'composed': cmp(rd(d / 'composed.f32'), pre), 'smoothed': cmp(rd(d / 'smoothed.f32'), sm),
         'final': cmp(rd(d / 'final.f32'), fin),
         'final_display_max_code_diff': float(np.abs(disp(rd(d / 'final.f32')) - disp(fin)).max() * 255),
         'weight': cmp(rd(d / 'weight_g1.f32')[..., 0], w),
         'py_constraint': {k: diag[k] for k in ('support_pixels', 'changed_pixels', 'baseline_fallback_pixels', 'remaining_display_errors', 'new_negative_luminance')}}
    rows.append(r); print(json.dumps(r), flush=True)

ok = all(r['final_display_max_code_diff'] == 0 and all(r[k]['max_rel'] <= a.tol for k in ('composed', 'smoothed', 'final', 'weight'))
         and r['cpp']['remaining'] == r['py_constraint']['remaining_display_errors']
         and r['cpp'].get('new_negative_y', 0) == r['py_constraint']['new_negative_luminance'] for r in rows)
summary = {'parity_pass': ok, 'forbid_new_negative_y': a.forbid_new_negative_y, 'tolerance_rel': a.tol, 'frames': len(rows),
           'bit_exact_frames': sum(all(r[k]['max_abs'] == 0 for k in ('composed', 'smoothed', 'final', 'weight')) for r in rows), 'rows': rows}
if a.out: Path(a.out).write_text(json.dumps(summary, indent=1))
print('parity_pass', ok, 'bit_exact_frames', summary['bit_exact_frames'], '/', len(rows))
sys.exit(0 if ok else 1)
