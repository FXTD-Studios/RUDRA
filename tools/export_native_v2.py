"""Export a v9 (full_context) ACES checkpoint as the native v2 staged contract.

The network part ("G1") is pure tensor code and becomes one TorchScript module:
    G1(source BCHW codes[0,1], baseline BCHW ACEScg) ->
        restored_source (B,3), highlight_fields (B,3), full_fields (B,3), weight (B,1)
Everything colour-managed stays outside the network and is done natively with
OCIO (same built-in config the checkpoint pins):
    1. baseline  = inverse_view(source)
    2. G1
    3. restored  = inverse_view(restored_source)
    4. compose   (staged_compose below; plain per-pixel arithmetic)
    5. hybrid clipped smoothing (training/aces_clipped_smoothing.py contract)
    6. appearance constraint (training/aces_colour_backtrack.py, forward_view)
    6b. the lock, optional (training/aces_partial_tighten.py, --lock-registration/--lock-arm):
        the registered arm, written to the manifest's "lock" so the engine runs it
    7. mask = weight where (weight>0 & max source>.5) else 0

`staged_reference` is the executable specification the C++ stages must match.
This script writes:  <out>/rudra_g1.ts, <out>/manifest.json, <out>/parity.json
and fails unless the staged reference reproduces model(x).acescg (pre-smoothing)
and the TorchScript G1 reproduces the eager G1.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.rudra_aces_export import load_model, CLIPPED_SMOOTHING
from training.aces_model import reconstruction_weight
from training.aces_reference import inverse_aces2_reference, reference_view

NATIVE_CONTRACT = 'rudra_native_aces_v2_staged'
NATIVE_MAJOR = 2
NATIVE_MAJOR_LOCK = 3   # 7 Nov 2026: a package with the lock; apps without it refuse major 3


class G1(nn.Module):
    """Tensor-only network stage of the v9 model (no OCIO inside)."""
    def __init__(self, model):
        super().__init__()
        if model.contract().get('full_context') != 'independent_pixel_rms':
            raise ValueError('v9 full_context checkpoint required')
        self.predictor = model.predictor
        self.restorer = model.restorer
        self.highlight_context = model.highlight_context
        self.full_context = model.full_context

    def forward(self, source, baseline):
        fields = self.predictor(source, baseline)
        weight = reconstruction_weight(source)
        ordinary = weight == 0
        codes = torch.where(ordinary, self.restorer(source), fields[:, :3])
        restored_source = (source + (.49 / 255) * torch.tanh(codes)).clamp(0, 1)
        hl = self.highlight_context(source, baseline)[:, 3:]
        full = self.full_context(source, baseline)[:, 3:]
        return restored_source, hl, full, weight


def staged_compose(source, baseline, restored, hl, full, weight):
    """Stage 4. All BCHW float32. Mirrors ACESModel/ACESContextModel/ACESFullContextModel."""
    scale = restored.abs().amax(1, keepdim=True) + .001
    top = source.amax(1, keepdim=True)
    high = (weight > 0) & (top > .5)
    full_support = (weight > 0) & (source.amin(1, keepdim=True) >= 254.5 / 255)
    shadow = (weight > 0) & (top <= .5)
    out = restored + weight * (scale * hl)
    out = torch.where(full_support, restored + weight * (scale * full), out)
    shadow_blend = torch.where(weight >= 1, baseline, weight * baseline + (1 - weight) * restored)
    out = torch.where(shadow, shadow_blend, torch.where(high, out, restored))
    return out, torch.where(high, weight, torch.zeros_like(weight))


def lock_arm(registration, name):
    """The registered lock arm `name` from a registration's arms.partial_tighten, checked to be one the
    native engine implements (core/aces_lock.hpp: mode lock, band 0, target dequant or source)."""
    reg = json.loads(Path(registration).read_text(encoding='utf-8'))
    arms = {t['name']: t for t in reg.get('arms', {}).get('partial_tighten', [])}
    if name not in arms:
        raise SystemExit(f'no arm {name!r} in {registration}')
    t = arms[name]
    if t.get('mode') != 'lock' or float(t.get('band_codes', 0.0)) != 0.0 or t.get('target', 'source') not in ('dequant', 'source'):
        raise SystemExit(f'{name}: the native engine implements mode lock, band 0, target dequant or source')
    if not 0.0 < float(t.get('strength', 1.0)) <= 1.0 or not 3 <= int(t.get('iters', 4)) <= 6:
        raise SystemExit(f'{name}: strength must be in (0, 1] and iters in 3..6')
    return {'name': name, 'mode': 'lock', 'strength': float(t.get('strength', 1.0)), 'target': t.get('target', 'source'),
            'iters': int(t.get('iters', 4)), 'band_codes': 0.0, 'dq_iters': int(t.get('dq_iters', 200)),
            'dq_max_lap': float(t.get('dq_max_lap', 1.5)), 'registration': Path(registration).as_posix(),
            'registration_sha256': sha(registration)}


def apply_lock(source_hwc, out, baseline_hwc, weight_hw, lock):
    """Stage 6b, as the engine runs it: tighten_partial in lock mode on the constraint's output."""
    from training.aces_partial_tighten import tighten_partial
    kw = {'target': 'dequant', 'dq_iters': lock['dq_iters'], 'dq_max_lap': lock['dq_max_lap']} if lock['target'] == 'dequant' else {}
    res, _ = tighten_partial(source_hwc, out, baseline_hwc, weight_hw, reference_view(), mode='lock', iters=lock['iters'],
                             band_codes=0.0, strength=lock['strength'], **kw)
    return res


def staged_reference(g1, source_hwc, smoothing='hybrid64', constrain=True, forbid_new_negative_y=False, ceiling=True, lock=None):
    """Full staged pipeline on one HxWx3 frame -> (acescg HxWx3, mask HxW, pre_smoothing HxWx3).
    ceiling (3 Nov 2026): the native engine's highlight ceiling (training/aces_ceiling.py) on the
    baseline and the restored source after their inverses and on the master before the constraint.
    lock (7 Nov 2026): lock_arm()'s settings, applied after the constraint (stage 6b)."""
    from training.aces_ceiling import cap_luminance_np, cap_luminance_torch, inverse_white_luma
    limit = inverse_white_luma() if ceiling else None
    x = torch.from_numpy(np.ascontiguousarray(source_hwc, np.float32)).permute(2, 0, 1)[None]
    with torch.inference_mode():
        baseline = inverse_aces2_reference(x)                       # stage 1 (OCIO)
        if limit is not None:
            baseline, _ = cap_luminance_torch(baseline, limit)
        restored_source, hl, full, weight = g1(x, baseline)         # stage 2
        restored = inverse_aces2_reference(restored_source)          # stage 3 (OCIO)
        if limit is not None:
            restored, _ = cap_luminance_torch(restored, limit)
        acescg, mask = staged_compose(x, baseline, restored, hl, full, weight)  # stage 4
    pre = acescg[0].permute(1, 2, 0).numpy().astype(np.float32)
    out = pre
    if smoothing != 'none':                                          # stage 5
        from training.aces_clipped_smoothing import smooth_clipped
        out, _ = smooth_clipped(out, source_hwc, **CLIPPED_SMOOTHING[smoothing])
    if limit is not None:                                            # the master, before the constraint
        out, _ = cap_luminance_np(np.asarray(out, np.float32), limit)
    if constrain:                                                    # stage 6 (OCIO forward)
        from training.aces_colour_backtrack import constrain_colour
        out, _ = constrain_colour(source_hwc, out, baseline[0].permute(1, 2, 0).numpy(), weight[0, 0].numpy(),
                                  reference_view(), include_fully_clipped=True,
                                  forbid_new_negative_y=forbid_new_negative_y)
        if lock is not None:                                         # stage 6b (OCIO forward)
            out = apply_lock(source_hwc, out, baseline[0].permute(1, 2, 0).numpy(), weight[0, 0].numpy(), lock)
    return out.astype(np.float32), mask[0, 0].numpy().astype(np.float32), pre


ONNX_LIMIT_CODES = 0.25   # final master, max |display(onnx) - display(torch)| in 8-bit codes; a discrete
                          # constraint step can flip on ULP-level field differences (measured 0.106 on the synthetic probe)
ONNX_LIMIT_REL = 0.005    # final master, max relative ACEScg difference


def export_onnx(g1, example, ex_base, path, probes, forbid_new_negative_y=False, lock=None):
    """rudra_g1.onnx (opset 17, dynamic H/W) + ORT CPU tolerance vs the LibTorch staged master."""
    import onnxruntime as ort
    from training.aces_reference import reference_view
    torch.onnx.register_custom_op_symbolic('aten::asinh', lambda g, x: g.op('Asinh', x), 9)
    names = ['restored_source', 'highlight_fields', 'full_fields', 'weight']
    torch.onnx.export(g1, (example, ex_base), str(path), input_names=['source', 'baseline'], output_names=names,
                      dynamic_axes={k: {2: 'h', 3: 'w'} for k in ['source', 'baseline'] + names}, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])

    def g_ort(x, b):
        return tuple(torch.from_numpy(v) for v in sess.run(None, {'source': x.numpy(), 'baseline': b.numpy()}))

    rows = []
    for src in probes:
        a, ma, _ = staged_reference(g1, src, forbid_new_negative_y=forbid_new_negative_y, lock=lock)
        b, mb, _ = staged_reference(g_ort, src, forbid_new_negative_y=forbid_new_negative_y, lock=lock)
        rows.append({'shape': list(src.shape), 'max_display_code_diff': float(np.abs(reference_view().forward(a) - reference_view().forward(b)).max() * 255),
                     'max_rel': float((np.abs(a - b) / (np.abs(a) + 1e-3)).max()), 'mask_exact': bool((ma == mb).all())})
    out = {'final_max_display_code_diff': max(r['max_display_code_diff'] for r in rows), 'final_max_rel': max(r['max_rel'] for r in rows),
           'mask_exact': all(r['mask_exact'] for r in rows), 'limit_display_codes': ONNX_LIMIT_CODES, 'limit_rel': ONNX_LIMIT_REL,
           'onnxruntime': ort.__version__, 'probes': rows}
    out['pass'] = out['final_max_display_code_diff'] <= ONNX_LIMIT_CODES and out['final_max_rel'] <= ONNX_LIMIT_REL and out['mask_exact']
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--frames', nargs='*', type=Path, default=[], help='HxWx3 .npy display codes or stills for parity')
    ap.add_argument('--smoothing', default='hybrid64', choices=('none',) + tuple(CLIPPED_SMOOTHING))
    ap.add_argument('--forbid-new-negative-y', action='store_true', help='appearance constraint also backtracks new negative AP1 luminance')
    ap.add_argument('--no-onnx', action='store_true', help='skip rudra_g1.onnx (ONNX Runtime / DirectML route)')
    ap.add_argument('--lock-registration', type=Path, help='registration holding the lock arm (arms.partial_tighten)')
    ap.add_argument('--lock-arm', help='the registered lock arm the package runs, e.g. lock_dqs050g')
    ap.add_argument('--provenance', type=Path, help='JSON written into the manifest as "provenance" (checkpoint, holdout)')
    a = ap.parse_args()
    if bool(a.lock_registration) != bool(a.lock_arm):
        ap.error('--lock-registration and --lock-arm go together')
    lock = lock_arm(a.lock_registration, a.lock_arm) if a.lock_arm else None
    provenance = json.loads(a.provenance.read_text(encoding='utf-8')) if a.provenance else None
    a.out.mkdir(parents=True, exist_ok=False)
    model, state = load_model(str(a.checkpoint), 'cpu')
    model.eval()
    g1 = G1(model).eval()
    rng = np.random.default_rng(7)
    probes = []
    for f in a.frames:
        if f.suffix == '.npy':
            probes.append(np.load(f).astype(np.float32))
        else:
            from tools.rudra_aces_export import read_still
            probes.append(read_still(f))
    # synthetic probe with clipped blobs, crushed shadows and ordinary gradients
    h, w = 96, 128
    yy, xx = np.mgrid[:h, :w] / np.array([h, w])[:, None, None]
    syn = np.stack([xx, yy, 1 - xx], -1).astype(np.float32)
    syn[20:50, 30:70] = 1.0
    syn[60:80, 90:120] = [1.0, .9, .7]
    syn[5:15, 5:40] = .01
    syn = np.clip(syn + rng.normal(0, .01, syn.shape), 0, 1).astype(np.float32)
    probes.insert(0, syn)

    example = torch.from_numpy(syn).permute(2, 0, 1)[None]
    with torch.inference_mode():
        ex_base = inverse_aces2_reference(example)
    scripted = torch.jit.trace(g1, (example, ex_base), check_trace=False)
    scripted = torch.jit.freeze(scripted.eval())
    ts = a.out / 'rudra_g1.ts'
    scripted.save(str(ts))
    loaded = torch.jit.load(str(ts))

    parity = []
    for i, src in enumerate(probes):
        x = torch.from_numpy(np.ascontiguousarray(src)).permute(2, 0, 1)[None]
        with torch.inference_mode():
            ref = model(x).acescg[0].permute(1, 2, 0).numpy()
            base = inverse_aces2_reference(x)
            e = g1(x, base); t = loaded(x, base)
        g1_err = max(float((p - q).abs().max()) for p, q in zip(e, t))
        # Against the unmodified model: the ceiling off (it is not part of model(x)).
        _, _, pre = staged_reference(g1, src, smoothing='none', constrain=False, ceiling=False)
        staged_err = float(np.abs(pre - ref).max())
        staged_rel = float((np.abs(pre - ref) / (np.abs(ref) + 1e-3)).max())
        parity.append({'probe': i, 'shape': list(src.shape), 'torchscript_vs_eager_max_abs': g1_err,
                       'staged_vs_model_max_abs': staged_err, 'staged_vs_model_max_rel': staged_rel})
        print(parity[-1], flush=True)
    ok = all(p['torchscript_vs_eager_max_abs'] <= 1e-4 and p['staged_vs_model_max_rel'] <= 1e-5 for p in parity)
    onnx = None
    if not a.no_onnx:
        onnx = export_onnx(g1, example, ex_base, a.out / 'rudra_g1.onnx', probes, a.forbid_new_negative_y, lock)
        print('onnx', {k: v for k, v in onnx.items() if k != 'probes'}, flush=True)
        ok = ok and onnx['pass']
    manifest = {
        'contract': NATIVE_CONTRACT, 'contract_major': NATIVE_MAJOR_LOCK if lock else NATIVE_MAJOR,
        'model_contract': state['contract'],
        'checkpoint_sha256': sha(a.checkpoint), 'g1_sha256': sha(ts), 'g1_file': ts.name,
        'g1_inputs': ['source BCHW float32 display codes [0,1] Rec709 Rec1886', 'baseline BCHW float32 ACEScg = OCIO inverse(source)'],
        'g1_outputs': ['restored_source B3HW', 'highlight_fields B3HW', 'full_fields B3HW', 'weight B1HW'],
        'ocio': {k: state['contract'].get(k) for k in ('ocio_version', 'config', 'view', 'display') if k in state['contract']},
        'stages': ['inverse_view(source)->baseline', 'G1', 'inverse_view(restored_source)->restored', 'compose',
                   'clipped_smoothing', 'appearance_constraint(include_fully_clipped)'] + (['lock'] if lock else []) + ['mask'],
        'compose': {'scale': 'max_c|restored|+0.001', 'high': 'w>0 & max(src)>0.5', 'full': 'w>0 & min(src)>=254.5/255',
                    'shadow': 'w>0 & max(src)<=0.5 -> w>=1 ? baseline : w*baseline+(1-w)*restored', 'else': 'restored'},
        'clipped_smoothing': None if a.smoothing == 'none' else {'name': a.smoothing, **CLIPPED_SMOOTHING[a.smoothing],
                                                                  'erode': 'cross 3x3 x2, border 0', 'components': 8,
                                                                  'gaussian': 'k=int(4*sigma)|1, BORDER_REFLECT_101'},
        'appearance_constraint': {'max_display_error_codes': 1, 'include_fully_clipped': True, 'method': 'halving backtrack toward baseline',
                                  'forbid_new_negative_y': bool(a.forbid_new_negative_y)},
        'onnx_file': None if onnx is None else 'rudra_g1.onnx',
        'onnx_sha256': None if onnx is None else sha(a.out / 'rudra_g1.onnx'),
        'onnx_tolerance': None if onnx is None else {k: onnx[k] for k in ('final_max_display_code_diff', 'final_max_rel', 'mask_exact', 'limit_display_codes', 'limit_rel')},
        'lock': lock,
        'provenance': provenance,
        'parity_pass': ok, 'release_authorized': False,
    }
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    (a.out / 'parity.json').write_text(json.dumps({'torchscript_and_staged': parity, 'onnx': onnx}, indent=1))
    print('parity_pass', ok)
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
