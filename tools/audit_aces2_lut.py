"""Feasibility of differentiable trilinear ACES2 inverse LUT; no model migration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import PyOpenColorIO as ocio
import torch
from torch.nn import functional as F


def sample_lut(lut, source):
    # LUT axes are B,G,R; grid_sample coordinate order is x=R,y=G,z=B.
    grid = (source * 2 - 1).reshape(1, 1, 1, -1, 3)
    return F.grid_sample(lut, grid, mode='bilinear', padding_mode='border',
                         align_corners=True)[0, :, 0, 0].T


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--size', type=int, default=65)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('new output required')
    if args.size < 2:
        raise ValueError('size must be >=2')
    torch.set_num_threads(2)
    name = 'cg-config-v4.0.0_aces-v2.0_ocio-v2.5'
    config = ocio.Config.CreateFromBuiltinConfig(name)
    view = 'ACES 2.0 - SDR 100 nits (Rec.709)'
    transform = ocio.DisplayViewTransform(src='ACEScg', display='Rec.1886 Rec.709 - Display', view=view)
    forward = config.getProcessor(transform).getDefaultCPUProcessor()
    transform.setDirection(ocio.TRANSFORM_DIR_INVERSE)
    inverse = config.getProcessor(transform).getDefaultCPUProcessor()
    axis = np.linspace(0, 1, args.size, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing='ij')
    table = np.stack((r,g,b), -1).reshape(-1,3).copy()
    inverse.applyRGB(table)
    lut = torch.from_numpy(table.reshape(args.size,args.size,args.size,3).transpose(3,0,1,2).copy())[None]
    rng = np.random.default_rng(20261023)
    sets = {'random8bit65536': rng.integers(0,256,(65536,3)).astype(np.float32)/255,
            'random_continuous65536': rng.random((65536,3),dtype=np.float32),
            'grey256': np.repeat((np.arange(256,dtype=np.float32)/255)[:,None],3,axis=1)}
    result = {'config': name, 'config_sha256': hashlib.sha256(config.serialize().encode()).hexdigest(),
              'ocio_version': ocio.__version__, 'view': view, 'size': args.size,
              'input': 'Rec709 Rec1886 display RGB codes', 'output': 'ACEScg',
              'sets': {}, 'release_authorized': False,
              'scope': 'synthetic analytic LUT feasibility only; no trained model, export or native integration'}
    for label, source in sets.items():
        x = torch.from_numpy(source).requires_grad_()
        predicted = sample_lut(lut,x)
        grad, = torch.autograd.grad(predicted.sum(), x)
        restored = predicted.detach().numpy().copy()
        forward.applyRGB(restored)
        exact = source.copy(); inverse.applyRGB(exact); forward.applyRGB(exact)
        errors = np.abs(restored-source).max(1)*255
        exact_errors = np.abs(exact-source).max(1)*255
        result['sets'][label] = {'within1_fraction': float(np.mean(errors<=1)),
                                'p999_code_error': float(np.quantile(errors,.999)),
                                'max_code_error': float(errors.max()),
                                'exact_inverse_within1_fraction': float(np.mean(exact_errors<=1)),
                                'finite_gradients': bool(torch.isfinite(grad).all()),
                                'finite_output': bool(torch.isfinite(predicted).all())}
    args.out.mkdir(parents=True)
    np.save(args.out/'inverse_lut.npy',lut.numpy())
    (args.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
