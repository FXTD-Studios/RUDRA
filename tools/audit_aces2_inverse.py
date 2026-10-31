"""Probe the official ACES 2 inverse; this does not migrate RUDRA's model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import PyOpenColorIO as ocio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('use a new report file')
    name = 'cg-config-v4.0.0_aces-v2.0_ocio-v2.5'
    config = ocio.Config.CreateFromBuiltinConfig(name)
    axis = np.unique(np.rint(np.linspace(0, 255, 33))).astype(np.float32) / 255
    cube = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    full = np.arange(256, dtype=np.float32) / 255
    uv = np.stack(np.meshgrid(full, full, indexing='ij'), axis=-1).reshape(-1, 2)
    faces = []
    for channel in range(3):
        for value in (0., 1.):
            face = np.empty((len(uv), 3), np.float32)
            face[:, channel] = value
            face[:, [k for k in range(3) if k != channel]] = uv
            faces.append(face)
    sets = {'cube33': cube, 'six_full_8bit_faces': np.concatenate(faces),
            'grey256': np.repeat(full[:, None], 3, axis=1)}
    result = {'ocio_version': ocio.__version__, 'config': name,
              'config_sha256': hashlib.sha256(config.serialize().encode()).hexdigest(),
              'working_space': 'ACEScg', 'view': 'ACES 2.0 - SDR 100 nits (Rec.709)',
              'sets': {}, 'scope': 'analytic inverse/forward color samples only; no trained model or native export integration',
              'release_authorized': False}
    for display in ('Rec.1886 Rec.709 - Display', 'sRGB - Display'):
        transform = ocio.DisplayViewTransform(src='ACEScg', display=display, view=result['view'])
        forward = config.getProcessor(transform).getDefaultCPUProcessor()
        transform.setDirection(ocio.TRANSFORM_DIR_INVERSE)
        inverse = config.getProcessor(transform).getDefaultCPUProcessor()
        results = {}
        for label, source in sets.items():
            scene = np.ascontiguousarray(source.copy())
            inverse.applyRGB(scene)
            restored = scene.copy()
            forward.applyRGB(restored)
            errors = np.abs(restored - source).max(axis=1) * 255
            worst = int(errors.argmax())
            results[label] = {'samples': len(source), 'finite': bool(np.isfinite(scene).all() and np.isfinite(restored).all()),
                              'within_1_code_fraction': float(np.mean(errors <= 1)),
                              'p999_code_error': float(np.quantile(errors, .999)), 'max_code_error': float(errors.max()),
                              'worst_source': source[worst].tolist(), 'worst_restored': restored[worst].tolist(),
                              'scene_min': float(scene.min()), 'scene_max': float(scene.max())}
        result['sets'][display] = results
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
