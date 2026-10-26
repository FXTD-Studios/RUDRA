"""Convert declared Linear Rec.709 HDR candidates to ACEScg without model inference."""
import hashlib
import json
from pathlib import Path
import sys

import Imath
import numpy as np
import OpenEXR
import PyOpenColorIO as ocio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rudra.delivery.exr import write_exr, read_exr
from rudra.delivery.colorspace import AP1_CHROMATICITIES


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    evidence = Path('reports/ambientcg_holdout_candidates_20261025')
    output = Path('G:/datasets/corpora/ambientcg_quarantine_acescg_20261025')
    output.mkdir(exist_ok=False)
    config_name = 'cg-config-v4.0.0_aces-v2.0_ocio-v2.5'
    config = ocio.Config.CreateFromBuiltinConfig(config_name)
    forward = config.getProcessor('Linear Rec.709 (sRGB)', 'ACEScg').getDefaultCPUProcessor()
    reverse = config.getProcessor('ACEScg', 'Linear Rec.709 (sRGB)').getDefaultCPUProcessor()
    rows = []
    for asset in json.loads((evidence / 'downloads.json').read_text()):
        source = Path(next(f['path'] for f in asset['files'] if f['path'].endswith('.exr')))
        supplied = source.with_name(source.name.replace('_HDR.exr', '.blend'))
        declaration = evidence / (supplied.stem + '_color.json')
        declared = json.loads(declaration.read_text())
        assert declared['autoexec_disabled']
        assert any(i['name'] == source.name and i['color_space'] == 'Linear Rec.709'
                   for i in declared['images'])
        expected = next(f['sha256'] for f in asset['files'] if f['path'] == source.as_posix())
        assert sha(source) == expected
        f = OpenEXR.InputFile(str(source)); header = f.header(); box = header['dataWindow']
        shape = (box.max.y - box.min.y + 1, box.max.x - box.min.x + 1)
        rgb = np.stack([np.frombuffer(f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)),
                                     np.float32).reshape(shape) for c in ('R', 'G', 'B')], -1)
        f.close()
        assert np.isfinite(rgb).all()
        master = rgb.copy(); forward.applyRGB(master.reshape(-1, 3))
        restored = master.copy(); reverse.applyRGB(restored.reshape(-1, 3))
        assert np.isfinite(master).all()
        assert np.allclose(rgb, restored, rtol=2e-6, atol=2e-5)
        path = output / (asset['asset_id'] + '_acescg_float32.exr')
        write_exr(path, master, half=False, chromaticities=AP1_CHROMATICITIES,
                  attributes={'rudra:sourceSpace': 'ap1', 'rudra:exposureScale': '1.0',
                              'rudra:status': 'quarantine-reference-not-model-output',
                              'rudra:sourceSHA256': expected})
        check, _ = read_exr(path)
        assert np.array_equal(check, master)
        f = OpenEXR.InputFile(str(path)); h = f.header()
        assert all(v.type == Imath.PixelType(Imath.PixelType.FLOAT) for v in h['channels'].values())
        f.close()
        rows.append({'asset_id': asset['asset_id'], 'source_sha256': expected,
                     'provider_blend_sha256': sha(supplied), 'declaration_sha256': sha(declaration),
                     'target': path.as_posix(), 'target_sha256': sha(path),
                     'float32_exact_readback': True,
                     'roundtrip_max_scaled_error': float((np.abs(restored - rgb) / (1 + np.abs(rgb))).max()),
                     'split': 'quarantine', 'holdout_certified': False, 'model_evaluated': False})
        print(asset['asset_id'], 'converted and verified', flush=True)
    report = {'rows': rows, 'config': config_name, 'ocio_version': ocio.__version__,
              'config_serialized_sha256': hashlib.sha256(config.serialize().encode()).hexdigest(),
              'source_space': 'Linear Rec.709 (sRGB)', 'target_space': 'ACEScg',
              'exposure_scale': 1.0, 'clipping': False, 'builder_sha256': sha(__file__),
              'scope': 'Color-coordinate conversion of provider half-float HDR references; no new captured precision, model inference or holdout certification.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    (output / 'builder.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
