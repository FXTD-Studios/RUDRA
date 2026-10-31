"""Build preregistered prospective stills, without model inference or training."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.prepare_next_hdr_training import perspective
from rudra.delivery.exr import read_exr
from rudra.aces_view import ACES2View


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    registration = Path('configs/experiments/ambientcg_prospective_stills_20261025.json')
    reg = json.loads(registration.read_text())
    assert sha('pipeline/prepare_next_hdr_training.py') == reg['projection']['implementation_file_sha256']
    output = Path('G:/datasets/corpora/ambientcg_prospective_stills_20261025')
    output.mkdir(exist_ok=False)
    view = ACES2View(); view.set_threads(2); cv2.setNumThreads(2)
    rows = []
    for asset in reg['sources']:
        assert sha(asset['path']) == asset['sha256']
        pano, _ = read_exr(asset['path'])
        luminance = pano @ np.array([.2722287168, .6740817658, .0536895174], np.float32)
        positive = luminance[luminance > 0]
        assert positive.size and np.isfinite(pano).all()
        scale = .18 / float(np.median(positive))
        folder = output / asset['asset_id']; folder.mkdir()
        for yaw in reg['projection']['yaw_degrees']:
            for pitch in reg['projection']['pitch_degrees']:
                projected = perspective(pano, yaw, pitch, width=512, height=288, hfov=90)
                for ev in reg['exposure']['ev_offsets']:
                    target = np.asarray(projected * (scale * 2.**ev), np.float32)
                    assert np.isfinite(target).all()
                    stem = f'yaw{yaw}_pitch{pitch}_ev{ev:+d}'
                    target_path = folder / (stem + '_reference.npy')
                    np.save(target_path, target)
                    clean = np.clip(view.forward(target), 0, 1)
                    for bits in reg['source_bits']:
                        quantized = np.rint(clean * (2**bits - 1)) / (2**bits - 1)
                        stored = np.rint(quantized.astype(np.float64) * 65535).astype(np.uint16)
                        path = folder / (stem + f'_codes{bits}.png')
                        assert cv2.imwrite(str(path), stored[..., ::-1])
                        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., ::-1]
                        assert np.array_equal(decoded, stored) and decoded.dtype == np.uint16
                        error = float(np.max(np.abs(decoded.astype(np.float64) / 65535 - quantized)))
                        assert error <= .5001 / 65535
                        rows.append({'scene_id': asset['asset_id'], 'yaw': yaw, 'pitch': pitch,
                                     'ev': ev, 'scene_normalization_scale': scale, 'source_bits': bits,
                                     'source_path': path.as_posix(), 'source_sha256': sha(path),
                                     'reference_path': target_path.as_posix(), 'reference_sha256': sha(target_path),
                                     'max_storage_error_codes255': error * 255,
                                     'split': 'prospective_only', 'used_for_optimization': False})
        print(asset['asset_id'], 'prepared', flush=True)
    assert len(rows) == reg['expected_input_rows'] == 48
    assert len({r['source_path'] for r in rows}) == 48
    report = {'rows': rows, 'registration_sha256': sha(registration), 'builder_sha256': sha(__file__),
              'view': view.identity, 'model_evaluated': False, 'used_for_optimization': False,
              'release_authorized': False, 'scope': reg['scope']}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    (output / 'registration.json').write_bytes(registration.read_bytes())
    (output / 'builder.py').write_bytes(Path(__file__).read_bytes())
    print('48 sources prepared; no model evaluation', flush=True)


if __name__ == '__main__':
    main()
