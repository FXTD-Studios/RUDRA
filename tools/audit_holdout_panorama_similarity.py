"""CPU panorama similarity screen; flags candidates, never certifies independence."""
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2
import Imath
import numpy as np
import OpenEXR

WIDTH, HEIGHT = 64, 32


def signature(rgb):
    image = cv2.resize(np.asarray(rgb, np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    log_light = np.log(np.maximum(image, 1e-8)).mean(axis=2).astype(np.float64)
    log_light -= log_light.mean()
    norm = np.linalg.norm(log_light)
    if norm < 1e-8:
        raise ValueError('No usable image variation; manual review required')
    return log_light / norm


def similarity(a, b):
    # Correlation over all circular panorama yaw shifts, without model inference.
    correlations = np.fft.irfft(np.conj(np.fft.rfft(a, axis=1)) *
                               np.fft.rfft(b, axis=1), n=WIDTH, axis=1).sum(axis=0)
    shift = int(np.argmax(correlations))
    return float(correlations[shift]), shift


def load_rgb(path):
    f = OpenEXR.InputFile(str(path))
    try:
        window = f.header()['dataWindow']
        shape = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)
        rgb = np.stack([np.frombuffer(f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)),
                                     np.float32).reshape(shape) for c in ('R', 'G', 'B')], -1)
        if not np.isfinite(rgb).all():
            raise ValueError('Nonfinite source')
        return rgb
    finally:
        f.close()


def main():
    cv2.setNumThreads(2)
    evidence = Path('reports/ambientcg_holdout_candidates_20261025')
    out = evidence / 'panorama_similarity'
    out.mkdir(exist_ok=False)
    originals = {}
    for root in (Path('G:/datasets/sources/polyhaven'), Path('G:/rudra_v02/panoramas')):
        for p in sorted(root.glob('*.exr')):
            key = re.sub(r'_\d+k$', '', p.stem)
            if key not in originals or p.stat().st_size < originals[key].stat().st_size:
                originals[key] = p
    candidates = {}
    for a in json.loads((evidence / 'downloads.json').read_text()):
        path = next(x['path'] for x in a['files'] if x['path'].endswith('.exr'))
        candidates[a['asset_id']] = signature(load_rgb(path))
    registration = {'algorithm': '64x32 area-resized RGB, mean log channels, centered unit norm; max circular horizontal correlation.',
                    'flag_threshold': .98, 'candidate_ids': list(candidates),
                    'known_panorama_paths': {k: str(v) for k, v in originals.items()},
                    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'scope': 'Same-scene low-resolution yaw/exposure screen only; crops, local edits, missing sources and difficult illumination changes can escape detection.'}
    (out / 'registration.json').write_text(json.dumps(registration, indent=2))
    records = []; errors = []; positive_controls = []
    with (out / 'scores.jsonl').open('w') as log:
        for i, (asset, path) in enumerate(originals.items()):
            try:
                rgb = load_rgb(path); known = signature(rgb)
                if i < 3:
                    shifted = np.roll(rgb * 4, rgb.shape[1] // 4, axis=1)
                    score, _ = similarity(known, signature(shifted))
                    positive_controls.append({'asset_id': asset, 'exposure4_yaw90_score': score})
                    assert score > .99999
                for candidate, sig in candidates.items():
                    score, shift = similarity(sig, known)
                    row = {'candidate': candidate, 'known': asset, 'score': score,
                           'yaw_shift_bins': shift, 'flagged': score >= .98}
                    records.append(row); log.write(json.dumps(row) + '\n')
            except Exception as ex:
                errors.append({'path': str(path), 'error': repr(ex)})
            if (i + 1) % 50 == 0:
                log.flush(); print(f'{i+1}/{len(originals)} panoramas screened', flush=True)
    result = {'known_count': len(originals), 'candidate_count': len(candidates),
              'scores': len(records), 'errors': errors, 'positive_controls': positive_controls,
              'flagged': [r for r in records if r['flagged']],
              'nearest': {k: sorted([r for r in records if r['candidate'] == k],
                                    key=lambda r: -r['score'])[:5] for k in candidates},
              'holdout_certified': False, 'model_evaluated': False, 'scope': registration['scope']}
    (out / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    assert not errors, 'Incomplete screen; inspect failures without certifying holdout'


if __name__ == '__main__':
    main()
