"""Independently decode generated-review videos and compare saved EXR sample frames.

This verifies transport and the declared display transform, not grading quality.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rudra.delivery.aces_video import ACES2HDR1000View
from rudra.delivery.exr import read_exr
from rudra.delivery.video import colour_tags, expected_tags
from tools.summarize_aces_generated import sha


def verify(report_path, expected_frames=None):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    if expected_frames is None:
        registration=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/aces_v1_generated_review_20261024.json').read_text())
        expected_frames=next(x['expected_frames'] for x in registration['expected_inputs'] if x['clip']==report_path.parent.name)
    if not isinstance(expected_frames,int) or expected_frames<=0:raise ValueError('positive registered frame count required')
    movie = report['video']
    if sha(movie['path']) != movie['sha256']:
        raise ValueError('video hash mismatch')
    probe = json.loads(subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
         '-show_streams', '-of', 'json', movie['path']], check=True,
        capture_output=True).stdout)['streams'][0]
    samples = [r for r in report['frames'] if r.get('exr')]
    if not samples:
        raise ValueError('no saved EXR samples')
    select = '+'.join(f'eq(n\\,{r["frame"]})' for r in samples)
    raw = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', movie['path'], '-vf', f'select={select}',
         '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', 'pipe:1'],
        check=True, capture_output=True).stdout
    shape = (len(samples), int(probe['height']), int(probe['width']), 3)
    if len(raw) != int(np.prod(shape))*2:
        raise ValueError('decoded sample count mismatch')
    frames = np.frombuffer(raw, dtype='<u2').reshape(shape)
    view = ACES2HDR1000View()
    errors = []
    for row, frame in zip(samples, frames):
        if sha(row['exr']) != row['sha256']:
            raise ValueError('EXR hash mismatch')
        rgb, _ = read_exr(row['exr'])
        expected = (view.forward(rgb).astype(np.float64)*65535+.5).astype(np.uint16)
        errors.append(int(np.abs(frame.astype(np.int32)-expected.astype(np.int32)).max()))
    tags = colour_tags(movie['path'])
    checks = {
        'all_frames_present': int(probe['nb_read_frames']) == len(report['frames']) == expected_frames,
        'rgb16': probe['pix_fmt'] == 'gbrp16le' and probe.get('bits_per_raw_sample') == '16',
        'full_range': probe.get('color_range') == 'pc',
        'hdr_tags': all(tags.get(k) == v for k,v in expected_tags('ffv1rgb16').items()),
        'sample_codes_bit_exact': max(errors) == 0,
    }
    return {'report': str(report_path), 'report_sha256': sha(report_path),
            'checks': checks, 'pass': all(checks.values()),
            'sample_frames': [r['frame'] for r in samples], 'sample_max_code_errors': errors,
            'decoded_frame_count': int(probe['nb_read_frames']), 'tags': tags}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_v1_generated_review_20261024.json'))
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    registration=json.loads(a.registration.read_text())
    expected={r['clip']:r['expected_frames'] for r in registration['expected_inputs']}
    if not expected or len(expected)!=len(registration['expected_inputs']):raise ValueError('unique nonempty registered clips required')
    rows=[];found=[]
    for path in sorted(a.root.glob('*/report.json')):
        if path.parent.name not in expected:raise ValueError('unregistered export report')
        row=verify(path,expected[path.parent.name]);rows.append(row);found.append(path.parent.name)
        print(json.dumps(row), flush=True)
    a.out.write_text(json.dumps({'clips':rows,'pass':set(found)==set(expected) and all(r['pass'] for r in rows),'registration_sha256':sha(a.registration),'verifier_sha256':sha(__file__),
        'scope':'source-capped frame counts and sampled EXR-to-PQ exact codes; not visual, native, player or release acceptance'},indent=2))


if __name__=='__main__':main()
