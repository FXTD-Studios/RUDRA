"""Verify local FFV1 RGB 16-bit transport; this does not validate HDR colour transforms."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def verify(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    # Include every possible 16-bit code, so truncation to 10/12 bits is detectable.
    ramp = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    frame = np.stack((ramp, np.flip(ramp, 0), np.flip(ramp, 1)), axis=-1)
    frames = np.stack((frame, np.flip(frame, 1))).astype('<u2')
    raw = frames.tobytes()
    movie = out / 'rgb16_lossless.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb48le',
                    '-video_size', '256x256', '-framerate', '24', '-i', 'pipe:0',
                    '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'gbrp16le', str(movie)],
                   input=raw, check=True, capture_output=True)
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(movie), '-f', 'rawvideo',
                              '-pix_fmt', 'rgb48le', 'pipe:1'], check=True, capture_output=True).stdout
    probe = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_streams',
                                       '-of', 'json', str(movie)], check=True,
                                      capture_output=True).stdout)
    stream = probe['streams'][0]
    report = {'exact_roundtrip': decoded == raw, 'frames': 2,
              'pixel_format': stream['pix_fmt'], 'bits_per_raw_sample': stream.get('bits_per_raw_sample'),
              'input_sha256': hashlib.sha256(raw).hexdigest(),
              'decoded_sha256': hashlib.sha256(decoded).hexdigest(),
              'scope': '16-bit RGB codec transport only; no HDR transform or player compatibility claim'}
    (out / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['exact_roundtrip'] or stream['pix_fmt'] != 'gbrp16le':
        raise RuntimeError('16-bit transport failed')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    verify(parser.parse_args().out)
