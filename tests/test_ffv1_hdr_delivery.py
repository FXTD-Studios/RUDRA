import json
import shutil
import subprocess
import numpy as np
import pytest
from rudra.delivery.video import encode_sequence, _encode_frame, TARGETS


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg required')
def test_ffv1_hdr_preserves_rgb_pq16_codes_and_tags(tmp_path):
    ramp = np.geomspace(0.001, 1000, 64 * 48).reshape(48, 64)
    nits = np.stack([ramp, ramp * 0.5, ramp * 0.25], -1).astype(np.float32)
    frames = [nits, np.flip(nits, 1).copy()]
    movie = encode_sequence(iter(frames), tmp_path / 'hdr.mkv', target='ffv1rgb16',
                            fps=24, source_space='rec2020', shoulder=False)
    expected = b''.join(_encode_frame(f, TARGETS['ffv1rgb16'], 1000, 'rec2020', False)
                        .astype('<u2').tobytes() for f in frames)
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(movie), '-f', 'rawvideo',
                              '-pix_fmt', 'rgb48le', 'pipe:1'], check=True, capture_output=True).stdout
    assert decoded == expected
    info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json',
                                     str(movie)], check=True, capture_output=True).stdout)['streams'][0]
    assert info['pix_fmt'] == 'gbrp16le'
    assert info['bits_per_raw_sample'] == '16'
    assert info['color_range'] == 'pc'
    assert info['color_space'] == 'gbr'
    assert info['color_transfer'] == 'smpte2084'
    assert info['color_primaries'] == 'bt2020'
