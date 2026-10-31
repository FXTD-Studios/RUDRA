"""CPU-only HEVC10 input to float EXR and PQ16 transport check, procedural data."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.rudra_aces_export import video_frames, probe_video, sha
from rudra.delivery.exr import read_exr
from rudra.delivery.aces_video import ACES2HDR1000View


def main():
    root = Path('reports/aces_hevc10_transport_20261025')
    root.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    width, height = 1024, 32
    # Original 10-bit code ramp, not an upconversion of eight-bit media.
    ramp = np.minimum(np.arange(width), 876).astype('<u2') + 64
    y = np.broadcast_to(ramp, (height, width)).copy()
    uv = np.full((height // 2, width // 2), 512, dtype='<u2')
    frame = y.tobytes() + uv.tobytes() + uv.tobytes()
    raw = frame * 2
    source = root / 'original_10bit_ramp.mkv'
    command = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv420p10le',
               '-s', f'{width}x{height}', '-r', '24', '-i', 'pipe:0', '-frames:v', '2',
               '-c:v', 'libx265', '-preset', 'fast', '-x265-params',
               'lossless=1:pools=2:frame-threads=1:log-level=error',
               '-pix_fmt', 'yuv420p10le', '-color_range', 'tv', '-colorspace', 'bt709',
               '-color_trc', 'bt709', '-color_primaries', 'bt709', str(source)]
    subprocess.run(command, input=raw, check=True)
    decoded_yuv = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(source),
                                           '-f', 'rawvideo', '-pix_fmt', 'yuv420p10le', 'pipe:1'])
    precise = list(video_frames(source, decode_bits=16))
    legacy = list(video_frames(source, decode_bits=8))
    checks = {'source_yuv_codes_exact': decoded_yuv == raw,
              'two_frames': len(precise) == len(legacy) == 2,
              'more_than_800_levels': np.unique(precise[0][0, :, 0]).size > 800,
              'legacy_at_most256_levels': np.unique(legacy[0][0, :, 0]).size <= 256,
              'monotonic_ramp': bool(np.all(np.diff(precise[0][0, :, 0]) >= 0))}
    checkpoint = Path('G:/rudra_training/aces_guarded_projected600_20261024/step600.pt')
    out = root / 'export'
    export = [sys.executable, 'tools/rudra_aces_export.py', str(source), '--checkpoint',
              str(checkpoint), '--out', str(out), '--device', 'cpu', '--video',
              '--ocio-threads', '2', '--colour-backtrack-all', '--clipped-smoothing', 'hybrid64']
    with (root / 'export.log').open('w') as log:
        subprocess.run(export, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((out / 'report.json').read_text())
    checks['default_decode16'] = report['provenance']['video_decode_bits'] == 16
    checks['exr_exact_readback'] = report['checks']['all_exr_bit_exact_readback']
    masters = [read_exr(path)[0] for path in sorted((out / 'exr').glob('*.exr'))]
    checks['float32_masters'] = len(masters) == 2 and all(m.dtype == np.float32 and np.isfinite(m).all() for m in masters)
    view = ACES2HDR1000View(threads=2)
    expected = np.stack([(view.forward(m).astype(np.float64) * 65535 + .5).astype('<u2') for m in masters])
    video = report['video']['path']
    decoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', video, '-f', 'rawvideo', '-pix_fmt', 'rgb48le', 'pipe:1'])
    checks['pq16_codes_exact'] = bool(np.array_equal(np.frombuffer(decoded, '<u2').reshape(expected.shape), expected))
    meta = probe_video(video)
    checks['pq16_metadata'] = meta['pix_fmt'] == 'gbrp16le' and meta['color_transfer'] == 'smpte2084' and meta['color_primaries'] == 'bt2020'
    result = {'checks': {k: bool(v) for k, v in checks.items()}, 'source': probe_video(source),
              'source_sha256': sha(source), 'checkpoint_sha256': sha(checkpoint),
              'precise_levels': int(np.unique(precise[0][0, :, 0]).size),
              'legacy_levels': int(np.unique(legacy[0][0, :, 0]).size),
              'encoding_command': command, 'scope': 'Original procedural ramp, lossless HEVC10, two CPU frames with guarded600 research weights. Not natural-footage, lossy compression, model quality or release qualification.'}
    (root / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    assert all(checks.values())


if __name__ == '__main__':
    main()
