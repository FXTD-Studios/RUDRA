"""CPU-only measurement of native zscale versus research swscale input codes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    root = ap.parse_args().out
    root.mkdir(exist_ok=False)
    width, height = 128, 64
    y, x = np.indices((height, width))
    # Original procedural 10-bit limited-range colour; no downloaded media.
    planes = [64 + (x * 7 + y * 3) % 877,
              256 + (x // 8 * 47 + y * 5) % 513,
              256 + (y // 8 * 59 + x * 3) % 513]
    raw = b''.join(p.astype('<u2').tobytes() for p in planes) * 2
    colour = root / 'original_colour10.mkv'
    encode = ['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'yuv444p10le',
              '-s', f'{width}x{height}', '-r', '24',
              '-color_range', 'tv', '-colorspace', 'bt709', '-color_trc', 'bt709',
              '-color_primaries', 'bt709', '-i', 'pipe:0', '-frames:v', '2',
              '-c:v', 'ffv1', '-level', '3', '-threads', '1', '-pix_fmt', 'yuv444p10le',
              '-color_range', 'tv', '-colorspace', 'bt709', '-color_trc', 'bt709',
              '-color_primaries', 'bt709', str(colour)]
    subprocess.run(encode, input=raw, check=True)
    roundtrip = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(colour),
                                        '-f', 'rawvideo', '-pix_fmt', 'yuv444p10le', 'pipe:1'])
    assert roundtrip == raw
    filters = {
        'research': 'scale=in_color_matrix=bt709:flags=accurate_rnd+full_chroma_int',
        'native': 'zscale=matrixin=709:rangein=limited:matrix=gbr:range=full,format=gbrp16le,format=rgb48le',
    }
    fixtures = [colour, Path('reports/aces_hevc10_transport_20261025/original_10bit_ramp.mkv')]
    records = []
    for source in fixtures:
        decoded = {}
        for name, vf in filters.items():
            command = ['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-noautorotate',
                       '-i', str(source), '-map', '0:v:0', '-vf', vf,
                       '-fps_mode', 'passthrough', '-pix_fmt', 'rgb48le', '-f', 'rawvideo', 'pipe:1']
            data = subprocess.check_output(command)
            decoded[name] = np.frombuffer(data, '<u2').astype(np.float64)
        a, b = decoded['research'], decoded['native']
        assert a.size == b.size and a.size > 0
        delta = (a - b) * 255 / 65535
        records.append(dict(source=str(source), source_sha256=digest(source), samples=int(a.size),
                            exact=bool(np.array_equal(a,b)), max_absolute_codes255=float(abs(delta).max()),
                            rms_codes255=float(np.sqrt(np.mean(delta**2))),
                            mean_signed_codes255=float(delta.mean()),
                            research_unique_codes=int(np.unique(a).size), native_unique_codes=int(np.unique(b).size)))
    report = dict(filters=filters, fixtures=records, colour_source_yuv_exact=True,
                  native_filter_source_sha256=digest(Path('native/media/src/video_probe.cpp')),
                  research_decoder_source_sha256=digest(Path('tools/rudra_aces_export.py')),
                  ffmpeg_version=subprocess.check_output(['ffmpeg','-version'], text=True).splitlines()[0],
                  scope='Two synthetic inputs, no model inference. Measures decoder differences; not a model-quality or broad video qualification.',
                  release_authorized=False)
    (root / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
