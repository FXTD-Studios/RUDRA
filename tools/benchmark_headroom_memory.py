"""Isolate decoder/filter/raw-output threading and compare bounded native runs.

Uses existing local synthetic fixtures. No download, dataset edit or training.
Separate FFmpeg/ffprobe process peaks and memory traces distinguish decode from
source preparation. All float frame hashes must match, including slow consumers.
"""
import argparse
import json
from pathlib import Path
import shutil

from benchmark_headroom_frames import monitored, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    sources_group = parser.add_mutually_exclusive_group(required=True)
    sources_group.add_argument('--sources', type=Path, nargs='+')
    sources_group.add_argument('--generate-fixtures', type=Path, help='Create HD/UHD synthetic fixtures; never overwrite')
    parser.add_argument('--frames', type=int, default=240, choices=[24, 120, 240])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3, choices=range(1, 6))
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--profiles', nargs='+', default=['4,0,0', '4,0,1', '4,1,0', '4,1,1', '4,4,1', '1,0,1'])
    parser.add_argument('--modes', nargs='+', choices=['stream', 'sequential', 'legacy', 'feeder'], default=['stream', 'sequential'])
    parser.add_argument('--delay-ms', type=int, default=0, choices=range(251))
    args = parser.parse_args()
    if args.sweep and (args.baseline or args.delay_ms):
        parser.error('Sweep uses one diagnostic binary and no consumer delay')
    if args.delay_ms and args.modes != ['stream']:
        parser.error('Consumer delay requires --modes stream')
    binary = args.binary.resolve(strict=True)
    baseline = args.baseline.resolve(strict=True) if args.baseline else None
    ffmpeg_executable = shutil.which('ffmpeg')
    if not ffmpeg_executable:
        parser.error('Pinned FFmpeg must be on PATH')
    ffmpeg = Path(ffmpeg_executable).resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.generate_fixtures:
        args.generate_fixtures.mkdir(parents=True, exist_ok=False)
        sources = []
        for name, width, height in [('hd', 1920, 1080), ('uhd', 3840, 2160)]:
            movie = (args.generate_fixtures / f'{name}_{args.frames}frames_ffv1.mkv').resolve()
            monitored([
                str(ffmpeg), '-v', 'error', '-nostdin', '-n', '-f', 'lavfi', '-i',
                f'testsrc2=size={width}x{height}:rate=24000/1001,format=yuv444p10le,lutyuv=y=val*1.08,'
                'setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709',
                '-frames:v', str(args.frames), '-c:v', 'ffv1', '-level', '3', '-threads', '4',
                '-pix_fmt', 'yuv444p10le', '-color_range', 'tv', '-color_trc', 'bt709',
                '-color_primaries', 'bt709', '-colorspace', 'bt709', '-output_ts_offset', '5', str(movie),
            ], args.out / f'generate_{name}.log', timeout=180)
            sources.append(movie)
            print(f'Generated {movie.name}: {movie.stat().st_size/2**20:.1f} MiB', flush=True)
    else:
        sources = [p.resolve(strict=True) for p in args.sources]
    if len({p.stem for p in sources}) != len(sources):
        parser.error('Source stems must be distinct')
    summary = {
        'schema': 'headroom-memory-benchmark-v1',
        'binaries': {str(p): sha256(p) for p in [binary, *([baseline] if baseline else [])]},
        'ffmpeg': str(ffmpeg), 'ffmpeg_sha256': sha256(ffmpeg),
        'sources': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sources],
        'runs': [], 'consumer_delay_ms': args.delay_ms,
        'scope': 'CPU decode and consumer hash only; preparation separately timed; 180s maximum per case',
    }
    profiles = args.profiles
    for repeat in range(args.repeats):
        for source in sources:
            cases = [('sequential', p) for p in profiles] if args.sweep else [(m, None) for m in args.modes]
            cases = cases[repeat % len(cases):] + cases[:repeat % len(cases)]
            for mode, profile in cases:
                binaries = [('baseline', baseline), ('candidate', binary)] if baseline else [('candidate', binary)]
                if repeat % 2:
                    binaries.reverse()
                for variant, executable in binaries:
                    stem = f'{source.stem}_{repeat}_{variant}_{mode}' + (f'_{profile.replace(",", "-")}' if profile else '')
                    report = (args.out / f'{stem}.json').resolve()
                    argv = [str(executable), str(source), mode, str(report)]
                    if profile:
                        argv.append(profile)
                    elif args.delay_ms:
                        argv.append(f'delay={args.delay_ms}')
                    metrics = monitored(argv, args.out / f'{stem}.log', timeout=180, trace=True)
                    data = json.loads(report.read_text())
                    assert data['ok'], data
                    row = {'fixture': source.stem, 'repeat': repeat, 'variant': variant,
                           'profile': profile, 'report': str(report), **data, **metrics}
                    summary['runs'].append(row)
                    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
                    rss = max((p['peak_rss_bytes'] for p in metrics['process_peaks']
                               if p['name'] in ('ffmpeg', 'ffmpeg.exe')), default=0)
                    print(f'{stem}: {data["feed_ms"]:.1f} ms; FFmpeg {rss/2**20:.1f} MiB; '
                          f'tree {metrics["sampled_tree_working_set_bytes"]/2**20:.1f} MiB', flush=True)
    for source in sources:
        for legacy in (False, True):
            rows = [r for r in summary['runs'] if r['fixture'] == source.stem and (r['mode'] == 'legacy') == legacy]
            if rows:
                assert all(r['frame_xxh64'] == rows[0]['frame_xxh64'] for r in rows), 'Frame hash mismatch'
                assert all(len(r['frame_xxh64']) == r['frames'] for r in rows)
        for row in (r for r in summary['runs'] if r['fixture'] == source.stem and r['mode'] == 'stream'):
            assert row['decode_receipt_validated'] and row['decoder_starts'] == 1
            assert row['peak_pixel_bytes'] <= 268435456
    summary['frame_hash_parity'] = True
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
