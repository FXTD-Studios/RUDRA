"""Compare frozen/current synchronous streams with asynchronous playback.

Uses existing synthetic clips, no generation/download/training. Preparation is
separately timed; async feed starts from a prepared source. Each case is bounded
to 180 s. Includes one UHD slow-consumer comparison to exercise queue pressure.
"""
import argparse
import json
from pathlib import Path
import shutil
import statistics

from benchmark_headroom_frames import monitored, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, choices=range(1, 6), default=2)
    args = parser.parse_args()
    candidate = args.candidate.resolve(strict=True)
    baseline = args.baseline.resolve(strict=True)
    sources = [p.resolve(strict=True) for p in args.sources]
    if len({p.stem for p in sources}) != len(sources):
        parser.error('Distinct source stems required')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        parser.error('Pinned FFmpeg must be on PATH')
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {
        'schema': 'headroom-playback-benchmark-v1',
        'binaries': {str(p): sha256(p) for p in (baseline, candidate)},
        'ffmpeg': ffmpeg, 'ffmpeg_sha256': sha256(Path(ffmpeg)),
        'sources': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sources],
        'scope': 'CPU preparation/feed/hash; no rendering, display or export. 180s per case.', 'runs': [],
    }
    references = {}
    identities = {}

    def save():
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    def run(source, variant, executable, mode, repeat, delay):
        stem = f'{source.stem}_{repeat}_{variant}' + (f'_delay{delay}' if delay else '')
        report = (args.out / f'{stem}.json').resolve()
        argv = [str(executable), str(source), mode, str(report)]
        if delay:
            argv.append(f'delay={delay}')
        metrics = monitored(argv, args.out / f'{stem}.log', timeout=180, trace=True)
        data = json.loads(report.read_text())
        assert data['ok'], data
        assert data['decode_receipt_validated'] and data['decoder_starts'] == 1
        assert data['decoded_frames'] == data['frames'] == len(data['frame_xxh64'])
        assert data['peak_pixel_bytes'] <= 268435456
        if mode == 'async':
            assert data['shared_pixel_limit'] == 268435456
            assert data['submit_ms'] < 100, 'Request dispatch exceeded 100ms acceptance bound'
        reference = references.setdefault(source.stem, data['frame_xxh64'])
        assert reference == data['frame_xxh64'], 'Float pixel mismatch'
        identity = identities.setdefault(source.stem, data['cache_id'])
        assert identity == data['cache_id'], 'Content identity mismatch'
        summary['runs'].append({'fixture': source.stem, 'variant': variant,
                                'repeat': repeat, 'report': str(report), **data, **metrics})
        save()
        print(f'{stem}: feed {data["feed_ms"]:.1f}ms, '
              f'pixels {data["peak_pixel_bytes"]/2**20:.1f}MiB, '
              f'tree {metrics["sampled_tree_working_set_bytes"]/2**20:.1f}MiB', flush=True)

    variants = [('baseline', baseline, 'stream'), ('current_stream', candidate, 'stream'),
                ('async', candidate, 'async')]
    for repeat in range(args.repeats):
        for source in sources:
            order = variants if repeat % 2 == 0 else variants[::-1]
            for variant, executable, mode in order:
                run(source, variant, executable, mode, repeat, 0)
    # The last source is the supplied UHD fixture. Same frames, retained for
    # 100ms by the consumer; intentionally includes those waits in feed time.
    for variant, executable, mode in variants:
        run(sources[-1], variant, executable, mode, 0, 100)
    assert all(sha256(Path(s['path'])) == s['sha256'] for s in summary['sources'])
    summary['pixel_and_identity_parity'] = True
    summary['comparisons'] = []
    for source in sources:
        for delay in (0, 100):
            rows = [r for r in summary['runs'] if r['fixture'] == source.stem and r['consumer_delay_ms'] == delay]
            if not rows:
                continue
            comparison = {'fixture': source.stem, 'delay_ms': delay}
            for variant, _, _ in variants:
                cases = [r for r in rows if r['variant'] == variant]
                comparison[variant] = {
                    'runs': len(cases),
                    'feed_ms': statistics.median(r['feed_ms'] for r in cases),
                    'first_frame_ms': statistics.median(r['first_frame_ms'] for r in cases),
                    'prepare_ms': statistics.median(r['prepare_ms'] for r in cases),
                    'peak_pixels_mib': max(r['peak_pixel_bytes'] for r in cases)/2**20,
                    'peak_tree_mib': statistics.median(r['sampled_tree_working_set_bytes']/2**20 for r in cases),
                }
            summary['comparisons'].append(comparison)
    save()


if __name__ == '__main__':
    main()
