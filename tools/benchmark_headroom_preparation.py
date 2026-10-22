"""Bounded source preparation/reuse/cancellation and pixel parity measurement.

Read existing local fixtures only; no dataset edits, training or downloads.
Three seeks (start, end, middle) create independent owners. Every seek hash must
match the current full sequential control, with receipts and budgets checked.
"""
import argparse
import json
from pathlib import Path
import shutil
import statistics

from benchmark_headroom_frames import monitored, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, choices=range(1, 6), default=2)
    parser.add_argument('--timeout', type=int, choices=[180, 300], default=180,
                        help='Maximum seconds per case; 300 allows three slow full UHD probes')
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    sources = [p.resolve(strict=True) for p in args.sources]
    if len({p.stem for p in sources}) != len(sources):
        parser.error('Source stems must be distinct')
    ffmpeg = shutil.which('ffmpeg')
    ffprobe = shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        parser.error('Pinned FFmpeg and ffprobe must be on PATH')
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {
        'schema': 'headroom-preparation-runs-v1', 'binary': str(binary), 'binary_sha256': sha256(binary),
        'ffmpeg': ffmpeg, 'ffmpeg_sha256': sha256(Path(ffmpeg)),
        'ffprobe': ffprobe, 'ffprobe_sha256': sha256(Path(ffprobe)),
        'sources': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sources],
        'scope': 'CPU preparation and three one-frame seeks; no render/display/export',
        'case_timeout_seconds': args.timeout,
        'runs': [],
    }

    def save():
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    def run(source, mode, repeat):
        stem = f'{source.stem}_{repeat}_{mode}'
        report = (args.out / f'{stem}.json').resolve()
        metrics = monitored([str(binary), str(source), mode, str(report)],
                            args.out / f'{stem}.log', timeout=args.timeout, trace=True)
        data = json.loads(report.read_text())
        assert data['ok'], data
        row = {'fixture': source.stem, 'repeat': repeat, 'report': str(report), **data, **metrics}
        summary['runs'].append(row)
        save()
        print(f'{stem}: {metrics["wall_seconds"]:.3f}s', flush=True)
        return row

    for source in sources:
        reference = run(source, 'sequential', 0)['frame_xxh64']
        identity = None
        for repeat in range(args.repeats):
            for mode in (['reopen', 'reuse'] if repeat % 2 == 0 else ['reuse', 'reopen']):
                data = run(source, mode, repeat)
                assert len(reference) == data['frames']
                assert [r['first'] for r in data['ranges']] == [0, len(reference)-1, len(reference)//2]
                for r in data['ranges']:
                    assert r['xxh64'] == reference[r['first']], 'Seek pixel mismatch'
                    assert r['complete'] and r['decoder_starts'] == 1
                    assert r['peak_pixel_bytes'] <= 268435456
                    identity = identity or r['cache_id']
                    assert identity == r['cache_id'], 'Source identity changed'
            data = run(source, 'cancel', repeat)
            assert data['cancel_ms'] < 2000, 'Cancellation exceeded 2s acceptance bound'
    summary['frame_hash_parity'] = True
    summary['comparisons'] = []
    for source in sources:
        rows = [r for r in summary['runs'] if r['fixture'] == source.stem]
        by_mode = {}
        for mode in ('reopen', 'reuse'):
            selected = [r for r in rows if r['mode'] == mode]
            by_mode[mode] = {
                'three_seeks_total_ms': statistics.median(r['total_ms'] for r in selected),
                'first_prepare_ms': statistics.median(r['ranges'][0]['prepare_ms'] for r in selected),
                'later_prepare_and_bind_ms': statistics.median(
                    q['prepare_ms'] + q['bind_ms'] for r in selected for q in r['ranges'][1:]),
                'later_first_frame_including_prepare_ms': statistics.median(
                    q['prepare_ms'] + q['bind_ms'] + q['first_frame_ms']
                    for r in selected for q in r['ranges'][1:]),
                'sampled_tree_peak_mib': statistics.median(r['sampled_tree_working_set_bytes'] / 2**20 for r in selected),
            }
        summary['comparisons'].append({'fixture': source.stem, **by_mode,
            'cancel_ms': [r['cancel_ms'] for r in rows if r['mode'] == 'cancel']})
    assert all(sha256(Path(s['path'])) == s['sha256'] for s in summary['sources']), 'Fixture changed'
    save()


if __name__ == '__main__':
    main()
