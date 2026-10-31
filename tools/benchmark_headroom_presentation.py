"""Compare frozen/current full-resolution headroom previews on local fixtures.

No generation, downloads or training. Each diagnostic runs eight frames through
both direct CPU and session preview paths, twice in reversed order. The outer
comparison reverses binary order on alternate repeats. CPU work includes decode,
grade, baseline and hashes; no GPU display, scopes or export. RSS is sampled,
not a process-wide cap. Each invocation has a 180-second timeout.
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
    parser.add_argument('--repeats', type=int, default=2, choices=range(1, 6))
    args = parser.parse_args()
    binaries = {name: path.resolve(strict=True) for name, path in
                [('baseline', args.baseline), ('candidate', args.candidate)]}
    sources = [p.resolve(strict=True) for p in args.sources]
    if len({p.stem for p in sources}) != len(sources):
        parser.error('Distinct source stems required')
    executables = {name: shutil.which(name) for name in ['ffmpeg', 'ffprobe']}
    if not all(executables.values()):
        parser.error('Pinned FFmpeg and ffprobe must be on PATH')
    args.out.mkdir(parents=True, exist_ok=False)
    identities = {str(p): sha256(p) for p in [*binaries.values(), *sources,
                  *(Path(p) for p in executables.values())]}
    summary = {'schema': 'headroom-preview-comparison-v1', 'ok': False,
               'scope': __doc__, 'sha256': identities, 'tools': executables,
               'binaries': {k: str(v) for k, v in binaries.items()}, 'cases': []}

    def save():
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    references = {}
    cache_ids = {}
    save()
    try:
        for repeat in range(args.repeats):
            for source in sources:
                order = list(binaries.items())
                if repeat % 2:
                    order.reverse()
                for variant, binary in order:
                    stem = f'{source.stem}_{repeat}_{variant}'
                    report = (args.out / f'{stem}.json').resolve()
                    metrics = monitored([str(binary), str(source), str(report)],
                                        args.out / f'{stem}.log', timeout=180, trace=True)
                    data = json.loads(report.read_text(encoding='utf-8'))
                    assert data['ok'], data
                    assert data['source_sha256'] == identities[str(source)], 'Source identity mismatch'
                    assert cache_ids.setdefault(source.stem, data['cache_id']) == data['cache_id']
                    assert len(data['runs']) == 4 and data['frames_per_run'] == 8
                    for run in data['runs']:
                        assert run['pixels_match'] and run['decoder_starts'] == 1
                        assert len(run['hashes']) == data['frames_per_run']
                        assert references.setdefault(source.stem, run['hashes']) == run['hashes'], 'Pixel mismatch'
                        if run['mode'] == 'session_preview':
                            assert run['source_pixel_peak_bytes'] <= run['source_pixel_limit_bytes']
                    summary['cases'].append({'fixture': source.stem, 'variant': variant,
                                             'repeat': repeat, 'report': str(report), **data, **metrics})
                    save()
                    duration = statistics.median(r['feed_render_hash_ms'] for r in data['runs']
                                                 if r['mode'] == 'session_preview')
                    print(f'{stem}: preview {duration:.1f} ms / 8 frames; '
                          f'parent peak {metrics["sampled_parent_working_set_bytes"]/2**20:.1f} MiB', flush=True)
        assert all(sha256(Path(p)) == digest for p, digest in identities.items()), 'Input/binary/tool changed'
        summary['comparisons'] = []
        for source in sources:
            comparison = {'fixture': source.stem}
            for variant in binaries:
                cases = [c for c in summary['cases'] if c['fixture'] == source.stem and c['variant'] == variant]
                runs = [r for c in cases for r in c['runs'] if r['mode'] == 'session_preview']
                duration = statistics.median(r['feed_render_hash_ms'] for r in runs)
                comparison[variant] = {
                    'runs': len(runs), 'median_preview_ms': duration,
                    'preview_fps': 8000 / duration,
                    'median_first_frame_ms': statistics.median(r['first_frame_ms'] for r in runs),
                    'median_prepare_ms': statistics.median(c['prepare_ms'] for c in cases),
                    'median_parent_peak_mib': statistics.median(c['sampled_parent_working_set_bytes'] / 2**20 for c in cases),
                    'median_tree_peak_mib': statistics.median(c['sampled_tree_working_set_bytes'] / 2**20 for c in cases),
                }
            comparison['preview_time_reduction_percent'] = 100 * (
                1 - comparison['candidate']['median_preview_ms'] / comparison['baseline']['median_preview_ms'])
            summary['comparisons'].append(comparison)
        summary['pixel_and_identity_parity'] = True
        summary['ok'] = True
    except BaseException as error:
        summary['error'] = str(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
