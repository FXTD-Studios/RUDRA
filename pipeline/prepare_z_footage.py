"""Isolated, resumable CPU intake/QC and paired rendering of owner-supplied Z: footage.

Never changes the active training run. PQ footage is inspected separately and
is not passed through a scene-linear ACES output transform a second time.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import cv2
import numpy as np
from pipeline import ingest_fxtd_footage as camera
from pipeline import render_pipeline_pairs as render
from pipeline.hdr_io import encode_hdr_u16, decode_hdr_u16, pq_eotf

INTAKE = REPO / 'reports/z_footage_intake_20261018'
DEFAULT_OUT = Path('G:/datasets/corpora/rudra_z_pipeline_20261018_v1')
FFMPEG = r'C:\Program Files\ffmpeg-2025\bin\ffmpeg.exe'
SPLITS = {'sudan': 'val', 'taha': 'train', 'mataha': 'test',
          'amin_elrihany': 'train', 'fashonista': 'val', 'fouad_hamza': 'train'}


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(temp, path)


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def write_rows(path, rows):
    temp = Path(str(path) + '.tmp')
    temp.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    os.replace(temp, path)


def digest(path, progress=False):
    h = hashlib.sha256()
    done = 0
    last = time.monotonic()
    with Path(path).open('rb') as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
            done += len(b)
            if progress and time.monotonic() - last > 20:
                print(f'[hash] {Path(path).name}: {done / 1e9:.1f} GB', flush=True)
                last = time.monotonic()
    return h.hexdigest()


def key(row):
    return hashlib.sha256(row['source_path'].casefold().encode()).hexdigest()[:16]


def take_id(row):
    name = Path(row['source_path']).stem
    match = re.search(r'(?:[A-Z]_\d{4}C\d{3}_\d{6}|[A-Z]\d{3}C\d{3,4}_\d{6}|EP_\d+_SC_\d+_SH_\d+)', name)
    return match.group(0).lower() if match else 'unresolved_' + key(row)


def unchanged(row):
    st = Path(row['source_path']).stat()
    if st.st_size != row['bytes'] or st.st_mtime_ns != row['mtime_ns']:
        raise ValueError('Source changed since intake: ' + row['source_path'])


def inputs():
    probes = {r['path']: r for r in read_rows(INTAKE / 'probes.jsonl')}
    rows = read_rows(INTAKE / 'intake_candidates_NOT_FOR_TRAINING.jsonl')
    rows = [r for r in rows if r['status'] == 'candidate_pending_qc']
    for r in rows:
        s = next(s for s in probes[r['source_path']]['probe']['streams'] if s['codec_type'] == 'video')
        rate = s.get('avg_frame_rate')
        if not rate or rate in ('0/0', '0/1'):
            rate = s['r_frame_rate']
        r['fps'] = float(Fraction(rate))
        n = s.get('nb_frames')
        r['frame_count_basis'] = 'stream_nb_frames' if n and n != 'N/A' else 'duration_times_fps'
        r['frame_count'] = int(n) if n and n != 'N/A' else int(float(r['duration_seconds']) * r['fps'] + .5)
        if r['frame_count'] < 1:
            raise ValueError('Empty source')
        r['take_id'] = take_id(r)
        r['split'] = SPLITS[r['production_group']]
    return rows


def decode(row, frame):
    unchanged(row)
    # Seek to the exact frame boundary of the intra-frame ProRes source.
    matrix = {'bt709': 'bt709', 'bt2020nc': 'bt2020'}.get(row['tagged_matrix'])
    if matrix is None:
        if row['production_group'] != 'fashonista':
            raise ValueError('Unreviewed YCbCr matrix')
        matrix = 'bt709'  # Explicit HD ProRes/MXF interpretation, retained in receipt.
    in_range = 'pc' if row['tagged_range'] == 'pc' else 'tv'
    filters = f'scale=in_color_matrix={matrix}:in_range={in_range}:out_range=pc'
    if row['source_interpretation'] != 'PQ / BT.2020':
        # Matrix-only YCbCr -> RGB: normalize reserved signaling so swscale
        # accepts the ProRes. Identical input/output transfer tags avoid any
        # transfer conversion; the actual LogC EOTF is applied separately.
        filters = 'setparams=color_primaries=bt709:color_trc=bt709,' + filters
    args = [FFMPEG, '-nostdin', '-v', 'error', '-threads', '1',
            '-ss', f"{frame / row['fps']:.9f}", '-i', row['source_path'],
            '-map', '0:v:0', '-frames:v', '1', '-threads', '1',
            '-vf', filters,
            '-filter_threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-']
    result = subprocess.run(args, capture_output=True, timeout=90)
    expected = row['width'] * row['height'] * 6
    if result.returncode or len(result.stdout) != expected:
        raise ValueError(f"Decode {frame}: {len(result.stdout)}/{expected}: {result.stderr[-1000:]!r}")
    code = np.frombuffer(result.stdout, dtype='<u2').reshape(row['height'], row['width'], 3)
    raw_hash = hashlib.sha256(result.stdout).hexdigest()
    return code.astype(np.float32) / 65535, raw_hash


def linear_camera(code, row):
    if row['source_interpretation'] == 'ARRI LogC4':
        return camera.decode_log(code, 'logc4'), 'logc4', 'awg4'
    if row['source_interpretation'] == 'ARRI LogC3 EI800':
        return camera.decode_log(code, 'logc3'), 'logc3', 'awg3'
    raise ValueError('Unsupported scene input: ' + str(row['source_interpretation']))


def shot_recipe(scene):
    # Pin the installed ACES 1.3 output family; never silently substitute an
    # unavailable config and never alter the active training environment.
    recipe = render.shot_recipe(scene, codecs=[])
    fam = render.FAMILIES[0]
    recipe.update(family=fam[0], config=fam[1], sdr_display=list(fam[2]),
                  hdr_display=list(fam[3]), hdr_peak_nits=fam[4])
    return recipe


def display_preview(code, row):
    if row['source_interpretation'] == 'PQ / BT.2020':
        # Contact-sheet preview only; not a paired SDR training target.
        x = pq_eotf(code).astype(np.float32) / 203
        x = camera.downsample(x, 480)
        x = x @ np.linalg.inv(render.M_709_2020).T
        x = np.maximum(x, 0)
        return np.rint(render._srgb_oetf(x / (1 + x)) * 255).astype(np.uint8)
    cam, _, gamut = linear_camera(code, row)
    x = np.maximum(camera.downsample(cam, 480) @ camera.camera_to_rec2020(gamut).T, 0)
    recipe = shot_recipe('fxtd_z_' + row['production_group'] + '_' + row['take_id'])
    recipe['codec'] = 'none'
    return render.render_pair(x, recipe)[2]


def qc(out):
    rows = inputs()
    results = []
    def inspect(row):
        k = key(row)
        record_path = out / 'qc' / (k + '.json')
        unchanged(row)
        if (record_path.exists() and json.loads(record_path.read_text()).get('qc_version') == 2
                and not json.loads(record_path.read_text()).get('error')):
            record = json.loads(record_path.read_text())
        else:
            record = dict(qc_version=2, source=row, samples=[], error=None)
            try:
                for frame in camera.even_picks(row['frame_count'], 3):
                    code, raw_hash = decode(row, frame)
                    preview = display_preview(code, row)
                    gray = cv2.cvtColor(cv2.resize(preview, (32, 32)), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
                    d = cv2.dct(gray)[:8, :8].ravel()[1:]
                    phash = sum(int(v > np.median(d)) << j for j, v in enumerate(d))
                    flat = gray.ravel() - gray.mean()
                    norm = np.linalg.norm(flat)
                    record['samples'].append(dict(frame=frame, decoded_sha256=raw_hash,
                        code_percentiles=np.percentile(code[::8, ::8], [0, 1, 50, 99, 100]).tolist(),
                        code_ceiling_fraction=float(np.mean(code.max(-1) >= .999)),
                        preview_mean=float(preview.mean() / 255), preview_std=float(gray.std()),
                        phash=str(phash), feature=(flat / max(float(norm), 1e-9)).tolist()))
                    if len(record['samples']) == 2:
                        render._write_png(out / 'previews' / (k + '.png'), preview)
                record['low_information'] = all(s['preview_std'] < .01 for s in record['samples'])
                record['near_black'] = all(s['preview_mean'] < .02 for s in record['samples'])
            except Exception as e:
                record['error'] = str(e)
            dump(record_path, record)
        return record
    # Build the one shared OCIO processor before workers start.
    recipe = shot_recipe('processor_warmup')
    render.processor(recipe['config'], *recipe['sdr_display'])
    render.processor(recipe['config'], *recipe['hdr_display'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, record in enumerate(pool.map(inspect, rows), 1):
            results.append(record)
            dump(out / 'status.json', dict(stage='qc', completed=i, total=len(rows), updated_unix=time.time()))
            print(f"[qc] {i}/{len(rows)} {Path(record['source']['source_path']).name}: {record['error'] or 'decoded'}", flush=True)
    write_rows(out / 'qc_results.jsonl', results)
    # Same-title similarities are grouping hints. Cross-title matches require review.
    flags = []
    for i, a in enumerate(results):
        if a['error']:
            continue
        for b in results[:i]:
            if b['error']:
                continue
            sa, sb = a['source'], b['source']
            if abs((sa['width']/sa['height'])/(sb['width']/sb['height'])-1) > .03:
                continue
            best = None
            for x in a['samples']:
                for y in b['samples']:
                    if min(x['preview_std'], y['preview_std']) < .01:
                        continue
                    ham = (int(x['phash']) ^ int(y['phash'])).bit_count()
                    corr = float(np.dot(x['feature'], y['feature']))
                    if x['decoded_sha256'] == y['decoded_sha256'] or (ham <= 6 and corr >= .98):
                        best = dict(hamming=ham, correlation=corr, frames=[x['frame'], y['frame']])
            if best:
                flags.append(dict(a=key(sa), b=key(sb), split_a=sa['split'], split_b=sb['split'],
                                  cross_split=sa['split'] != sb['split'], **best))
    dump(out / 'similarity_flags.json', flags)
    dump(out / 'qc_summary.json', dict(files=len(results), samples=sum(len(r['samples']) for r in results),
        decode_errors=sum(bool(r['error']) for r in results),
        low_information=sum(bool(r.get('low_information')) for r in results),
        near_black=sum(bool(r.get('near_black')) for r in results),
        similarity_flags=len(flags), cross_split_flags=sum(r['cross_split'] for r in flags),
        limits='Three sampled frames per file; does not prove whole-clip integrity or detect all crops/retimes.'))
    return results


def plan(out):
    results = read_rows(out / 'qc_results.jsonl')
    flags = json.loads((out / 'similarity_flags.json').read_text())
    blocked = {f[k] for f in flags if f['cross_split'] for k in ('a', 'b')}
    # Union similar same-production versions, in addition to camera take IDs.
    parent = {key(r['source']): key(r['source']) for r in results}
    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    by_take = {}
    for q in results:
        r = q['source']
        t = (r['production_group'], r['take_id'])
        k = key(r)
        if t in by_take:
            parent[root(k)] = root(by_take[t])
        by_take[t] = k
    for f in flags:
        if not f['cross_split']:
            # Same split alone does not imply same title; only merge within title.
            a = next(q['source'] for q in results if key(q['source']) == f['a'])
            b = next(q['source'] for q in results if key(q['source']) == f['b'])
            if a['production_group'] == b['production_group']:
                parent[root(f['a'])] = root(f['b'])
    blocked_roots = {root(k) for k in blocked}
    groups = defaultdict(list)
    decisions = []
    for q in results:
        r = q['source']
        reason = ('decode_error' if q['error'] else
                  'cross_split_similarity_review' if root(key(r)) in blocked_roots else
                  'low_information' if q.get('low_information') or q.get('near_black') else
                  'pq_separate_display_target_review' if r['source_interpretation'] == 'PQ / BT.2020' else None)
        if reason:
            decisions.append(dict(source_path=r['source_path'], decision=reason))
        else:
            groups[root(key(r))].append(r)
    selected = []
    for group in groups.values():
        # Keep one longest usable member of each take/version group, tie-break by path.
        group.sort(key=lambda r: (-r['frame_count'], -r['width']*r['height'], r['source_path']))
        selected.append(group[0])
        for r in group[1:]:
            decisions.append(dict(source_path=r['source_path'], decision='take_or_similar_version_not_selected',
                                  selected_path=group[0]['source_path']))
    selected.sort(key=lambda r: (r['production_group'], r['take_id']))
    payload = dict(schema_version=1, sources=selected, decisions=decisions,
        splits=SPLITS, frames_per_take=10, longest_side=1920,
        input_manifest_sha256=digest(INTAKE / 'intake_candidates_NOT_FOR_TRAINING.jsonl'),
        qc_results_sha256=digest(out / 'qc_results.jsonl'),
        render_convention='Existing ACES paired output transforms; standard AWG4/LogC4 and AWG3/LogC3 pairing.',
        decode_convention='16-bit RGB from ProRes; tagged YCbCr matrix retained, untagged FASHONISTA assumes BT.709; unspecified range treated as video/limited.',
        reserved_tag_handling='Log camera files use setparams to normalize transfer/primaries metadata for matrix-only swscale conversion; actual LogC curve/gamut applied separately.',
        pq_policy='Inspected separately; excluded from scene-linear ACES pairing.',
        release_authorized=False)
    dump(out / 'plan.json', payload)
    print(json.dumps(dict(selected_files=len(selected), by_production=dict(Counter(r['production_group'] for r in selected)),
                         decisions=dict(Counter(r['decision'] for r in decisions))), indent=2), flush=True)
    return payload


def prepare(out):
    p = json.loads((out / 'plan.json').read_text())
    if p['qc_results_sha256'] != digest(out / 'qc_results.jsonl'):
        raise ValueError('QC changed after planning')
    if p['input_manifest_sha256'] != digest(INTAKE / 'intake_candidates_NOT_FOR_TRAINING.jsonl'):
        raise ValueError('Intake changed after planning')
    code_files = [Path(__file__), REPO/'pipeline/ingest_fxtd_footage.py',
                  REPO/'pipeline/render_pipeline_pairs.py', REPO/'pipeline/hdr_io.py',
                  REPO/'training/prepare_training_data.py']
    code_identity = {str(f): digest(f) for f in code_files}
    code_receipt = out / 'preparation_code.json'
    if code_receipt.exists() and json.loads(code_receipt.read_text()) != code_identity:
        raise ValueError('Preparation code changed; use a fresh output directory')
    dump(code_receipt, code_identity)
    source_seen = {}
    all_records = []
    for i, row in enumerate(p['sources'], 1):
        if row['split'] != SPLITS[row['production_group']]:
            raise ValueError('Invalid production split in plan')
        unchanged(row)
        k = key(row)
        record_path = out / 'records' / (k + '.json')
        if record_path.exists():
            saved = json.loads(record_path.read_text())
            if saved['plan_sha256'] != digest(out / 'plan.json'):
                raise ValueError('Plan changed; use a fresh output directory')
            all_records.extend(saved['rows'])
            for r in saved['rows']:
                source_seen[r['source_sha256']] = row['source_path']
            continue
        source_hash_path = out / 'source_hashes' / (k + '.json')
        if source_hash_path.exists():
            identity = json.loads(source_hash_path.read_text())
            if identity['bytes'] != row['bytes'] or identity['mtime_ns'] != row['mtime_ns']:
                raise ValueError('Hash receipt identity mismatch')
        else:
            print(f"[hash] {i}/{len(p['sources'])} {Path(row['source_path']).name} ({row['bytes']/1e9:.2f} GB)", flush=True)
            identity = dict(source_path=row['source_path'], bytes=row['bytes'], mtime_ns=row['mtime_ns'],
                            sha256=digest(row['source_path'], progress=True))
            unchanged(row)
            dump(source_hash_path, identity)
        if identity['sha256'] in source_seen:
            dump(record_path, dict(plan_sha256=digest(out/'plan.json'), rows=[],
                                  duplicate_of=source_seen[identity['sha256']]))
            continue
        source_seen[identity['sha256']] = row['source_path']
        samples, peaks = [], []
        for frame in camera.even_picks(row['frame_count'], p['frames_per_take']):
            code, frame_hash = decode(row, frame)
            cam, encoding, gamut = linear_camera(code, row)
            if not np.isfinite(cam).all():
                raise ValueError('Nonfinite decoded scene')
            peaks.append(camera.frame_peak(cam))
            samples.append((frame, frame_hash, camera.downsample(cam, p['longest_side'])))
        top, clipped = camera.channel_plateaus(peaks)
        matrix = camera.camera_to_rec2020(gamut)
        camera_ceiling = camera.shot_ceiling_nits(top, clipped, matrix)
        scene = 'fxtd_z_' + row['production_group'] + '_' + row['take_id']
        recipe = shot_recipe(scene)
        # CPU-only JPEG/none recipes; no video encoder subprocess contention.
        ceiling = render.shot_ceiling(recipe, camera_ceiling)
        records = []
        for frame, frame_hash, small in samples:
            lin = np.maximum(small @ matrix.T, 0).astype(np.float32)
            sdr, hdr_lin, code8 = render.render_pair(lin, recipe)
            hdr, stats = encode_hdr_u16(hdr_lin, render.STORAGE)
            aid = f'{scene}_{k}_f{frame:07d}_pipe'
            lum = hdr_lin @ np.array([.2627, .6780, .0593], np.float32) * 203
            excluded = 'near_black' if float(code8.mean()/255) < .02 or stats['peak_nits'] < 1 else None
            record = dict(asset_id=aid, scene_id=scene, split=row['split'],
                project=row['production_group'], production_group=row['production_group'],
                source_id='z_' + row['production_group'], shot=row['take_id'], frame_index=frame,
                shot_frames=row['frame_count'], source_path=row['source_path'],
                source_sha256=identity['sha256'], decoded_frame_sha256=frame_hash,
                source_encoding=encoding, source_gamut=gamut, source_evidence=row['interpretation_evidence'],
                camera=row['source_interpretation'], width=sdr.shape[1], height=sdr.shape[0],
                sdr_path=str((out/'sdr'/f'{aid}.png').resolve()), hdr_path=str((out/'hdr'/f'{aid}.png').resolve()),
                sdr_kind='rendered', render_family='pipeline', pipeline_recipe=recipe,
                source_curve='unknown', sdr_curve=recipe['family'], sdr_codec=recipe['codec'],
                sdr_source='ACES SDR output, rendered not graded', hdr_source='ACES PQ HDR output',
                tonemap_ev=None, sdr_encoding='srgb2020_from_bt709', hdr_encoding='log2_extended',
                ceiling_nits=ceiling, peak_at_ceiling=bool(stats['peak_nits'] >= ceiling),
                peak_nits=stats['peak_nits'], clipped_fraction=stats['clipped_fraction'],
                sdr_clipped_fraction=float((code8.max(-1) >= 254).mean()), sdr_mean_code=float(sdr.mean()/65535),
                hdr_p50_nits=float(np.percentile(lum, 50)), hdr_p99_nits=float(np.percentile(lum, 99)),
                sensor_clipped_channels=clipped.tolist(), sensor_clip_camera=top.tolist(),
                is_video=False, excluded=excluded, licence_source='fxtd',
                licence_licence='Owner supplied FXTD project footage; internal training scope', commercial_ok=True)
            if not excluded:
                render._write_png(Path(record['sdr_path']), sdr)
                render._write_png(Path(record['hdr_path']), hdr)
            records.append(record)
        dump(record_path, dict(plan_sha256=digest(out/'plan.json'), rows=records))
        all_records.extend(records)
        dump(out/'status.json', dict(stage='render', completed=i, total=len(p['sources']), updated_unix=time.time()))
        print(f"[render] {i}/{len(p['sources'])} {scene}: {len(records)} pairs", flush=True)
    kept = [r for r in all_records if not r['excluded']]
    if len({r['asset_id'] for r in kept}) != len(kept):
        raise ValueError('Duplicate pair identities')
    write_rows(out/'sdr_hdr_manifest.jsonl', kept)
    config_path = out / '_ingest_config.json'
    # The sentinel is post-processed below (one family, this corpus), so on a
    # resume it would no longer match what write_sentinel() checks against.
    # Write it only once; a resumed run must belong to the same plan.
    if not config_path.exists():
        render.write_sentinel(out)
    config = json.loads(config_path.read_text())
    if config.get('plan_sha256') not in (None, digest(out/'plan.json')):
        raise SystemExit(f"{config_path}: written for a different plan; use a fresh --out")
    config['families'] = [config['families'][0]]
    config['families'][0]['weight'] = 1.0
    config.update(corpus='rudra_z_pipeline_20261018_v1', tool='pipeline/prepare_z_footage.py',
                  codec_pool=['none', 'jpeg'], plan_sha256=digest(out/'plan.json'))
    dump(config_path, config)
    dump(out/'preparation_summary.json', dict(pairs=len(kept), excluded_pairs=len(all_records)-len(kept),
        scenes=len({r['scene_id'] for r in kept}), by_split=dict(Counter(r['split'] for r in kept)),
        by_production=dict(Counter(r['production_group'] for r in kept)),
        source_bytes_hashed=sum(r['bytes'] for r in p['sources']), release_authorized=False,
        verification_pending=True))
    dump(out/'status.json', dict(stage='render_complete_verification_pending', pairs=len(kept), updated_unix=time.time()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['qc', 'plan', 'prepare', 'all'], default='all')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / 'preparation.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        if args.stage in ('qc', 'all'):
            qc(args.out)
        if args.stage in ('plan', 'all'):
            if (args.out/'records').exists():
                raise ValueError('Rendering already started; resume with --stage prepare')
            plan(args.out)
        if args.stage in ('prepare', 'all'):
            prepare(args.out)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
