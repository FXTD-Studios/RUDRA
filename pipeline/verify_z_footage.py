"""Verify isolated Z pairs with the existing owned corpus and write a review receipt.

Does not start training, approve distribution, or change the current run.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import cv2
import numpy as np
from pipeline import prepare_z_footage as z
from pipeline import commercial_preflight as commercial
from training.sdr2hdr_dataset import corpus_ev_of

BASE = Path('G:/datasets/corpora/rudra_commercial_review_20261018_v1/primary_fxtd_pipeline.jsonl')
BASE_HASH = 'e1871a61c180af34e96f74c993e9648ff78a32e3d8c7af352359854e6888aa74'
OLD_SPLITS = {'manchale':'train', 'lab':'train', 'elmatareed':'val', 'mataha':'test'}


def inspect(row):
    row = dict(row)
    arrays = {}
    features = {}
    for kind in ('sdr', 'hdr'):
        p = Path(row[kind+'_path'])
        row[kind+'_sha256'] = commercial.sha256(p)
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None or a.dtype != np.uint16 or a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f'Invalid {kind} image: {p}')
        arrays[kind] = a
        gray = cv2.cvtColor(cv2.resize(a, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)/65535
        d = cv2.dct(gray)[:8, :8].ravel()[1:]
        phash = sum(int(v > np.median(d)) << j for j, v in enumerate(d))
        flat = gray.ravel() - gray.mean()
        features[kind] = (phash, flat / max(float(np.linalg.norm(flat)), 1e-9), float(gray.std()))
    if arrays['sdr'].shape != arrays['hdr'].shape or arrays['hdr'].shape[:2] != (row['height'],row['width']):
        raise ValueError('Mismatched pair geometry')
    decoded = z.decode_hdr_u16(arrays['hdr'][::8,::8,::-1], z.render.STORAGE)
    if not np.isfinite(decoded).all():
        raise ValueError('Nonfinite HDR')
    return row, features


def verify(out):
    cv2.setNumThreads(1)
    if commercial.sha256(BASE) != BASE_HASH:
        raise ValueError('Original selection changed')
    old = z.read_rows(BASE)
    new = z.read_rows(out/'sdr_hdr_manifest.jsonl')
    for r in old:
        if r['split'] != OLD_SPLITS[r['source_id']]:
            raise ValueError('Original split changed')
    for r in new:
        if r['split'] != z.SPLITS[r['production_group']]:
            raise ValueError('Z production split changed')
    rows = old + new
    ids = [r['asset_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate asset IDs')
    checked, features = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(r,f) in enumerate(pool.map(inspect, rows),1):
            checked.append(r); features.append(f)
            if i%100 == 0 or i == len(rows):
                print(f'[pair-qc] {i}/{len(rows)} hashed and decoded', flush=True)
    exact = defaultdict(set)
    lineage = defaultdict(set)
    for r in checked:
        lineage[str(Path(r['source_path']).resolve()).casefold()].add(r['split'])
        for kind in ('sdr','hdr'):
            exact[(kind,r[kind+'_sha256'])].add(r['split'])
    if any(len(v)>1 for v in exact.values()) or any(len(v)>1 for v in lineage.values()):
        raise ValueError('Exact content or source path crosses splits')
    flags = []
    # Compare new against old and new across held-out splits. Old/old was screened
    # by the original run; repeat screening here for consistent final evidence.
    for i,a in enumerate(checked):
        for j in range(i):
            b = checked[j]
            if a['split'] == b['split'] or abs((a['width']/a['height'])/(b['width']/b['height'])-1) > .03:
                continue
            for kind in ('sdr','hdr'):
                ha,fa,sa = features[i][kind]
                hb,fb,sb = features[j][kind]
                if min(sa,sb)<.01:
                    continue
                if (ha^hb).bit_count() <= 6 and float(np.dot(fa,fb)) >= .98:
                    flags.append(dict(a=a['asset_id'],b=b['asset_id'],kind=kind))
    quality = dict(pairs_decoded=len(checked), new_pairs=len(new),
        by_split=dict(Counter(r['split'] for r in checked)), exact_cross_split_flags=0,
        perceptual_cross_split_flags=flags,
        limits='Full pair decoding plus sampled whole-frame SDR/HDR similarity. Crops, retimes and complete source footage are not exhaustively compared.',
        release_authorized=False)
    z.dump(out/'combined_quality.json', quality)
    if flags:
        raise ValueError('Cross-split similarity flags need review; combined_quality.json')
    manifest = out/'combined_manifest.jsonl'
    z.write_rows(manifest,checked)
    corpus_ev_of(manifest, checked, pipeline_only_ev=0)
    evidence = out/'owner_training_scope.txt'
    evidence.write_text(
        'Owner asked to use their own commercially usable footage, prepare/start training, supplied the ten Z: project roots, and said "all next" on 18 October 2026.\n'
        'This receipt covers isolated preparation and internal scratch training on those selected project sources plus the existing FXTD-only selection.\n'
        'LogC4 confirmation applies to Sudan, FASHONISTA and FOUAD HAMZA; other interpretations are tied to explicit camera/export metadata.\n'
        'Source interpretation, gamut/range assumptions and exclusions are recorded in the Z intake and plan.\n'
        'No Netflix, academic material, Poly Haven auxiliary target, PQ display-target candidate or inherited model weights are admitted.\n'
        'This is provenance verification under the owner request, not contract certification or weight-distribution approval.\n', encoding='utf-8')
    review = dict(schema_version=2, scope='internal-pilot', release_authorized=False,
        manifest_sha256=commercial.sha256(manifest), reviewed_at='2026-10-18',
        reviewed_by='Automated provenance and pixel verification under owner instruction',
        sources={sid:dict(commercial_training=True, weight_distribution=False,
                         evidence_path=evidence.name,evidence_sha256=commercial.sha256(evidence))
                 for sid in sorted({r['source_id'] for r in checked})},
        decoder_dependencies={str(p):commercial.sha256(p) for p in commercial.decoder_dependencies(checked)},
        preparation_dependencies={str(p):commercial.sha256(p) for p in
             (out/'plan.json',out/'qc_summary.json',out/'combined_quality.json',
              REPO/'pipeline/prepare_z_footage.py',REPO/'pipeline/verify_z_footage.py',
              REPO/'pipeline/ingest_fxtd_footage.py',REPO/'pipeline/render_pipeline_pairs.py',
              REPO/'training/prepare_training_data.py')})
    receipt = out/'commercial_review.json'
    z.dump(receipt,review)
    # Independently rehash media using the trainer's existing preflight.
    result = commercial.validate(manifest, receipt)
    z.dump(out/'preflight_result.json',result)
    z.dump(out/'ready.json',dict(manifest=str(manifest),manifest_sha256=commercial.sha256(manifest),
        receipt=str(receipt),receipt_sha256=commercial.sha256(receipt),
        pairs=len(checked),new_pairs=len(new),release_authorized=False))
    z.dump(out/'status.json',dict(stage='ready_for_separate_scratch_run',pairs=len(checked),new_pairs=len(new)))
    print(json.dumps(quality,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=z.DEFAULT_OUT)
    verify(parser.parse_args().out)
