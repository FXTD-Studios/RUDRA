"""Prepare and run an owned-footage scratch candidate. No external weights."""
from __future__ import annotations
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SOURCE = Path('G:/datasets/corpora/rudra_commercial_review_20261018_v1/primary_fxtd_pipeline.jsonl')
SOURCE_HASH = 'e1871a61c180af34e96f74c993e9648ff78a32e3d8c7af352359854e6888aa74'
SPLITS = {'manchale':'train', 'lab':'train', 'elmatareed':'val', 'mataha':'test'}
ROOTS = {
    'manchale': Path(r'F:\CINEMA\IRAH\Manchale\Manchale_Song\(Footage)\Song\FOOTAGE'),
    'lab': Path(r'F:\CINEMA\IRAH\Lab_Scene\Project 4 folder\(Footage)\Footage'),
    'elmatareed': Path(r'F:\CINEMA\ELMATAREED\RAW'),
    'mataha': Path(r'F:\work\deaging\Mataha'),
}

def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def prepare(run):
    import cv2
    import numpy as np
    from pipeline.commercial_preflight import sha256, decoder_dependencies
    from training.sdr2hdr_dataset import corpus_ev_of
    cv2.setNumThreads(1)
    data = run / 'data'
    if data.exists():
        require((data / 'ready.json').is_file(), 'Incomplete preparation; use a new run name')
        ready = json.loads((data / 'ready.json').read_text())
        for name, digest in ready['hashes'].items():
            require(sha256(data / name) == digest, f'Prepared artifact changed: {name}')
        # Trainer independently rehashes every media and decoder dependency.
        return data
    raw = SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SOURCE_HASH, 'Selected source manifest changed')
    rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
    require(len(rows) == 1209, 'Unexpected source count')
    corpus_ev_of(SOURCE, rows, pipeline_only_ev=0)
    ids = set()
    groups = collections.defaultdict(set)
    media_root = Path('G:/datasets/corpora/rudra_fxtd_pipeline_20261017').resolve()
    for r in rows:
        sid = r.get('source_id')
        require(sid in SPLITS and r.get('licence_source') == 'fxtd' and r.get('commercial_ok') is True, 'Unapproved source')
        require(r['split'] == SPLITS[sid], 'Production split changed')
        require(r['asset_id'] not in ids, 'Duplicate identity')
        ids.add(r['asset_id'])
        source = Path(r['source_path']).resolve()
        require(source.is_relative_to(ROOTS[sid].resolve()) and source.is_file(), 'Original provenance mismatch')
        groups[r['scene_id']].add(r['split'])
        groups[str(source).casefold()].add(r['split'])
        for k in ('sdr_path','hdr_path'):
            require(Path(r[k]).resolve().is_relative_to(media_root), 'Unexpected media root')
    require(all(len(v)==1 for v in groups.values()), 'Scene/original path crosses splits')
    data.mkdir()
    features = []
    exact = collections.defaultdict(set)
    bytes_read = 0

    def inspect(r):
        r = dict(r)
        images = {}
        nbytes = 0
        for kind in ('sdr','hdr'):
            b = Path(r[kind+'_path']).read_bytes()
            nbytes += len(b)
            r[kind+'_sha256'] = hashlib.sha256(b).hexdigest()
            im = cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED)
            require(im is not None and im.ndim == 3 and im.shape[2] == 3, 'Invalid RGB pair')
            require(im.dtype in (np.uint8,np.uint16), 'Unexpected pixel type')
            images[kind] = im
        require(images['hdr'].dtype == np.uint16, 'HDR must have explicit uint16 storage')
        require(images['hdr'].shape == images['sdr'].shape, 'Pair geometry mismatch')
        sdr = images['sdr'].astype(np.float32) / np.iinfo(images['sdr'].dtype).max
        gray = cv2.cvtColor(cv2.resize(sdr,(32,32)),cv2.COLOR_BGR2GRAY)
        coeff = cv2.dct(gray)[:8,:8].flatten()[1:]
        bits = coeff > np.median(coeff)
        phash = sum(int(v)<<i for i,v in enumerate(bits))
        thumb = cv2.resize(gray,(16,16)).flatten()
        thumb -= thumb.mean()
        std = float(gray.std())
        norm = float(np.linalg.norm(thumb))
        feature = (phash, thumb/max(norm,1e-12), std, sdr.shape[1]/sdr.shape[0])
        return r, feature, nbytes

    checked = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(r,feature,nbytes) in enumerate(pool.map(inspect,rows),1):
            checked.append(r); features.append(feature); bytes_read += nbytes
            for kind in ('sdr','hdr'):
                exact[(kind,r[kind+'_sha256'])].add(r['split'])
            if i%100==0 or i==len(rows):
                print(f'[prepare] hashed and decoded {i}/{len(rows)} pairs',flush=True)
    require(all(len(v)==1 for v in exact.values()), 'Exact media duplicate crosses splits')
    flags = []
    for i,(ha,ta,sa,aa) in enumerate(features):
        if sa < .01:
            continue
        for j in range(i):
            hb,tb,sb,ab = features[j]
            if checked[i]['split']==checked[j]['split'] or sb<.01 or abs(aa/ab-1)>.03:
                continue
            if (ha ^ hb).bit_count()<=6 and float(np.dot(ta,tb))>=.98:
                flags.append([checked[i]['asset_id'],checked[j]['asset_id']])
    dump(data/'quality.json',{'pairs_decoded':len(checked),'bytes_hashed':bytes_read,
         'exact_cross_split_flags':0,'perceptual_cross_split_flags':flags,
         'low_information_unassessed':sum(f[2]<.01 for f in features),
         'limits':'SDR thumbnail screening may miss crops/reframing or HDR-only similarity; no calibrated image-quality certification.'})
    require(not flags, 'Perceptual split flags require review; see quality.json')
    manifest = data/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in checked),encoding='utf-8')
    evidence = data/'owner_scope.txt'
    evidence.write_text(
        'Owner instruction in this chat on 18 October 2026: prepare and start training give me the scripts to run the train.\n'
        'Scope: first scratch development run on selected FXTD owned-footage pipeline pairs only.\n'
        'Existing ingest records FXTD training-rights confirmation on 16 October. No third-party grant or distribution approval is asserted.\n'
        'Netflix, academic sources, Poly Haven auxiliary targets and older checkpoints are not used in this run.\n'
        'Source colour assumptions remain evaluation limitations; this is not a production-quality or legal release receipt.\n',encoding='utf-8')
    source_config = media_root/'_ingest_config.json'
    receipt = dict(schema_version=2,scope='internal-pilot',release_authorized=False,
        manifest_sha256=sha256(manifest),reviewed_by='Automated provenance verification; training requested by project owner in this chat',
        reviewed_at='2026-10-18',sources={s:dict(commercial_training=True,weight_distribution=False,
            evidence_path=evidence.name,evidence_sha256=sha256(evidence)) for s in SPLITS},
        decoder_dependencies={str(p):sha256(p) for p in decoder_dependencies(checked)},
        original_ingest_config_sha256=sha256(source_config))
    dump(data/'review.json',receipt)
    dump(data/'ready.json',{'source_manifest_sha256':SOURCE_HASH,
         'hashes':{p.name:sha256(p) for p in (manifest,evidence,data/'review.json',data/'quality.json')}})
    return data

def run_training(run, steps, hours):
    import torch
    from training import train_sdr2hdr as trainer
    require(torch.cuda.is_available(), 'CUDA is required; CPU fallback is disabled')
    torch.set_num_threads(4)
    data = prepare(run)
    output = run/'checkpoints'
    argv = ['--mode','image','--manifest',str(data/'manifest.jsonl'),
        '--commercial-evidence',str(data/'review.json'),'--output-dir',str(output),
        '--pipeline-corpus-ev','0','--device','cuda','--steps',str(steps),
        '--batch-size','4','--crop-size','256','--workers','0','--base-channels','32',
        '--lr','0.0001','--baseline-source','--baseline-dropout','0',
        '--degradation-probability','0.25','--augmentation-strength','0.5',
        '--crushed-hold','1','--crushed-hold-domain','preclamp',
        '--best-metric','composite_gain','--best-smoothing','1',
        '--eval-every','200','--eval-batches','0','--save-every','1000',
        '--log-every','20','--seed','20261018']
    previous = sys.argv
    try:
        sys.argv = ['train_sdr2hdr.py',*argv]
        args = trainer.parse_args()
    finally:
        sys.argv = previous
    dump(run/'recipe.json',{'trainer_args':argv,'max_hours_after_preflight':hours,
         'smoke_step':200,'smoke_max_regression_db':.5,'initialization':'scratch',
         'promotion':False,'resume_supported':False,
         'meaning':'Internal development candidate from commercial-eligible owned sources; not approved for release.'})
    started = None
    last_step = 0
    initial = {}
    done = threading.Event()
    status_lock = threading.Lock()
    def status(state,reason=None):
        with status_lock:
            dump(run/'status.json',{'state':state,'reason':reason,'pid':os.getpid(),
                'step':last_step,'target_steps':steps,'updated_unix':time.time(),
                'elapsed_seconds':time.monotonic()-started if started else 0,
                'checkpoint_directory':str(output),'release_authorized':False})
    def guard():
        while not done.wait(1):
            if (run/'STOP').exists() or (started and time.monotonic()-started>hours*3600):
                status('stopped','Stop requested or wall-time budget exhausted; saved checkpoints retained')
                os._exit(124)  # Zero loader workers: no orphan data processes.
    def observe(event,step,values):
        nonlocal started,last_step,initial
        last_step=step
        if event=='ready':
            started=time.monotonic()
            status('initial_validation')
        if event=='loss':
            require(all(bool(torch.isfinite(v).all()) for v in values.values()),'Nonfinite loss/output')
            if step==1 or step%20==0:
                status('training')
        if event=='eval':
            require(all(math.isfinite(float(v)) for v in values.values()),'Nonfinite validation')
            if step==0:
                initial=dict(values)
            if step==200:
                for key in ('clean_gain_db','hard_gain_db'):
                    require(values[key]>=initial[key]-.5,f'Smoke stopped: {key} regressed more than 0.5 dB')
                print('[smoke] numerical and validation non-regression checks passed; continuing',flush=True)
            status('training')
    threading.Thread(target=guard,daemon=True).start()
    try:
        result=trainer.train(args,observer=observe)
        status('completed')
        print(f'[complete] {result}; evaluation required before promotion',flush=True)
    except BaseException as e:
        status('failed',f'{type(e).__name__}: {e}')
        raise
    finally:
        done.set()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',required=True,type=Path)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--steps',type=int,default=20000)
    ap.add_argument('--max-hours',type=float,default=8)
    a=ap.parse_args()
    require(200<=a.steps<=50000 and 0<a.max_hours<=24,'Invalid step/time limit')
    a.run_dir.mkdir(parents=True,exist_ok=True)
    lock=a.run_dir/'runner.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Runner lock exists; use Status and a new RunName after a failed run')
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    try:
        dump(a.run_dir/'status.json',dict(state='preparing',pid=os.getpid(),step=0,updated_unix=time.time()))
        if a.prepare_only:
            prepare(a.run_dir)
            dump(a.run_dir/'status.json',dict(state='prepared',pid=os.getpid(),step=0,updated_unix=time.time()))
        else:
            run_training(a.run_dir,a.steps,a.max_hours)
    except BaseException as e:
        dump(a.run_dir/'status.json',dict(state='failed',reason=f'{type(e).__name__}: {e}',pid=os.getpid(),updated_unix=time.time()))
        raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    main()
