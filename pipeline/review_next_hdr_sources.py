"""Decode downloaded source data and record provenance, without training admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import OpenEXR
from PIL import Image,ImageDraw
from pipeline.collect_next_hdr_sources import ROOT,dump,sha

def read_exr(path):
    with OpenEXR.File(str(path)) as f:
        ch=f.channels()
        a=ch['RGB'].pixels if 'RGB' in ch else ch['RGBA'].pixels[...,:3]
        return np.asarray(a,dtype=np.float32),str(f.header().get('chromaticities','not recorded'))

def exr_qc(path):
    a,chroma=read_exr(path)
    if not np.isfinite(a).all():raise ValueError('Nonfinite source '+str(path))
    sample=a[::4,::4]
    return a,dict(path=str(path),sha256=sha(path),shape=list(a.shape),finite=True,chromaticities=chroma,
        minimum=float(a.min()),maximum=float(a.max()),above_one_fraction=float((sample>1).mean()),
        negative_fraction=float((sample<0).mean()),percentiles=np.percentile(sample,[1,50,99,99.9]).tolist())

def sheet(items,path):
    canvas=Image.new('RGB',(1200,180*((len(items)+4)//5)),(20,20,20));d=ImageDraw.Draw(canvas)
    for i,(name,im) in enumerate(items):
        im.thumbnail((240,155));x=i%5*240;y=i//5*180;canvas.paste(im,(x,y));d.text((x+3,y+158),name,fill='white')
    canvas.save(path)

def sihdr():
    out=ROOT/'sihdr';status=json.loads((out/'status.json').read_text())
    if status['stage']!='downloaded_pending_pixel_review':raise ValueError('Download incomplete')
    refs=sorted((out/'unpacked').rglob('*.exr'));inputs=sorted((out/'unpacked').rglob('*.png'))
    if len(refs)!=181 or len(inputs)!=362:raise ValueError('Unexpected SI-HDR counts')
    receipts=json.loads((out/'archives.json').read_text());expected={f['path']:f['sha256'] for r in receipts for f in r['files']}
    publisher_checks=[]
    for name in ('reference','input'):
        meta=json.loads((ROOT/'licence_evidence'/f'sihdr_{name}_metadata.json').read_text())
        p=out/'downloads'/(name+'.zip');h=hashlib.md5()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        if p.stat().st_size!=meta['sizeBytes'] or h.hexdigest()!=meta['checkSum']['value']:raise ValueError('Publisher archive checksum mismatch')
        publisher_checks.append(dict(file=str(p),publisher_md5=h.hexdigest(),size=p.stat().st_size))
    metrics=[];previews=[]
    for i,p in enumerate(refs):
        a,q=exr_qc(p)
        if q['sha256']!=expected[str(p)]:raise ValueError('Source changed')
        # Downloaded files are 1888x1280, despite the README's nominal 1920 width.
        if a.shape[:2]!=(1280,1888):raise ValueError('Unexpected reference geometry')
        q['source_scene_id']=p.stem;metrics.append(q)
        a=np.maximum(a,0)
        # Diagnostic exposure only, never written into source pixels or targets.
        a*=.18/max(float(np.median(a[::8,::8])),1e-8)
        a=(a/(1+a))**(1/2.2)
        previews.append((p.stem,Image.fromarray((np.clip(a,0,1)*255).astype('uint8'))))
        if len(previews)==30 or i==len(refs)-1:
            sheet(previews,out/f'contact_{i//30+1:02d}.jpg');previews=[]
        if (i+1)%30==0:print('[review] SIHDR',i+1,flush=True)
    for p in inputs:
        im=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
        if im is None or im.shape[:2]!=(1280,1888) or sha(p)!=expected[str(p)]:raise ValueError('Input decode/hash failed')
    if {p.stem for p in inputs}!={p.stem for p in refs}:raise ValueError('Unpaired scene IDs')
    dump(out/'pixel_review.json',dict(references=metrics,input_count=len(inputs),all_decoded=True,publisher_checks=publisher_checks,training_enabled=False,
        limits='Photographed multi-exposure HDR references; SDR inputs are simulated. No motion benchmark or absolute-nit calibration. Colour/scale review required.'))

def tos():
    out=ROOT/'tears_of_steel';rows=json.loads((out/'sequence_manifest.json').read_text())
    expected_counts={'03_2a':48,'04_4k':22,'04_5f':48,'06_1a':48}
    if len(rows)!=sum(expected_counts.values()):raise ValueError('Incomplete sequence acquisition')
    metrics=[];groups={};previews=[]
    for i,r in enumerate(rows):
        p=Path(r['path']);a,q=exr_qc(p)
        if a.shape!=(800,1920,3):raise ValueError('Unexpected ToS geometry')
        if q['sha256']!=r['receipt']['sha256']:raise ValueError('Changed sequence file')
        q['scene']=r['scene'];q['frame_number']=int(p.stem.rsplit('_',1)[-1]);metrics.append(q)
        groups.setdefault(r['scene'],[]).append(q['frame_number'])
        scene_index=sum(m['scene']==r['scene'] for m in metrics)-1
        if scene_index in (0,expected_counts[r['scene']]//2,expected_counts[r['scene']]-1):
            a=np.maximum(a,0);a=(a/(1+a))**(1/2.2)
            previews.append((p.stem,Image.fromarray((np.clip(a,0,1)*255).astype('uint8'))))
        if (i+1)%24==0:print('[review] ToS',i+1,flush=True)
    for scene,frames in groups.items():
        if len(frames)!=expected_counts[scene]:raise ValueError('Incomplete scene')
        if scene!='04_4k' and any(b-a!=1 for a,b in zip(frames,frames[1:])):raise ValueError('Nonconsecutive sequence')
    if set(groups)!=set(expected_counts):raise ValueError('Missing sequence')
    sheet(previews,out/'sequence_contact.jpg')
    dump(out/'sequence_review.json',dict(frames=metrics,all_decoded=True,
        continuous_sequences={k:v for k,v in groups.items() if k!='04_4k'},
        sparse_still_samples={'04_4k':groups['04_4k']},excluded_scene='00_1a: three nonfinite source frames; entire acquired section excluded',training_enabled=False,
        limits='Mixed live action/CG composited EXR, not camera RAW or an independently graded HDR master; colour primaries and scale need confirmation.'))

def overlap(novel=False):
    out=ROOT/('polyhaven_novel' if novel else 'polyhaven');selected=json.loads((out/'selection.json').read_text())
    catalog=json.loads((out/'metadata/catalog.json').read_text())
    aux=Path('G:/datasets/corpora/rudra_commercial_review_20261018_v1/auxiliary_fxtd_camera_polyhaven.jsonl')
    family=lambda s:re.sub(r'_\d+$','',s.removesuffix('_puresky'))
    site=lambda aid:tuple(round(float(x),2) for x in catalog.get(aid,{}).get('coords',[]) or [])
    old={}
    for line in aux.read_text().splitlines():
        r=json.loads(line)
        if r.get('licence_source')=='polyhaven':
            aid=re.sub(r'_\d+k$','',r['scene_id'].split('::')[-1]);old.setdefault(aid,set()).add(r['split'])
    blocked={k for k,v in old.items() if v&{'val','test'}};old_sites={site(k) for k in blocked if site(k)}
    checks=[]
    for r in selected:
        aid=r['asset_id'];reasons=[]
        if any(family(aid)==family(k) for k in blocked):reasons.append('historical held-out capture/family')
        if site(aid) and site(aid) in old_sites:reasons.append('historical held-out approximate location')
        checks.append(dict(asset_id=aid,previous_splits=sorted(old.get(aid,[])),blocked_reasons=reasons,
            next_role='excluded_from_next_train' if reasons else 'train_candidate_pending_pixel_dedup_and_pairing'))
    dump(out/'overlap_review.json',dict(reference=str(aux),reference_sha256=sha(aux),assets=checks,
        scope='Capture/family and rounded location heuristic only. Whole-image and cross-provider similarity still required.'))

def overlap_novel():
    overlap(novel=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=['sihdr','tos','overlap','overlap_novel']);a=p.parse_args();cv2.setNumThreads(1);globals()[a.group]()
