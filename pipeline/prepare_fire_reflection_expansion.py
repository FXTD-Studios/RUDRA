"""Freeze/convert/verify v2 fire/reflection candidate, without launching training."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import re
import sys
os.environ.setdefault('OMP_NUM_THREADS','2')
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pipeline import prepare_highlight_expansion as prep
from pipeline import commercial_preflight as commercial
from pipeline.verify_highlight_expansion import feature,contact
import cv2
import numpy as np

OUT=Path('G:/datasets/corpora/rudra_fire_reflection_expansion_20261018_v2')
SRC=Path('G:/datasets/sources/fire_reflection_expansion_20261018_v2')
BASE=Path('G:/datasets/corpora/rudra_highlight_expansion_20261018_v1')
BASE_MANIFEST=BASE/'combined_candidate_manifest.jsonl'
prep.OUT=OUT

def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return json.loads((OUT/'plan.json').read_text())
    sources=json.loads((SRC/'sources.json').read_text())
    prior=json.loads((BASE/'plan.json').read_text())
    hdr=next(x for x in prior['panorama_sources'] if x['asset_id']=='valley_of_desolation')
    if prep.sha(hdr['path'])!=hdr['sha256']:raise ValueError('HDR source changed')
    jobs=[]
    for source in sources:
        j=dict(id='v2_'+source['id'].lower(),frames=12,seed=202610180+len(jobs),split='train',
            source_asset_id=source['id'],source_receipt=str(SRC/'receipts'/(source['id']+'.json')),
            source_receipt_sha256=prep.sha(SRC/'receipts'/(source['id']+'.json')))
        paths=[Path(f['path']) for f in source['extracted_files']]
        if source['provider']=='ambientCG':
            maps={role:str(next(p for p in paths if p.name.endswith('_'+suffix+'.png'))) for role,suffix in
                [('Diffuse','Color'),('Rough','Roughness'),('Metal','Metalness'),('NormalGL','NormalGL')]}
            j.update(category='metal',material_maps=maps,hdri_path=hdr['path'],hdri_sha256=hdr['sha256'],hdri_capture=hdr['capture_group'])
        else:
            frames=sorted([p for p in paths if p.suffix.lower()=='.vdb'],key=lambda p:int(re.search(r'(\d+)\.vdb$',p.name).group(1)))
            start={'ground_explosion':55,'gasoline_explosion':35,'fire_plume':100}[source['id']]
            selected=frames[start:start+12]
            if len(selected)!=12:raise ValueError('Incomplete selected source sequence')
            j.update(category='fire_reflection',vdb_frames=[str(p) for p in selected],vdb_sha256=[prep.sha(p) for p in selected],
                source_frame_numbers=[int(re.search(r'(\d+)\.vdb$',p.name).group(1)) for p in selected],
                emission_strength=30,source_fps_verified=False)
        jobs.append(j)
    for category in ['glass','water','wet_floor','sparks']:
        jobs.append(dict(id='v2_'+category,category=category,frames=12,seed=202610180+len(jobs),split='train',
            source_asset_id='fxtd_authored_'+category))
    data=dict(schema_version=2,base_manifest=str(BASE_MANIFEST),base_manifest_sha256=prep.sha(BASE_MANIFEST),
        cg_jobs=jobs,width=640,height=360,samples=64,frames_per_sequence=12,fps=24,expected_new_pairs=sum(j['frames']*2 for j in jobs),
        scene_exposure_ev={'fire_reflection':-2.0},
        split_policy='All additions train-only, including every source/simulation/material variant. Earlier held-outs and rows retained byte-for-byte as JSON records.',
        source_manifest=str(SRC/'sources.json'),source_manifest_sha256=prep.sha(SRC/'sources.json'),
        colour_interpretation='Cycles linear Rec.709 EXR; relative radiometry; ACES 1.3 display HDR at 1000 nits and matched SDR; stored log2_extended.',
        limits=['Synthetic only; no new real-camera HDR','640x360 pilot resolution; 12 frames at assigned 24 fps (0.5 seconds) per setup',
            'VDB playback fps not verified; frame order preserved','Emissive values and HDRI relative scale not calibrated absolute luminance',
            'Procedural water is a moving bump shader, not a fluid simulation','No temporal training claim: existing image loader uses independent frames'],
        training_started=False,release_authorized=False)
    prep.dump(OUT/'plan.json',data);return data

def sentinel():
    d=json.loads((BASE/'_ingest_config.json').read_text());d.update(corpus=OUT.name,tool=__file__,plan_sha256=prep.sha(OUT/'plan.json'),
        licence='ambientCG/JangaFX/Poly Haven CC0 and FXTD-authored geometry; see source_evidence.json',trim_ev=0,codec_probability=0.5)
    prep.dump(OUT/'_ingest_config.json',d)

def convert(p):
    sentinel()
    for j in p['cg_jobs']:
        for frame in range(j['frames']):
            aid=f"{j['id']}_f{frame:04d}";rp=OUT/'render_records'/f'{aid}.json'
            rec=json.loads(rp.read_text());path=Path(rec['path'])
            if rec['source_job']!=j:raise ValueError('Stale render job')
            sid='jangafx_fire_reflection_v2' if j['category']=='fire_reflection' else 'ambientcg_metal_v2' if j['category']=='metal' else 'fxtd_reflection_v2'
            ev=p.get('scene_exposure_ev',{}).get(j['category'],0.0)
            prep.pair(prep.read_exr(path)*(2**ev),dict(asset_id=aid,scene_id=j['id'],sequence_id=j['id'],frame_index=frame,fps=24,
                source_id=sid,source_path=str(path),source_sha256=prep.sha(path),source_encoding='scene_linear',source_gamut='rec709',
                category=j['category'],source_asset_id=j['source_asset_id'],source_dependencies=j,render_record=str(rp),
                project='fire_reflection_expansion_v2',scene_exposure_ev=ev,radiometric_scale=2**ev,absolute_radiometry_verified=False))
        print('[pairs]',j['id'],flush=True)

def verify(p):
    if prep.sha(BASE_MANIFEST)!=p['base_manifest_sha256']:raise ValueError('Previous candidate changed')
    if prep.sha(SRC/'sources.json')!=p['source_manifest_sha256']:raise ValueError('Source inventory changed')
    sources=json.loads((SRC/'sources.json').read_text());source_checks=0
    for j in p['cg_jobs']:
        if j.get('hdri_path') and prep.sha(j['hdri_path'])!=j['hdri_sha256']:raise ValueError('Lighting source changed')
        if j.get('source_receipt') and prep.sha(j['source_receipt'])!=j['source_receipt_sha256']:raise ValueError('Source receipt changed')
        for path,digest in zip(j.get('vdb_frames',[]),j.get('vdb_sha256',[])):
            if prep.sha(path)!=digest:raise ValueError('Selected volume changed')
    for source in sources:
        for f in source['extracted_files']:
            if prep.sha(f['path'])!=f['sha256']:raise ValueError('Source changed '+f['path'])
            if Path(f['path']).suffix.lower()=='.png':
                im=cv2.imread(f['path'],cv2.IMREAD_UNCHANGED)
                if im is None:raise ValueError('Material map decode failed')
            source_checks+=1
    base=prep.rows(BASE_MANIFEST)
    new=[json.loads(f.read_text()) for f in sorted((OUT/'records').glob('*.json'))]
    if len(new)!=p['expected_new_pairs']:raise ValueError('Pair count mismatch')
    ids={f"{j['id']}_f{frame:04d}_{v}" for j in p['cg_jobs'] for frame in range(j['frames']) for v in ['clean','hard']}
    if {r['asset_id'] for r in new}!=ids:raise ValueError('Unexpected prepared records')
    variants=defaultdict(list);seq=defaultdict(list)
    for r in new:
        for kind in ['sdr','hdr']:
            im=cv2.imread(r[kind+'_path'],cv2.IMREAD_UNCHANGED)
            if im is None or im.dtype!=np.uint16 or im.shape!=(360,640,3):raise ValueError('Pair decode failed')
            if prep.sha(r[kind+'_path'])!=r[kind+'_sha256']:raise ValueError('Pair changed')
        hdr=prep.decode_hdr_u16(cv2.imread(r['hdr_path'],cv2.IMREAD_UNCHANGED),prep.render.STORAGE)
        if not np.isfinite(hdr).all():raise ValueError('Nonfinite target')
        if r['peak_nits']<=203 or r['highlight_fraction_above_203_nits']<=0:raise ValueError('No HDR highlight signal')
        variants[(r['sequence_id'],r['frame_index'])].append(r)
        if r['variant']=='clean':seq[r['sequence_id']].append(r)
    for v in variants.values():
        if len(v)!=2 or len({r['hdr_sha256'] for r in v})!=1 or len({r['sdr_sha256'] for r in v})!=2:raise ValueError('Invalid variant pair')
    for v in seq.values():
        if sorted(r['frame_index'] for r in v)!=list(range(12)) or len({r['hdr_sha256'] for r in v})!=12:raise ValueError('Sequence incomplete/static')
    held=[r for r in base if r['split']!='train'];flags=[]
    for kind in ['sdr','hdr']:
        hf=[feature(r[kind+'_path']) for r in held]
        for a in new:
            if kind=='hdr' and a['variant']=='hard':continue
            ha,fa,sa=feature(a[kind+'_path'])
            for b,(hb,fb,sb) in zip(held,hf):
                if min(sa,sb)>.01 and (ha^hb).bit_count()<=6 and float(np.dot(fa,fb))>=.98:flags.append(dict(new=a['asset_id'],held_out=b['asset_id'],kind=kind))
        print('[similarity]',kind,flush=True)
    prep.dump(OUT/'similarity_review.json',dict(cross_split_flags=flags,limits='Whole-frame pHash/correlation only; crops and semantic overlap not exhaustively screened. Base audit retained.'))
    if flags:raise ValueError('Cross-split review required')
    combined=base+new
    prep.write_rows(OUT/'sdr_hdr_manifest.jsonl',new);prep.write_rows(OUT/'combined_candidate_manifest.jsonl',combined)
    contact([r for r in new if r['variant']=='clean' and r['frame_index'] in [0,6,11]],OUT/'contact_sheet.jpg',columns=3)
    contact([r for r in new if r['variant']=='hard' and r['frame_index']==6],OUT/'degraded_contact_sheet.jpg',columns=3)
    approvals={sid:dict(v,evidence_path=str((BASE/v['evidence_path']).resolve())) for sid,v in json.loads((BASE/'commercial_review.json').read_text())['sources'].items()}
    deps=[OUT/'plan.json',SRC/'sources.json',SRC/'licence_evidence/jangafx.html',SRC/'licence_evidence/ambientcg.html',SRC/'licence_evidence/evidence.json',
        REPO/'pipeline/collect_fire_reflection_sources.py',REPO/'pipeline/render_fire_reflection_scenes.py',Path(__file__),REPO/'pipeline/prepare_highlight_expansion.py',
        REPO/'pipeline/render_highlight_scenes.py',REPO/'pipeline/render_pipeline_pairs.py']
    deps.extend(Path(j['source_receipt']) for j in p['cg_jobs'] if 'source_receipt' in j)
    evidence=dict(scope='internal commercial-development pilot only; no model release',owner_request='ok prepare it for next run',
        sources={'ambientCG':'Official asset CC0 grant','JangaFX':'Published free VDB CC0 grant; embedded licences retained','Poly Haven':'Previously audited CC0 HDRI; train-only capture reused',
            'FXTD':'Procedural geometry/shaders authored for this project'},
        changes='Authored CPU Cycles render, ACES SDR/HDR transforms, SDR-only clipped/JPEG variants',
        source_files_verified=source_checks,dependencies={str(f):prep.sha(f) for f in deps},release_authorized=False)
    prep.dump(OUT/'source_evidence.json',evidence)
    for sid in {r['source_id'] for r in new}:approvals[sid]=dict(commercial_training=True,weight_distribution=False,evidence_path='source_evidence.json',evidence_sha256=prep.sha(OUT/'source_evidence.json'))
    manifest=OUT/'combined_candidate_manifest.jsonl'
    review=dict(schema_version=2,scope='internal-pilot',release_authorized=False,manifest_sha256=prep.sha(manifest),reviewed_at='2026-10-18',
        reviewed_by='Automated source, split and pixel audit under owner preparation instruction',sources=approvals,
        decoder_dependencies={str(f):prep.sha(f) for f in commercial.decoder_dependencies(combined)},preparation_dependencies=evidence['dependencies'])
    prep.dump(OUT/'commercial_review.json',review)
    prep.dump(OUT/'preflight_result.json',commercial.validate(manifest,OUT/'commercial_review.json',progress=lambda x:print(x,flush=True)))
    from training.sdr2hdr_dataset import SDRHDRDataset
    import torch
    torch.set_num_threads(1)
    ds=SDRHDRDataset(OUT/'sdr_hdr_manifest.jsonl',split='train',crop_size=256,augment=False,degradation_probability=0)
    checks=[];seen=set()
    for i,r in enumerate(ds.records):
        key=(r['sequence_id'],r['variant'])
        if key in seen:continue
        sample=ds[i]
        for k in ['sdr','hdr']:
            if not torch.isfinite(sample[k]).all() or tuple(sample[k].shape)!=(3,256,256):raise ValueError('Loader failure')
        checks.append(dict(asset_id=r['asset_id'],hdr_max=float(sample['hdr'].max())));seen.add(key)
    prep.dump(OUT/'loader_smoke.json',dict(count=len(checks),checks=checks,training_performed=False))
    prep.dump(OUT/'preparation_summary.json',dict(stage='numeric_checks_passed_visual_review_pending',base_pairs=len(base),new_pairs=len(new),total_pairs=len(combined),
        by_split=dict(Counter(r['split'] for r in combined)),by_new_category=dict(Counter(r['category'] for r in new)),sequences=len(seq),frames_per_sequence=12,
        source_files_verified=source_checks,exact_cross_split_flags=0,perceptual_cross_split_flags=0,media_files_verified=len(combined)*2,
        loader_checks=len(checks),limits=p['limits'],training_started=False,release_authorized=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','convert','verify']);a=parser.parse_args();cv2.setNumThreads(1)
    p=plan()
    if a.action=='convert':convert(p)
    elif a.action=='verify':verify(p)
