"""Prepare the isolated native-1080p v3 addition; never launches training."""
import argparse
from collections import Counter, defaultdict
import copy
import json
import math
import os
from pathlib import Path
import sys
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import cv2
import numpy as np
import OpenEXR
from pipeline import prepare_highlight_expansion as old
from pipeline import commercial_preflight as commercial
from pipeline.fire_reflection_readiness import check_ready as check_base
from pipeline.verify_highlight_expansion import feature,contact

SRC=Path('G:/datasets/sources/next_hdr_sources_20261019_v3')
BASE=Path('G:/datasets/corpora/rudra_fire_reflection_expansion_20261018_v2')
OUT=Path('G:/datasets/corpora/rudra_native1080_expansion_20261019_v3')
W,H=1920,1080
sha,dump,rows=old.sha,old.dump,old.rows

def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():return json.loads((OUT/'plan.json').read_text())
    check_base(BASE)
    sources=[];excluded=[];deps={}
    for group in ('polyhaven','polyhaven_novel'):
        folder=SRC/group
        review=json.loads((folder/'overlap_review.json').read_text())
        allowed={r['asset_id'] for r in review['assets'] if not r['blocked_reasons']}
        excluded.extend(r for r in review['assets'] if r['blocked_reasons'])
        for r in json.loads((folder/'verified_files.json').read_text()):
            if r['asset_id'] not in allowed:continue
            if sha(r['path'])!=r['sha256']:raise ValueError('Changed panorama source')
            sources.append(dict(asset_id=r['asset_id'],path=r['path'],sha256=r['sha256'],category=r['category'],group=group))
            deps[r['path']]=r['sha256']
        for name in ('overlap_review.json','verified_files.json','selection.json'):
            deps[str(folder/name)]=sha(folder/name)
    if len(sources)!=17 or len(excluded)!=6:raise ValueError('Source selection changed; review before planning')
    prior=json.loads((BASE/'plan.json').read_text());byid={j['id']:j for j in prior['cg_jobs']}
    jobs=[]
    selected=[('v2_fire_plume',0),('v2_fire_plume',6),('v2_metal063',0),('v2_metal049a',6),
              ('v2_glass',0),('v2_water',6),('v2_wet_floor',0),('v2_sparks',6)]
    for n,(name,frame) in enumerate(selected):
        j=copy.deepcopy(byid[name]);j.update(id=f'v3_native_{n:02d}_{j["category"]}',sample_frame=frame,parent_scene_id=name,frames=1)
        jobs.append(j)
        paths=list(j.get('material_maps',{}).values())
        if j.get('hdri_path'):paths.append(j['hdri_path'])
        if j.get('source_receipt'):paths.append(j['source_receipt'])
        if j.get('vdb_frames'):paths.append(j['vdb_frames'][frame])
        for path in paths:deps[path]=sha(path)
    p=dict(schema_version=3,base_manifest=str(BASE/'combined_candidate_manifest.jsonl'),base_manifest_sha256=sha(BASE/'combined_candidate_manifest.jsonl'),
        base_ready_sha256=sha(BASE/'ready.json'),panorama_sources=sources,excluded_sources=excluded,cg_jobs=jobs,
        width=W,height=H,views_per_panorama=4,hfov=90,variants=['clean','hard'],expected_new_pairs=152,expected_total_pairs=3231,
        scene_exposure_ev=prior['scene_exposure_ev'],source_dependencies=deps,seed=20261019,
        colour_interpretation='Relative scene-linear Rec.709 interpretation for Poly Haven; authored Cycles is linear Rec.709. Shared ACES 1.3 SDR and 1000-nit HDR display transforms. No absolute source-nit claim.',
        exclusions='SI-HDR and Tears of Steel stay source/title-held-out evaluation candidates. Classroom and particle demos remain source assets; no unvalidated legacy scenes admitted.',
        limits=['New images are native 1920x1080 projections/renders, not enlarged 360p pairs.',
                'Panoramas are static photographs; CG additions are eight selected stills, not temporal training data.',
                'CG setups reuse eight existing training setups/frames at higher render resolution, not eight independent new scenes.',
                'The existing 3079 pairs and every validation/test row are retained exactly.',
                'The image trainer still samples 256x256 crops. Full-resolution inputs do not certify 4K output quality.',
                'Real-camera moving fire/sparks and independently graded HDR video remain gaps.'],training_started=False,release_authorized=False)
    dump(OUT/'plan.json',p)
    sentinel=json.loads((BASE/'_ingest_config.json').read_text());sentinel.update(corpus=OUT.name,tool=__file__,plan_sha256=sha(OUT/'plan.json'),created='2026-10-19',source_colour_interpretation=p['colour_interpretation'])
    dump(OUT/'_ingest_config.json',sentinel)
    return p

def perspective(pano,yaw,pitch,width=W,height=H,hfov=90):
    half=math.tan(math.radians(hfov)/2)
    xx,yy=np.meshgrid((np.arange(width)+.5-width/2)/(width/2)*half,(np.arange(height)+.5-height/2)/(width/2)*half)
    d=np.stack([xx,-yy,np.ones_like(xx)],-1);d/=np.linalg.norm(d,axis=-1,keepdims=True)
    y,p=math.radians(yaw),math.radians(pitch)
    ry=np.array([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]])
    rx=np.array([[1,0,0],[0,math.cos(p),-math.sin(p)],[0,math.sin(p),math.cos(p)]])
    v=d@(ry@rx).T
    # OpenCV coordinates address pixel centres: subtract 0.5 before padding offset.
    u=(np.arctan2(v[...,0],v[...,2])/(2*np.pi)+.5)*pano.shape[1]-.5+1
    t=(.5-np.arcsin(np.clip(v[...,1],-1,1))/np.pi)*pano.shape[0]-.5
    pad=np.concatenate([pano[:,-1:],pano,pano[:,:1]],axis=1)
    return cv2.remap(pad,u.astype('float32'),t.astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def save_pair(image,identity):
    if image.shape!=(H,W,3) or not np.isfinite(image).all():raise ValueError('Invalid native-resolution source')
    # Reuse the exact established display/storage transforms in this process only.
    old.OUT=OUT
    result=old.pair(image,identity)
    for r in result:
        r.update(width=W,height=H,is_video=False,sequence_available=False,source_resolution=[W,H],preparation_version=3)
        dump(OUT/'records'/(r['asset_id']+'.json'),r)

def panoramas(p):
    for i,a in enumerate(p['panorama_sources']):
        if sha(a['path'])!=a['sha256']:raise ValueError('Changed panorama')
        pano=old.read_exr(a['path'])
        if pano.shape!=(4096,8192,3):raise ValueError('Expected 8K panorama')
        lum=np.maximum(pano[::8,::8],0)@np.array([.2126,.7152,.0722],np.float32)
        small=cv2.resize(lum,(512,256),interpolation=cv2.INTER_AREA)
        py,px=np.unravel_index(np.argmax(small),small.shape)
        yaw=((px+.5)/512-.5)*360;pitch=-((.5-(py+.5)/256)*180)
        scale=float(np.clip(.18/max(float(np.median(lum)),1e-4),.01,100))
        for view in range(4):
            yy=yaw+view*90;pp=float(np.clip(pitch,-30,15)) if view==0 else 0.
            aid=f'v3_ph_{a["asset_id"]}_v{view}'
            image=perspective(pano,yy,pp)*scale
            path=OUT/'linear_exr'/(aid+'.exr');path.parent.mkdir(parents=True,exist_ok=True)
            with OpenEXR.File({}, {'RGB':np.ascontiguousarray(image,dtype=np.float32)}) as f:f.write(str(path))
            save_pair(image,dict(asset_id=aid,scene_id='ph_capture_'+a['asset_id'],sequence_id=aid,frame_index=0,
                source_id='polyhaven_native1080_v3',source_path=a['path'],source_sha256=a['sha256'],linear_exr_path=str(path),linear_exr_sha256=sha(path),
                source_gamut='linear_rec709_interpretation',source_encoding='relative_scene_linear',category=a['category'],source_capture=a['asset_id'],
                radiometric_scale=scale,absolute_radiometry_verified=False,camera_projection={'yaw':yy,'pitch':pp,'hfov':90},project='native1080_v3'))
        dump(OUT/'pair_progress.json',dict(stage='panoramas',completed=i+1,total=len(p['panorama_sources'])))
        print('[panorama]',i+1,'/',len(p['panorama_sources']),a['asset_id'],flush=True)

def convert(p):
    for j in p['cg_jobs']:
        frame=j['sample_frame'];aid=f'{j["id"]}_f{frame:04d}';rp=OUT/'render_records'/(aid+'.json')
        rec=json.loads(rp.read_text())
        if rec['source_job']!=j or rec['width']!=W or rec['height']!=H:raise ValueError('Stale render')
        path=Path(rec['path']);ev=p['scene_exposure_ev'].get(j['category'],0.)
        save_pair(old.read_exr(path)*(2**ev),dict(asset_id=aid,scene_id=j['parent_scene_id'],sequence_id=j['id'],frame_index=frame,
            source_id='fxtd_cc0_native1080_v3',source_path=str(path),source_sha256=sha(path),source_encoding='scene_linear',source_gamut='rec709',
            category=j['category'],source_dependencies=j,render_record=str(rp),project='native1080_v3',radiometric_scale=2**ev,absolute_radiometry_verified=False))
        print('[convert]',aid,flush=True)

def verify(p):
    check_base(BASE)
    if sha(Path(p['base_manifest']))!=p['base_manifest_sha256'] or sha(BASE/'ready.json')!=p['base_ready_sha256']:raise ValueError('Base candidate changed')
    for path,digest in p['source_dependencies'].items():
        if sha(path)!=digest:raise ValueError('Changed source '+path)
    base=rows(p['base_manifest']);new=[json.loads(f.read_text()) for f in sorted((OUT/'records').glob('*.json'))]
    expected={f'v3_ph_{a["asset_id"]}_v{v}_{variant}' for a in p['panorama_sources'] for v in range(4) for variant in ('clean','hard')}
    expected|={f'{j["id"]}_f{j["sample_frame"]:04d}_{v}' for j in p['cg_jobs'] for v in ('clean','hard')}
    if len(new)!=p['expected_new_pairs'] or {r['asset_id'] for r in new}!=expected:raise ValueError('Incomplete/unexpected prepared pairs')
    paired=defaultdict(list);checked_sources=set()
    for r in new:
        if (r['width'],r['height'],r['split'])!=(W,H,'train'):raise ValueError('Geometry or split mismatch')
        for kind in ('sdr','hdr'):
            path=Path(r[kind+'_path']);im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            if im is None or im.shape!=(H,W,3) or im.dtype!=np.uint16 or sha(path)!=r[kind+'_sha256']:raise ValueError('Pair decode/hash failure')
        target=old.decode_hdr_u16(im,old.render.STORAGE)
        if not np.isfinite(target).all():raise ValueError('Nonfinite decoded target')
        source=(r['source_path'],r['source_sha256'])
        if source not in checked_sources:
            if sha(source[0])!=source[1]:raise ValueError('Changed pair source')
            checked_sources.add(source)
        paired[r['asset_id'].rsplit('_',1)[0]].append(r)
    for variants in paired.values():
        if len(variants)!=2 or {r['variant'] for r in variants}!={'clean','hard'} or len({r['hdr_sha256'] for r in variants})!=1 or len({r['sdr_sha256'] for r in variants})!=2:raise ValueError('Invalid SDR variants')
    held=[r for r in base if r['split']!='train'];flags=[]
    for kind in ('sdr','hdr'):
        hf=[feature(r[kind+'_path']) for r in held]
        for a in new:
            if kind=='hdr' and a['variant']=='hard':continue
            ha,fa,sa=feature(a[kind+'_path'])
            for b,(hb,fb,sb) in zip(held,hf):
                if min(sa,sb)>.01 and (ha^hb).bit_count()<=6 and float(np.dot(fa,fb))>=.98:flags.append(dict(new=a['asset_id'],held_out=b['asset_id'],kind=kind))
        print('[similarity]',kind,flush=True)
    dump(OUT/'similarity_review.json',dict(cross_split_flags=flags,historical_excluded_captures=6,
        limits='Whole-frame pHash/correlation plus historical capture/family/location exclusions. Crops and cross-provider semantic duplicates are not exhaustively excluded. SI-HDR/ToS are not used as scored evaluation sets by this run.'))
    if flags:raise ValueError('Cross-split similarity requires review')
    combined=base+new
    old.write_rows(OUT/'sdr_hdr_manifest.jsonl',new);old.write_rows(OUT/'combined_candidate_manifest.jsonl',combined)
    # One overview covers every new clean and hard pair; detailed diagnostics recorded separately.
    contact([r for r in new if r['variant']=='clean'],OUT/'contact_sheet.jpg',columns=4)
    contact([r for r in new if r['variant']=='hard'],OUT/'degraded_contact_sheet.jpg',columns=4)
    for k in range(0,len(new),24):contact(new[k:k+24],OUT/f'preview_{k//24+1:02d}.jpg',columns=4)
    previous=json.loads((BASE/'commercial_review.json').read_text())
    approvals={sid:dict(v,evidence_path=str((BASE/v['evidence_path']).resolve())) for sid,v in previous['sources'].items()}
    deps=dict(previous['preparation_dependencies']);deps.update(p['source_dependencies'])
    paths=[OUT/'plan.json',BASE/'ready.json',BASE/'commercial_review.json',SRC/'collection_summary.json',SRC/'ATTRIBUTION_AND_SOURCE_USE.md',
        Path(__file__),REPO/'pipeline/render_next_hdr_scenes.py',REPO/'pipeline/prepare_highlight_expansion.py',REPO/'pipeline/render_highlight_scenes.py',
        REPO/'pipeline/render_pipeline_pairs.py',REPO/'pipeline/verify_highlight_expansion.py']
    paths+=list((SRC/'polyhaven/licence_evidence').glob('*'))+list((SRC/'polyhaven_novel/licence_evidence').glob('*'))
    for path in paths:
        if path.is_file():deps[str(path)]=sha(path)
    evidence=dict(scope='internal commercial-development pilot',owner_request='prepare it and show me the next training script',
        sources={'Poly Haven':'CC0 and explicit AI-use permission; saved provider evidence','CG':'Previously reviewed CC0 JangaFX/ambientCG/Poly Haven assets and FXTD-authored geometry'},
        changes='Native 1080p projection/render; same reviewed ACES SDR/HDR display and storage mapping; SDR-only hard variants.',
        colour_interpretation=p['colour_interpretation'],dependencies=deps,release_authorized=False)
    dump(OUT/'source_evidence.json',evidence)
    for sid in {r['source_id'] for r in new}:approvals[sid]=dict(commercial_training=True,weight_distribution=False,evidence_path='source_evidence.json',evidence_sha256=sha(OUT/'source_evidence.json'))
    manifest=OUT/'combined_candidate_manifest.jsonl'
    review=dict(schema_version=2,scope='internal-pilot',release_authorized=False,manifest_sha256=sha(manifest),reviewed_at='2026-10-19',
        reviewed_by='Source, split and pixel audit under owner preparation instruction',sources=approvals,
        preparation_dependencies=deps,decoder_dependencies={str(f):sha(f) for f in commercial.decoder_dependencies(combined)})
    dump(OUT/'commercial_review.json',review)
    dump(OUT/'preflight_result.json',commercial.validate(manifest,OUT/'commercial_review.json',progress=lambda x:print(x,flush=True)))
    from training.sdr2hdr_dataset import SDRHDRDataset
    import torch
    torch.set_num_threads(1)
    ds=SDRHDRDataset(OUT/'sdr_hdr_manifest.jsonl',split='train',crop_size=256,augment=False,degradation_probability=0)
    checks=[]
    for i,r in enumerate(ds.records):
        sample=ds[i]
        if any(not torch.isfinite(sample[k]).all() or tuple(sample[k].shape)!=(3,256,256) for k in ('sdr','hdr')):raise ValueError('Actual loader failed')
        checks.append(dict(asset_id=r['asset_id'],hdr_max=float(sample['hdr'].max())))
    dump(OUT/'loader_smoke.json',dict(count=len(checks),checks=checks,training_performed=False))
    dump(OUT/'preparation_summary.json',dict(stage='numeric_checks_passed_visual_review_pending',base_pairs=len(base),new_pairs=len(new),total_pairs=len(combined),
        by_split=dict(Counter(r['split'] for r in combined)),new_resolution=[W,H],new_by_category=dict(Counter(r['category'] for r in new)),
        training_started=False,release_authorized=False,limits=p['limits']))
    print('[verified]',len(combined),'pairs',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','panoramas','convert','verify']);args=parser.parse_args()
    cv2.setNumThreads(1);p=plan()
    if args.action!='plan':globals()[args.action](p)
    else:print(json.dumps({'new_pairs':p['expected_new_pairs'],'total_pairs':p['expected_total_pairs'],'resolution':[W,H]}))
