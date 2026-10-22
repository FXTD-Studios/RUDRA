"""Prepare an isolated, bounded highlight expansion. Never starts training.

Uses source receipts on G:, freezes train-only augmentation groups, renders
perspective HDRI samples, converts authored Cycles EXRs and writes a candidate
manifest. Existing validation/test rows and active code/data remain unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import cv2
import numpy as np
import OpenEXR
from pipeline import render_pipeline_pairs as render
from pipeline.hdr_io import encode_hdr_u16, decode_hdr_u16

PH=Path('G:/datasets/sources/polyhaven_highlights_commercial_20261018_v1')
BJ=Path('G:/datasets/sources/blender_jangafx_gap_sources_20261018_v1')
BASE=Path('G:/datasets/corpora/rudra_z_pipeline_20261018_v1')
OUT=Path('G:/datasets/corpora/rudra_highlight_expansion_20261018_v1')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(tmp,p)


def rows(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]


def write_rows(p,data):
    p=Path(p);p.write_text(''.join(json.dumps(r)+'\n' for r in data),encoding='utf-8')


def freeze_plan():
    if (OUT/'plan.json').exists():return json.loads((OUT/'plan.json').read_text())
    selected=json.loads((PH/'selection.json').read_text()); files=rows(PH/'files_manifest.jsonl')
    blocked_locations={a['location_group'] for a in selected if set(a.get('previous_auxiliary_splits',[]))&{'val','test'}}
    family=lambda name:re.sub(r'_\d+$','',name.removesuffix('_puresky'))
    blocked_families={family(a['asset_id']) for a in selected if set(a.get('previous_auxiliary_splits',[]))&{'val','test'}}
    allowed=[];excluded=[]
    for a in selected:
        if a['kind']!='hdri':continue
        reason=None
        if a['location_group'] in blocked_locations:reason='capture/location overlaps previous held-out or mixed splits'
        if family(a['asset_id']) in blocked_families:reason='capture family overlaps historical held-out/mixed splits'
        if len(a.get('previous_auxiliary_splits',[]))>1:reason='mixed historical capture splits'
        if reason:excluded.append(dict(asset_id=a['asset_id'],reason=reason));continue
        f=next(f for f in files if f['asset_id']==a['asset_id'])
        allowed.append(dict(asset_id=a['asset_id'],category=a['category'],capture_group=a['same_capture_resolution_group'],
            location_group=a['location_group'],path=f['path'],sha256=f['sha256'],split='train'))
    material_maps={f['role']:f['path'] for f in files if f['asset_id']=='metal_plate'}
    light=next(a for a in allowed if a['asset_id']=='valley_of_desolation')
    jobs=[]
    for i,cat in enumerate(['metal_polished','metal_textured','sparks_dark','sparks_bright','neon','bright_window','campfire','simple_fire']):
        j=dict(id='hl_'+cat,category=cat,seed=20261018+i,split='train',frames=4)
        if cat.startswith('metal'):
            j.update(hdri_path=light['path'],hdri_capture=light['capture_group'])
            if cat=='metal_textured':j['material_maps']=material_maps
        if cat in ('campfire','simple_fire'):
            import re
            v=sorted((BJ/'unpacked'/cat).rglob('*.vdb'),key=lambda p:int(re.search(r'(\d+)\.vdb$',p.name).group(1)))
            start=80 if cat=='campfire' else 40
            j['vdb_frames']=[str(p) for p in v[start:start+4]]
            j['vdb_sha256']=[sha(p) for p in v[start:start+4]]
        jobs.append(j)
    plan=dict(schema_version=1,base_manifest=str(BASE/'combined_manifest.jsonl'),base_manifest_sha256=sha(BASE/'combined_manifest.jsonl'),
        panorama_sources=allowed,excluded_sources=excluded,cg_jobs=jobs,width=640,height=360,views_per_panorama=2,frames_per_view=3,
        variants=['clean','hard'],seed=20261018,all_new_splits='train',expected_new_pairs=len(allowed)*12+64,
        split_policy='Original owned val/test unchanged. New frames and variants train-only. Block prior held-out capture/location groups.',
        colour_interpretation='HDRI relative scene-linear Rec.709 interpretation; absolute camera radiometry and individual source primaries not certified. Cycles authored EXRs explicitly linear Rec.709. ACES 1.3 output at 1000 nits.',
        limits=['Synthetic training expansion; not independent real HDR evaluation','HDRI pans have no scene motion or parallax','Four CG frames and three pan frames are short snippets, not full temporal benchmark',
                'Charge legacy Eevee projects held out pending compatible render validation'],training_started=False,release_authorized=False)
    dump(OUT/'plan.json',plan)
    sentinel=json.loads((BASE/'_ingest_config.json').read_text())
    sentinel.update(corpus=OUT.name,tool='pipeline/prepare_highlight_expansion.py',plan_sha256=sha(OUT/'plan.json'),licence='Poly Haven CC0, JangaFX CC0 and authored geometry; see source evidence',
                    warning='Synthetic/rendered training augmentation; not real SDR grades or calibrated radiance',source_colour_interpretation=plan['colour_interpretation'])
    dump(OUT/'_ingest_config.json',sentinel)
    print(json.dumps({'planned_pairs':plan['expected_new_pairs'],'hdri_captures':len(allowed),'excluded':len(excluded)}),flush=True)
    return plan


def read_exr(path):
    with OpenEXR.File(str(path)) as f:
        channels=f.channels()
        if 'RGB' in channels: a=channels['RGB'].pixels
        elif 'RGBA' in channels:a=channels['RGBA'].pixels[...,:3]
        else:a=np.stack([channels[k].pixels for k in ('R','G','B')],axis=-1)
        a=np.array(a,dtype=np.float32)
    if a.ndim!=3 or a.shape[-1]!=3 or not np.isfinite(a).all():raise ValueError('Invalid EXR '+str(path))
    return a


def perspective(pano,yaw,pitch):
    w,h=640,360; half=math.tan(math.radians(75)/2)
    xx,yy=np.meshgrid((np.arange(w)+.5-w/2)/(w/2)*half,(np.arange(h)+.5-h/2)/(w/2)*half)
    d=np.stack([xx,-yy,np.ones_like(xx)],-1);d/=np.linalg.norm(d,axis=-1,keepdims=True)
    y,p=math.radians(yaw),math.radians(pitch)
    ry=np.array([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]])
    rx=np.array([[1,0,0],[0,math.cos(p),-math.sin(p)],[0,math.sin(p),math.cos(p)]])
    v=d@(ry@rx).T
    u=(np.arctan2(v[...,0],v[...,2])/(2*np.pi)+.5)*pano.shape[1]+1
    t=(.5-np.arcsin(np.clip(v[...,1],-1,1))/np.pi)*pano.shape[0]
    pad=np.concatenate([pano[:,-1:],pano,pano[:,:1]],axis=1)
    return cv2.remap(pad,u.astype(np.float32),t.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def pair(linear709,identity):
    recipe=dict(family='aces1.3_1000',config=render.ACES1,sdr_display=list(render.FAMILIES[0][2]),hdr_display=list(render.FAMILIES[0][3]),
                hdr_peak_nits=1000.,trim_ev=0.,codec='none',quality=None)
    linear2020=np.maximum(linear709,0)@render.M_709_2020.T
    clean,hdr,code=render.render_pair(linear2020,recipe)
    encoded,stats=encode_hdr_u16(hdr,render.STORAGE)
    lum=hdr@np.array([.2627,.678,.0593],np.float32)*203
    if stats['clipped_fraction']!=0:raise ValueError('Storage clipping')
    results=[]
    for variant in ('clean','hard'):
        sdr=clean
        if variant=='hard':
            # Same HDR target; SDR-only lost detail, 8-bit quantization and JPEG.
            damaged=np.clip((code.astype(np.float32)-5)/(.90*255-5),0,1)
            damaged=np.rint(damaged*255).astype(np.uint8)
            damaged=render.sdr_render._jpeg(damaged,60)
            sdr=render.sdr_canonical(damaged)
        aid=identity['asset_id']+'_'+variant
        sp=OUT/'sdr'/f'{aid}.png';hp=OUT/'hdr'/f'{aid}.png'
        render._write_png(sp,sdr);render._write_png(hp,encoded)
        decoded=decode_hdr_u16(encoded,render.STORAGE)
        err=np.abs(np.log2(np.maximum(decoded*203,.005))-np.log2(np.maximum(hdr*203,.005)))
        if float(err.max())>.002:raise ValueError('HDR round-trip exceeds .002 stops')
        r=dict(identity,asset_id=aid,split='train',sdr_path=str(sp),hdr_path=str(hp),sdr_sha256=sha(sp),hdr_sha256=sha(hp),width=640,height=360,
            sdr_kind='rendered',render_family='pipeline',target_family='pipeline',pipeline_recipe=recipe,source_curve='unknown',
            sdr_curve='aces1.3_1000',sdr_codec='none' if variant=='clean' else 'jpeg',tonemap_ev=None,sdr_encoding='srgb2020_from_bt709',hdr_encoding='log2_extended',
            ceiling_nits=970.,peak_nits=stats['peak_nits'],clipped_fraction=stats['clipped_fraction'],hdr_p50_nits=float(np.percentile(lum,50)),hdr_p99_nits=float(np.percentile(lum,99)),
            highlight_fraction_above_203_nits=float((lum>203).mean()),variant=variant,artificial_degradation=None if variant=='clean' else {'black_code':5,'white_code':229.5,'jpeg_quality':60},
            is_video=False,sequence_available=True,commercial_ok=True,licence_licence='CC0 sources and authored geometry; source receipts retained',excluded=None,
            storage_roundtrip_max_stops=float(err.max()))
        dump(OUT/'records'/f'{aid}.json',r);results.append(r)
    return results


def panoramas(plan):
    for i,a in enumerate(plan['panorama_sources'],1):
        if sha(a['path'])!=a['sha256']:raise ValueError('Changed source '+a['path'])
        pano=read_exr(a['path']);lum=np.maximum(pano,0)@np.array([.2126,.7152,.0722],np.float32)
        small=cv2.resize(lum,(512,256),interpolation=cv2.INTER_AREA)
        py,px=np.unravel_index(np.argmax(small),small.shape)
        yaw=(px/512-.5)*360;pitch=-(.5-py/256)*180
        # Fixed scene normalization, shared across all views/frames. Relative CG exposure only.
        scale=float(np.clip(.18/max(float(np.percentile(lum[::8,::8],50)),1e-4),.01,100))
        for view in range(2):
            for frame in range(3):
                aid=f"hl_ph_{a['asset_id']}_v{view}_f{frame:04d}"
                if (OUT/'records'/f'{aid}_hard.json').exists():continue
                yy=yaw+view*50+(frame-1)*.6;pp=float(np.clip(pitch if view==0 else -22,-65,35))
                image=perspective(pano,yy,pp)*scale
                pair(image,dict(asset_id=aid,scene_id='hl_ph_'+a['capture_group'],sequence_id=f"hl_ph_{a['asset_id']}_v{view}",frame_index=frame,fps=24,
                    source_id='polyhaven_highlight_pipeline',source_path=a['path'],source_sha256=a['sha256'],source_gamut='linear_rec709_interpretation',source_encoding='relative_scene_linear',
                    category=a['category'],source_capture=a['capture_group'],location_group=a['location_group'],radiometric_scale=scale,absolute_radiometry_verified=False,
                    camera_projection={'yaw':yy,'pitch':pp,'hfov':75},project='highlight_expansion'))
        print(f"[panorama] {i}/{len(plan['panorama_sources'])} {a['asset_id']}",flush=True)


def convert(plan):
    for j in plan['cg_jobs']:
        for frame in range(4):
            aid=f"{j['id']}_f{frame:04d}";rp=OUT/'render_records'/f'{aid}.json'
            if not rp.exists():raise ValueError('Missing render '+str(rp))
            record=json.loads(rp.read_text());p=Path(record['path'])
            sid='jangafx_highlight_pipeline' if j['category'] in ('campfire','simple_fire') else 'fxtd_authored_highlight_pipeline'
            pair(read_exr(p),dict(asset_id=aid,scene_id=j['id'],sequence_id=j['id'],frame_index=frame,fps=24,source_id=sid,source_path=str(p),source_sha256=sha(p),
                source_encoding='scene_linear',source_gamut='rec709',category=j['category'],project='highlight_expansion',render_record=str(rp),
                source_dependencies=j,absolute_radiometry_verified=False))
        print('[convert]',j['id'],flush=True)


def finalize(plan):
    if sha(BASE/'combined_manifest.jsonl')!=plan['base_manifest_sha256']:raise ValueError('Base changed')
    records=[json.loads(p.read_text()) for p in sorted((OUT/'records').glob('*.json'))]
    allowed={a['asset_id'] for a in plan['panorama_sources']}
    new=[r for r in records if r['source_id']!='polyhaven_highlight_pipeline' or any(r['asset_id'].startswith('hl_ph_'+a+'_v') for a in allowed)]
    prep_excluded=[r['asset_id'] for r in records if r not in new]
    dump(OUT/'excluded_prepared_records.json',dict(asset_ids=prep_excluded,reason='Conservative capture-family split exclusion; files retained outside candidate manifest'))
    if len(new)!=plan['expected_new_pairs']:raise ValueError(f'Expected {plan["expected_new_pairs"]}, found {len(new)}')
    old=rows(BASE/'combined_manifest.jsonl');combined=old+new
    seen=set(); content=defaultdict(set);groups=defaultdict(set)
    for r in combined:
        if r['asset_id'] in seen:raise ValueError('Duplicate ID')
        seen.add(r['asset_id']);groups[(r['source_id'],r['scene_id'])].add(r['split'])
        for kind in ('sdr','hdr'):content[(kind,r[kind+'_sha256'])].add(r['split'])
    if any(len(v)>1 for v in list(content.values())+list(groups.values())):raise ValueError('Split overlap')
    for r in new:
        a=[]
        for kind in ('sdr','hdr'):
            p=Path(r[kind+'_path'])
            if sha(p)!=r[kind+'_sha256']:raise ValueError('Pair hash mismatch')
            im=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
            if im is None or im.dtype!=np.uint16 or im.shape!=(360,640,3):raise ValueError('Invalid saved pair')
            a.append(im)
        if not np.isfinite(decode_hdr_u16(a[1],render.STORAGE)).all():raise ValueError('Invalid HDR')
    write_rows(OUT/'sdr_hdr_manifest.jsonl',new);write_rows(OUT/'combined_candidate_manifest.jsonl',combined)
    from training.sdr2hdr_dataset import corpus_ev_of
    corpus_ev_of(OUT/'combined_candidate_manifest.jsonl',combined,pipeline_only_ev=0)
    dump(OUT/'preparation_summary.json',dict(stage='pairs_prepared_pending_visual_and_duplicate_review',base_pairs=len(old),new_pairs=len(new),total_pairs=len(combined),
        by_split=dict(Counter(r['split'] for r in combined)),new_by_category=dict(Counter(r['category'] for r in new)),exact_cross_split_flags=0,
        source_groups=len({r['scene_id'] for r in new}),training_started=False,release_authorized=False,
        limits=plan['limits'],next='Review contact sheets and cross-split near duplicates; bind source evidence before any next run.'))
    print(json.dumps({'base':len(old),'new':len(new),'total':len(combined)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','panoramas','convert','finalize']);a=p.parse_args()
    cv2.setNumThreads(1);plan=freeze_plan()
    if a.action=='panoramas':panoramas(plan)
    elif a.action=='convert':convert(plan)
    elif a.action=='finalize':finalize(plan)
