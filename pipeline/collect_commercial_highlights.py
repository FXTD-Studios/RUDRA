"""Collect a bounded CC0 Poly Haven highlight/material source pack on G:.

No trainer or active manifest is changed. API metadata, published licence/AI
permission, publisher MD5, local SHA-256 and decoded-image QC accompany assets.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlparse, unquote

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import cv2
import OpenEXR
import Imath
import numpy as np
import requests

OUT = Path('G:/datasets/sources/polyhaven_highlights_commercial_20261018_v1')
EXISTING = Path('G:/datasets/sources/polyhaven')
UA = 'FXTD-RUDRA-CommercialDatasetPrep/1.0'
MAX_BYTES = 8 * 1024**3
METALS = ['metal_plate','metal_plate_02','blue_metal_plate','rusty_metal',
          'rusty_metal_02','rusty_metal_03','rusty_metal_04','rusty_metal_05',
          'rusty_painted_metal','corrugated_iron','worn_corrugated_iron','green_metal_rust']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp,path)


def get(url):
    for attempt in range(4):
        try:
            r=requests.get(url,headers={'User-Agent':UA},timeout=(20,90))
            if r.status_code == 429:
                time.sleep(min(float(r.headers.get('Retry-After','5')),30))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 3: raise
            time.sleep(2**attempt)
    raise RuntimeError('API rate limit persists')


def file_hash(path, algorithm='sha256'):
    h=hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def licences():
    sources = {
        'asset_licence.html':'https://polyhaven.com/license',
        'ai_permission_faq.html':'https://docs.polyhaven.com/en/faq',
        'api_terms.md':'https://raw.githubusercontent.com/Poly-Haven/Public-API/master/ToS.md',
        'api_usage.html':'https://polyhaven.com/our-api',
    }
    receipts=[]
    for name,url in sources.items():
        p=OUT/'licence_evidence'/name
        if not p.exists():
            r=get(url);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
        receipts.append(dict(url=url,path=str(p),sha256=file_hash(p),retrieved_unix=p.stat().st_mtime))
    licence=(OUT/'licence_evidence/asset_licence.html').read_text(errors='replace')
    faq=(OUT/'licence_evidence/ai_permission_faq.html').read_text(errors='replace')
    terms=(OUT/'licence_evidence/api_terms.md').read_text(errors='replace')
    if 'CC0' not in licence or 'including commercial' not in licence:
        raise ValueError('Commercial asset licence evidence requires review')
    if 'including training AI models' not in faq:
        raise ValueError('Explicit AI permission evidence requires review')
    if 'including commercial use' not in terms:
        raise ValueError('Commercial API permission requires review')
    dump(OUT/'licence_review.json',dict(provider='Poly Haven',licence='CC0-1.0',
        commercial_asset_use_permitted=True,ai_training_explicitly_permitted=True,
        basis='Published provider licence and explicit AI FAQ, retrieved directly.',
        attribution='Assets from Poly Haven (https://polyhaven.com), CC0. API collector identifies itself with a unique User-Agent.',
        evidence=receipts,model_release_authorized=False,
        limits='Asset permission does not certify the complete model, all other training sources, or product release.'))


def select(catalog):
    chosen=[];used=set();sites=Counter()
    def site(aid,v):
        coords=v.get('coords')
        if coords and len(coords)==2:
            return 'geo_'+'_'.join(str(round(float(x),2)) for x in coords)
        return re.sub(r'_\d+$','',aid.removesuffix('_puresky'))
    rules = {
        'clear_sun':lambda v:'clear' in v.get('categories',[]) and 'high contrast' in v.get('categories',[]) and 'sunrise-sunset' not in v.get('categories',[]),
        'cloud_detail':lambda v:'partly cloudy' in v.get('categories',[]) and 'sunrise-sunset' not in v.get('categories',[]),
        'sunrise_sunset':lambda v:'sunrise-sunset' in v.get('categories',[]),
        'overcast':lambda v:'overcast' in v.get('categories',[]) and 'night' not in v.get('categories',[]),
    }
    for category,rule in rules.items():
        candidates=[(k,v) for k,v in catalog.items() if v['type']==0 and 'skies' in v.get('categories',[]) and rule(v)]
        candidates.sort(key=lambda kv:(-(not (EXISTING/(kv[0]+'_2k.exr')).exists()),
                       -float(kv[1].get('evs_cap') or 0),kv[0]))
        n=0
        for aid,v in candidates:
            group=site(aid,v)
            if aid in used or sites[group]>=2:continue
            chosen.append(dict(asset_id=aid,kind='hdri',category=category,location_group=group))
            used.add(aid);sites[group]+=1;n+=1
            if n==8:break
        if n!=8:raise ValueError('Insufficient assets for '+category)
    for aid in ('fireplace','veld_fire'):
        if aid not in catalog or catalog[aid]['type']!=0:raise ValueError('Missing fire reference')
        category = 'fireplace_environment' if aid == 'fireplace' else 'burned_landscape_smoke_reference'
        chosen.append(dict(asset_id=aid,kind='hdri',category=category,location_group=site(aid,catalog[aid])))
    for aid in METALS:
        if catalog[aid]['type']!=1:raise ValueError('Invalid metal asset')
        chosen.append(dict(asset_id=aid,kind='material',category='metal_material',location_group=aid))
    return chosen


def plan():
    path=OUT/'metadata/catalog.json'
    if not path.exists():dump(path,get('https://api.polyhaven.com/assets').json())
    catalog=json.loads(path.read_text())
    selected=select(catalog)
    files=[]
    for i,a in enumerate(selected,1):
        aid=a['asset_id'];meta=catalog[aid]
        p=OUT/'metadata'/(aid+'_files.json')
        if not p.exists():
            dump(p,get('https://api.polyhaven.com/files/'+aid).json())
            time.sleep(.15)
        available=json.loads(p.read_text())
        a.update(name=meta['name'],page_url='https://polyhaven.com/a/'+aid,
            authors=meta.get('authors',{}),licence='CC0-1.0',training_enabled=False,
            existing_2k_path=str(EXISTING/(aid+'_2k.exr')) if (EXISTING/(aid+'_2k.exr')).exists() else None,
            source_type='photographic_equirectangular_HDRI' if a['kind']=='hdri' else 'PBR_material_maps',
            split='unassigned',same_capture_resolution_group=aid.removesuffix('_puresky'),
            colour_space='linear Rec.709/sRGB primaries for HDRI; verify before pair rendering' if a['kind']=='hdri' else 'albedo sRGB, other selected maps non-colour',
            reported_exposure_bracket_stops=meta.get('evs_cap'))
        if a['kind']=='hdri':
            choices=[('hdri','8k','exr')]
        else:
            choices=[(role,'2k','png') for role in ('Diffuse','Rough','Metal','nor_gl','Displacement') if role in available]
            if not {'Diffuse','Rough','nor_gl'} <= {x[0] for x in choices}:
                raise ValueError('Incomplete material '+aid)
        a['selected_maps']=[x[0] for x in choices]
        for role,res,fmt in choices:
            data=available[role][res][fmt]
            host=urlparse(data['url']).hostname
            if host not in ('dl.polyhaven.org','dl.polyhaven.com'):
                raise ValueError('Unexpected download host '+str(host))
            name=Path(unquote(urlparse(data['url']).path)).name
            target=OUT/('hdri' if a['kind']=='hdri' else 'materials')/aid/name
            files.append(dict(asset_id=aid,kind=a['kind'],category=a['category'],role=role,
                resolution=res,format=fmt,url=data['url'],expected_bytes=data['size'],
                expected_md5=data['md5'],path=str(target),licence='CC0-1.0'))
        print(f'[plan] {i}/{len(selected)} {aid}',flush=True)
    total=sum(f['expected_bytes'] for f in files)
    if total>MAX_BYTES:raise ValueError('Collection exceeds 8 GiB bound')
    dump(OUT/'selection.json',selected)
    dump(OUT/'download_plan.json',dict(files=files,expected_bytes=total,maximum_bytes=MAX_BYTES))
    return selected,files


def download_ranges(f):
    """Recover a stalled full transfer using validated 8 MiB HTTP ranges."""
    p=Path(f['path']);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.range.part')
    md5=hashlib.md5();sha=hashlib.sha256()
    with tmp.open('wb') as out:
        for start in range(0,f['expected_bytes'],8*1024*1024):
            end=min(start+8*1024*1024,f['expected_bytes'])-1
            for attempt in range(3):
                try:
                    r=requests.get(f['url'],headers={'User-Agent':UA,'Range':f'bytes={start}-{end}'},timeout=(20,45))
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f"bytes {start}-{end}/{f['expected_bytes']}" or len(r.content)!=end-start+1:
                        raise ValueError('Invalid ranged response')
                    break
                except (requests.RequestException,ValueError):
                    if attempt==2:raise
                    time.sleep(2**attempt)
            md5.update(r.content);sha.update(r.content);out.write(r.content)
    if tmp.stat().st_size!=f['expected_bytes'] or md5.hexdigest()!=f['expected_md5']:
        raise ValueError('Ranged download checksum mismatch')
    os.replace(tmp,p)
    p.with_suffix(p.suffix+'.part').unlink(missing_ok=True)
    return dict(f,status='downloaded_in_ranges',sha256=sha.hexdigest())


def download(f):
    p=Path(f['path']);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.stat().st_size==f['expected_bytes'] and file_hash(p,'md5')==f['expected_md5']:
        return dict(f,status='already_present',sha256=file_hash(p))
    for attempt in range(3):
        tmp=p.with_suffix(p.suffix+'.part')
        try:
            with requests.get(f['url'],headers={'User-Agent':UA},stream=True,timeout=(20,90)) as r:
                r.raise_for_status();n=0;md5=hashlib.md5();sha=hashlib.sha256()
                with tmp.open('wb') as out:
                    for chunk in r.iter_content(1024*1024):
                        if not chunk:continue
                        n+=len(chunk)
                        if n>f['expected_bytes']:raise ValueError('Unexpected download size')
                        md5.update(chunk);sha.update(chunk);out.write(chunk)
            if n!=f['expected_bytes'] or md5.hexdigest()!=f['expected_md5']:
                raise ValueError('Publisher checksum/size mismatch')
            os.replace(tmp,p)
            return dict(f,status='downloaded',sha256=sha.hexdigest())
        except Exception:
            if attempt==2:return download_ranges(f)
            time.sleep(2**attempt)


def verify(f):
    p=Path(f['path'])
    exr_info={}
    if f['kind']=='hdri':
        exr=OpenEXR.InputFile(str(p))
        try:
            header=exr.header();dw=header['dataWindow']
            w=dw.max.x-dw.min.x+1;h=dw.max.y-dw.min.y+1
            pt=Imath.PixelType(Imath.PixelType.FLOAT)
            # BGR ordering matches OpenCV's PNG decoder and preview writer.
            img=np.stack([np.frombuffer(exr.channel(c,pt),dtype=np.float32).reshape(h,w)
                          for c in ('B','G','R')],axis=-1)
            exr_info=dict(decoder='OpenEXR',source_channels={k:str(v) for k,v in header['channels'].items()},
                source_chromaticities=str(header['chromaticities']) if 'chromaticities' in header else None)
        finally:
            exr.close()
    else:
        img=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
    if img is None or not np.isfinite(img).all():raise ValueError('Decode/nonfinite failure '+str(p))
    result=dict(f,shape=list(img.shape),dtype=str(img.dtype),finite=True,
                minimum=float(img.min()),maximum=float(img.max()),**exr_info)
    if f['kind']=='hdri':
        if img.ndim!=3 or img.shape[2]<3 or img.dtype!=np.float32 or img.shape[1] != 2*img.shape[0]:
            raise ValueError('Invalid floating point HDR panorama')
        if img.shape[1]<8000:raise ValueError('Unexpected HDR resolution')
        small=cv2.resize(img[...,:3],(512,256),interpolation=cv2.INTER_AREA)
        positive=np.maximum(small,0)
        display=np.power(positive/(1+positive),1/2.2)
        preview=OUT/'qc_previews'/(f['asset_id']+'.jpg');preview.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(preview),np.rint(display*255).astype(np.uint8))
        sample=img[::8,::8,:3]
        result.update(percentiles=np.percentile(sample,[1,50,99,99.9]).tolist(),
            negative_channel_fraction=float(np.mean(sample<0)),has_values_above_one=bool(img.max()>1),
            absolute_nits_calibrated=False,
            qc_scope='Finite HDR panorama and numeric headroom; not proof of unclipped sun/fire or temporal quality.')
    return result


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'collector.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        licences()
        selected,files=plan()
        results=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(download,f) for f in files]
            for i,job in enumerate(as_completed(jobs),1):
                f=job.result()
                results.append(f)
                dump(OUT/'status.json',dict(stage='download',completed=i,total=len(files),updated_unix=time.time()))
                print(f"[download] {i}/{len(files)} {Path(f['path']).name}",flush=True)
        order={f['path']:i for i,f in enumerate(files)}
        results.sort(key=lambda f:order[f['path']])
        verified=[]
        # One full-res HDR decode at a time to bound RAM/CPU contention.
        with (OUT/'files_manifest.jsonl').open('w',encoding='utf-8') as m:
            for i,f in enumerate(results,1):
                r=verify(f);verified.append(r);m.write(json.dumps(r)+'\n');m.flush()
                dump(OUT/'status.json',dict(stage='verify',completed=i,total=len(files),updated_unix=time.time()))
                print(f"[verify] {i}/{len(files)} {Path(f['path']).name}",flush=True)
        by_asset=Counter(f['asset_id'] for f in verified)
        if set(by_asset)!={a['asset_id'] for a in selected}:raise ValueError('Incomplete asset inventory')
        summary=dict(stage='collection_verified',assets=len(selected),files=len(verified),
            hdri_assets=sum(a['kind']=='hdri' for a in selected),metal_materials=sum(a['kind']=='material' for a in selected),
            categories=dict(Counter(a['category'] for a in selected)),
            existing_2k_capture_overlaps=sum(a['existing_2k_path'] is not None for a in selected),
            bytes=sum(f['expected_bytes'] for f in verified),publisher_checksums_verified=True,
            all_files_decoded_finite=True,manifest_sha256=file_hash(OUT/'files_manifest.jsonl'),
            permitted_for_commercial_asset_use=True,ai_training_permission='Explicit Poly Haven FAQ; CC0 assets',
            training_pairs_created=0,active_training_modified=False,model_release_authorized=False,
            missing_categories=['Camera RAW fire video','Camera RAW spark video','Real moving reflective-metal video'],
            next_stage='Freeze capture/location splits, check existing-corpus overlap, render compatible HDR/SDR pairs; material maps require synthetic scene rendering.')
        dump(OUT/'summary.json',summary);dump(OUT/'status.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as exc:
        dump(OUT/'status.json',dict(stage='failed',error=str(exc),updated_unix=time.time(),
            note='Verified downloads are retained; rerun collector to reuse matching files.'))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
