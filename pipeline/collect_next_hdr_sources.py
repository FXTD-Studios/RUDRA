"""Isolated next-run acquisition. Never edits a corpus or starts training.

Downloads only reviewed public data URLs; archives are extracted as data.
Raw scene values still need colour/scale/split review before training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urljoin
import zipfile

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import requests
from pipeline import collect_commercial_highlights as ph

ROOT = Path('G:/datasets/sources/next_hdr_sources_20261019_v3')
AUDIT = REPO/'reports/commercial_gap_collection_20261019'
if not AUDIT.exists():
    AUDIT = ROOT/'acquisition_audit'
UA = 'FXTD-RUDRA-NextHDRCollection/3.0'

def dump(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2),encoding='utf-8');t.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def get(url):
    r=requests.get(url,headers={'User-Agent':UA},timeout=(20,60));r.raise_for_status();return r

def download(url,path,max_bytes=3*1024**3):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    receipt=path.with_suffix(path.suffix+'.receipt.json')
    if path.exists() and receipt.exists():
        old=json.loads(receipt.read_text())
        if sha(path)==old['sha256']:return old
    tmp=path.with_suffix(path.suffix+'.part')
    for attempt in range(4):
        try:
            off=tmp.stat().st_size if tmp.exists() else 0
            headers={'User-Agent':UA}
            if off:headers['Range']=f'bytes={off}-'
            with requests.get(url,headers=headers,stream=True,timeout=(20,90)) as r:
                r.raise_for_status()
                if off and r.status_code!=206:off=0
                if off and not r.headers.get('Content-Range','').startswith(f'bytes {off}-'):raise ValueError('Bad range')
                length=int(r.headers.get('Content-Length',0));n=off;last=n
                if n+length>max_bytes:raise ValueError('Size bound exceeded')
                with tmp.open('ab' if off else 'wb') as f:
                    for b in r.iter_content(1024*1024):
                        if not b:continue
                        f.write(b);n+=len(b)
                        if n>max_bytes:raise ValueError('Size bound exceeded')
                        if n-last>=128*1024**2:print('[download]',path.name,round(n/1e9,2),'GB',flush=True);last=n
                if length and n!=off+length:raise ValueError('Incomplete response')
                info=dict(url=url,resolved_url=r.url,bytes=n,sha256=sha(tmp),etag=r.headers.get('ETag'),retrieved_unix=time.time())
            tmp.replace(path);dump(receipt,info);return info
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(2**attempt)

def unpack(path,dest,allowed):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);files=[]
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist())>12*1024**3:raise ValueError('Archive expansion bound exceeded')
        for i in z.infolist():
            target=(dest/i.filename.replace('\\','/')).resolve()
            if not target.is_relative_to(dest.resolve()):raise ValueError('Archive path escape')
            if (i.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive symlink')
            if i.is_dir() or target.suffix.lower() not in allowed:continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(i) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
            files.append(dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target)))
    return files

def evidence(out,names):
    for name in names:
        p=out/'licence_evidence'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(AUDIT/name,p)

def sihdr():
    out=ROOT/'sihdr';out.mkdir(parents=True,exist_ok=True)
    evidence(out,['cambridge.html','SIHDR_README.md','cambridge_links.json'])
    text=(out/'licence_evidence/cambridge.html').read_text(encoding='utf-8')
    if 'https://creativecommons.org/licenses/by/4.0/' not in text:raise ValueError('Licence changed')
    links=json.loads((AUDIT/'cambridge_links.json').read_text());records=[]
    # Original reference and simulated inputs only. Other models' reconstructions
    # are not ground truth; RAW stacks duplicate source scenes and can be acquired later.
    for name in ['reference.zip','input.zip']:
        u='https://www.repository.cam.ac.uk'+next(a['url'] for a in links if name in a['text'])
        archive=out/'downloads'/name;info=download(u,archive)
        files=unpack(archive,out/'unpacked',{'.exr','.png','.md','.txt'})
        records.append(dict(name=name,archive=info,files=files));dump(out/'archives.json',records)
        print('[unpacked]',name,len(files),flush=True)
    dump(out/'status.json',dict(stage='downloaded_pending_pixel_review',archives=records,licence='CC-BY-4.0',training_enabled=False))

def environments(novel=False):
    out=ROOT/('polyhaven_novel' if novel else 'polyhaven');ph.OUT=out;ph.UA=UA;out.mkdir(parents=True,exist_ok=True);ph.licences()
    cat=ph.get('https://api.polyhaven.com/assets?t=hdris').json();dump(out/'metadata/catalog.json',cat)
    groups={
        'night_practicals':['blaubeuren_night','blue_lagoon_night','cobblestone_street_night','modern_buildings_night','neon_photostudio','night_bridge','warm_restaurant_night','shanghai_riverside'],
        'bright_windows':['abandoned_bakery','afrikaans_church_interior','aft_lounge','cayley_interior'],
        'water_reflection':['river_rocks','radkow_lake','pond','lakeside_sunrise']}
    if novel:
        groups={'night_practicals':['donkin_reserve','ladybrand_heritage_house'],
            'interiors_and_mixed_light':['cowboy_town_saloon','debris_basement_corridor','relax_inn_seaview_suite','witsand_woolshop'],
            'glass_and_outdoor_transition':['ushaka_sea_world_aquarium','sundowner_deck']}
    selected=[];files=[]
    old=json.loads(Path('G:/datasets/sources/polyhaven_highlights_commercial_20261018_v1/selection.json').read_text())
    old_ids={a['asset_id'] for a in old}
    for category,ids in groups.items():
        for aid in ids:
            if aid in old_ids:continue
            meta=cat[aid];f=ph.get('https://api.polyhaven.com/files/'+aid).json();dump(out/'metadata'/(aid+'_files.json'),f)
            d=f['hdri']['8k']['exr'];p=out/'hdri'/aid/Path(d['url']).name
            selected.append(dict(asset_id=aid,category=category,metadata=meta,capture_group=aid,split='unassigned',licence='CC0-1.0'))
            files.append(dict(asset_id=aid,category=category,kind='hdri',url=d['url'],path=str(p),expected_bytes=d['size'],expected_md5=d['md5']))
            time.sleep(.2)
    if sum(f['expected_bytes'] for f in files)>8*1024**3:raise ValueError('Pack exceeds 8GiB')
    dump(out/'selection.json',selected);dump(out/'download_plan.json',files)
    verified=[]
    for i,f in enumerate(files,1):
        item=ph.verify(ph.download(f));verified.append(item)
        dump(out/'status.json',dict(stage='collecting',completed=i,total=len(files)))
        dump(out/'verified_files.json',verified);print('[environment]',i,len(files),f['asset_id'],flush=True)
    dump(out/'status.json',dict(stage='downloaded_decoded_pending_split_and_visual_review',count=len(verified),bytes=sum(f['expected_bytes'] for f in files),training_enabled=False))

def novel_environments():
    environments(novel=True)

def blender():
    out=ROOT/'blender';out.mkdir(parents=True,exist_ok=True);evidence(out,['blender.html','blender_links.json'])
    links=json.loads((AUDIT/'blender_links.json').read_text());records=[]
    for key,name in [('classroom.zip','classroom.zip'),('basic_particle_simulation.blend','basic_particle_simulation.blend')]:
        page=next(a['url'] for a in links if a['url'].endswith(key))
        html=get(page).text
        (out/'licence_evidence'/(name+'.download.html')).write_text(html,encoding='utf-8')
        direct=re.findall(r'href="(https://mirror\.blender\.org/[^"?]+)"',html)
        url=next(u for u in direct if u.endswith(key));p=out/'downloads'/name
        # Preserve an earlier landing-page response as evidence, not as an asset.
        if p.exists() and p.read_bytes()[:100].lstrip().lower().startswith(b'<!doctype html'):
            p.replace(out/'licence_evidence'/(name+'.landing.html'))
            receipt=p.with_suffix(p.suffix+'.receipt.json')
            if receipt.exists():receipt.replace(out/'licence_evidence'/(name+'.landing.receipt.json'))
        info=download(url,p,max_bytes=300*1024**2)
        files=unpack(p,out/'unpacked', {'.blend','.png','.jpg','.jpeg','.exr','.hdr','.txt','.md'}) if p.suffix=='.zip' else [dict(path=str(p),sha256=sha(p))]
        records.append(dict(name=name,receipt=info,files=files,licence='CC0-1.0',training_enabled=False))
    dump(out/'status.json',dict(stage='downloaded_pending_scene_inspection',sources=records))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('group',choices=['sihdr','environments','novel_environments','blender']);a=p.parse_args()
    try:globals()[a.group]()
    except Exception as e:
        dump(ROOT/(a.group+'_failure.json'),dict(error=str(e),time=time.time()));raise
