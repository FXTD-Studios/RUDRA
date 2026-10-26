"""Collect the approved free CC0 expansion. Never starts training or executes assets."""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import time
import zipfile
import requests

OUT=Path('G:/datasets/sources/fire_reflection_expansion_20261018_v2')
UA='FXTD-RUDRA-SourceReview/2.0'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2),encoding='utf-8');os.replace(t,p)

class Links(HTMLParser):
    def __init__(self): super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.append(dict(attrs))

def fetch(url):
    r=requests.get(url,headers={'User-Agent':UA},timeout=(20,60));r.raise_for_status();return r

def collect(a):
    receipt=OUT/'receipts'/(a['id']+'.json')
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if Path(saved['path']).exists() and sha(saved['path'])==saved['sha256']:
            print('[reuse]',a['id'],flush=True);return saved
    url=a['download_url']
    if a['provider']=='JangaFX':
        r=fetch(url);(OUT/'licence_evidence'/(a['id']+'_download.html')).write_bytes(r.content)
        links=Links();links.feed(r.text);url=next(x['href'] for x in links.links if x.get('id')=='downloadButton')
    name=Path(str(a['filename']).replace('\\','/')).name   # remote data: a bare basename only
    if not name or name in ('.','..'):raise ValueError('Unsafe download filename')
    p=OUT/'downloads'/name;p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.part')
    for attempt in range(4):
        offset=temp.stat().st_size if temp.exists() else 0
        headers={'User-Agent':UA}
        if offset:headers['Range']=f'bytes={offset}-'
        try:
            with requests.get(url,headers=headers,stream=True,timeout=(20,90)) as r:
                r.raise_for_status()
                if offset and r.status_code!=206:offset=0
                if offset and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Invalid resume range')
                size=int(r.headers.get('Content-Length',0));total=size+offset
                if total>9*1024**3:raise ValueError('Download exceeds bound')
                n=offset;last=offset
                with temp.open('ab' if offset else 'wb') as f:
                    for b in r.iter_content(1024*1024):
                        if not b:continue
                        f.write(b);n+=len(b)
                        if n>9*1024**3:raise ValueError('Download exceeds bound')
                        if n-last>256*1024**2:print('[download]',a['id'],round(n/1e9,2),'GB',flush=True);last=n
                if size and n!=total:raise ValueError('Incomplete download')
                if a.get('expected_bytes') and n!=a['expected_bytes']:raise ValueError('Publisher size mismatch')
            os.replace(temp,p);break
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(2**attempt)
    dest=OUT/'unpacked'/a['id'];dest.mkdir(parents=True,exist_ok=True)
    def safe(name):
        path=(dest/name.replace('\\','/')).resolve()
        if not path.is_relative_to(dest.resolve()):raise ValueError('Unsafe archive member')
    if p.suffix=='.zip':
        with zipfile.ZipFile(p) as z:
            for i in z.infolist():
                safe(i.filename)
                if (i.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive symlink')
            if sum(i.file_size for i in z.infolist())>40*1024**3:raise ValueError('Expanded archive too large')
            for i in z.infolist():
                if not i.is_dir() and Path(i.filename).suffix.lower() in ('.vdb','.txt','.png','.jpg','.jpeg'):
                    z.extract(i,dest) # read verifies CRC; no scripts/presets executed
    else:
        names=subprocess.check_output(['tar','-tf',str(p)],text=True).splitlines()
        for name in names:safe(name)
        # bsdtar has secure path/symlink handling; only data and text members selected.
        selected=[n for n in names if Path(n).suffix.lower() in ('.vdb','.txt')]
        # tar with no member arguments would extract everything; '--' stops option parsing.
        if not selected:raise ValueError('No extractable data members in archive')
        subprocess.run(['tar','-xf',str(p),'-C',str(dest),'--',*selected],check=True)
    media=[]
    for f in sorted(dest.rglob('*')):
        if f.is_file():media.append(dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)))
    licence_files=[f for f in media if 'license' in Path(f['path']).name.lower() or 'licence' in Path(f['path']).name.lower()]
    for f in licence_files:
        txt=Path(f['path']).read_text(errors='replace')
        if a['provider']=='JangaFX' and not ('CC0' in txt or 'Public Domain' in txt):raise ValueError('Review embedded licence')
    result=dict(a,path=str(p),bytes=p.stat().st_size,sha256=sha(p),resolved_download_url=url,unpacked_path=str(dest),
        extracted_files=media,embedded_licences=licence_files,retrieved_unix=time.time(),publisher_checksum=None,
        licence='CC0-1.0',training_enabled=False,release_authorized=False)
    dump(receipt,result);print('[complete]',a['id'],len(media),'files',flush=True);return result

def main():
    ev=OUT/'licence_evidence';ev.mkdir(parents=True,exist_ok=True)
    evidence=[]
    for name,url,needle in [('jangafx','https://jangafx.com/software/embergen/download/free-vdb-animations','CC0 (Public Domain)'),
        ('ambientcg','https://docs.ambientcg.com/license/','CC0 1.0')]:
        r=fetch(url)
        if needle not in r.text:raise ValueError('Expected public grant changed')
        p=ev/(name+'.html');p.write_bytes(r.content);evidence.append(dict(url=url,path=str(p),sha256=sha(p),retrieved_unix=time.time()))
    r=fetch('https://ambientcg.com/api/v2/full_json?type=Material&category=Metal&include=downloadData&limit=6')
    (ev/'ambientcg_selection_api.json').write_bytes(r.content)
    assets=[]
    for a in r.json()['foundAssets']:
        options=a['downloadFolders']['default']['downloadFiletypeCategories']['zip']['downloads']
        d=next(x for x in options if x['attribute']=='2K-PNG')
        assets.append(dict(id=a['assetId'],provider='ambientCG',title=a['displayName'],source_page=a['shortLink'],
            tags=a['tags'],filename=d['fileName'],download_url=d['downloadLink'],expected_bytes=d['size'],kind='PBR_material'))
    jpage=(ev/'jangafx.html').read_text(encoding='utf-8')
    for aid,title in [('ground_explosion','Ground Explosion'),('gasoline_explosion','Gasoline Explosion'),('fire_plume','Fire Plume')]:
        fragment=jpage[jpage.index(title):];parser=Links();parser.feed(fragment)
        url=next(x['href'] for x in parser.links if 'mediafire.com/file/' in x.get('href',''))
        assets.append(dict(id=aid,title=title,provider='JangaFX',source_page=evidence[0]['url'],download_url=url,
            filename=url.split('/')[-2],kind='animated_VDB_volume'))
    dump(OUT/'download_plan.json',assets);dump(ev/'evidence.json',evidence)
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(collect,assets))
    dump(OUT/'sources.json',results)
    dump(OUT/'summary.json',dict(stage='downloaded_extracted_pending_render_validation',assets=len(results),
        archive_bytes=sum(a['bytes'] for a in results),vdb_frames=sum(Path(f['path']).suffix=='.vdb' for a in results for f in a['extracted_files']),
        training_started=False,release_authorized=False))

if __name__=='__main__':main()
