"""Collect selected freely accessible CC-BY Blender and CC0 JangaFX sources.

Only downloads source assets; no rendering, training or package installation.
"""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
import hashlib
import json
import os
import time
import zipfile
import requests

AUDIT=Path('G:/datasets/research/blender_hdr_gap_audit_20261018')
OUT=Path('G:/datasets/sources/blender_jangafx_gap_sources_20261018_v1')
UA='FXTD-RUDRA-SourceReview/1.0'


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.append(dict(attrs))


def download(a):
    p=OUT/'downloads'/a['filename'];p.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            with requests.get(a['download_url'],headers={'User-Agent':UA},stream=True,timeout=(20,60)) as r:
                r.raise_for_status();size=int(r.headers.get('Content-Length',0));n=0;h=hashlib.sha256()
                if size>400*1024**2:raise ValueError('Unexpectedly large source archive')
                temp=p.with_suffix(p.suffix+'.part')
                with temp.open('wb') as f:
                    for b in r.iter_content(1024*1024):
                        if not b:continue
                        n+=len(b)
                        if n>400*1024**2:raise ValueError('Source exceeds size bound')
                        h.update(b);f.write(b)
                if size and n!=size:raise ValueError('Incomplete response')
            os.replace(temp,p)
            break
        except requests.RequestException:
            if attempt==2:raise
            time.sleep(2**attempt)
    result=dict(a,path=str(p),bytes=n,sha256=h.hexdigest(),retrieved_unix=time.time(),
        publisher_checksum=None,training_enabled=False,rendered_pairs=0)
    if p.suffix=='.zip':
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:raise ValueError('ZIP integrity failure')
            entries=z.infolist()
            if sum(i.file_size for i in entries)>2*1024**3:raise ValueError('Unexpected expanded archive size')
            destination=OUT/'unpacked'/a['id'];destination.mkdir(parents=True,exist_ok=True)
            for i in entries:
                target=(destination/i.filename).resolve()
                if not target.is_relative_to(destination.resolve()):raise ValueError('Unsafe archive path')
                if i.filename.endswith('/') or Path(i.filename).suffix.lower() in ('.blend','.vdb','.png','.jpg','.jpeg','.tif','.tiff','.exr','.txt','.cats','.ember'):
                    z.extract(i,destination)
            result.update(zip_crc_passed=True,archive_files=[i.filename for i in entries],
                vdb_frames=sum(i.filename.lower().endswith('.vdb') for i in entries),unpacked_path=str(destination))
    elif p.suffix=='.blend':
        head=p.open('rb').read(12)
        if not (head.startswith(b'BLENDER') or head.startswith(b'\x28\xb5\x2f\xfd') or head.startswith(b'\x1f\x8b')):
            raise ValueError('Unexpected Blender file header')
        result['header_hex']=head.hex()
    elif p.suffix=='.rar':
        if not p.open('rb').read(7).startswith(b'Rar!'):raise ValueError('Unexpected RAR header')
        result['archive_decode_status']='pending external archive verification'
    print(json.dumps({k:result[k] for k in ('id','bytes','sha256')}),flush=True)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assets=[]
    for key,filename,title,asset_id in [
        ('charge_fire','charge_smoke_fire.blend','Charge Smoke & Fire FX - Geometry Nodes','6132'),
        ('charge_muzzle','charge_muzzle_flash.blend','Charge Muzzle Flash FX (includes sparks)','6194'),
        ('charge_materials','charge_material_library.zip','Charge Material Asset Library - Freebie','7249')]:
        page=(AUDIT/(key+'.html')).read_text(encoding='utf-8')
        if 'https://creativecommons.org/licenses/by/4.0/' not in page:raise ValueError('Exact asset licence not confirmed')
        parser=Links();parser.feed(page)
        ext='.zip' if key=='charge_materials' else '.blend'
        link=next(a['href'] for a in parser.links if '/download-source/' in a.get('href','') and a['href'].endswith(ext))
        assets.append(dict(id=key,title=title,filename=filename,licence='CC-BY-4.0',
            attribution='Blender Foundation / Blender Studio; Simon Thommes'+(' and Beau Gerbrands' if key=='charge_materials' else ''),
            source_page='https://studio.blender.org/projects/charge/?asset='+asset_id,
            download_url=urljoin('https://studio.blender.org',link),kind='procedural_CG_source',
            commercial_copyright_permission=True,condition='Retain attribution, source link, licence and modification notices; no endorsement.'))
    if 'CC0 (Public Domain)' not in (AUDIT/'jangafx_licence.html').read_text(encoding='utf-8'):
        raise ValueError('JangaFX exact free-VDB licence missing')
    for key,filename,title in [('campfire','SmallCampfireVDB.zip','JangaFX Small Camp Fire'),('simple_fire','Fire_01.rar','JangaFX Simple Fire')]:
        parser=Links();parser.feed((AUDIT/(key+'_download_page.html')).read_text(encoding='utf-8'))
        link=next(a['href'] for a in parser.links if a.get('id')=='downloadButton')
        assets.append(dict(id=key,title=title,filename=filename,licence='CC0-1.0',attribution='JangaFX',
            source_page='https://jangafx.com/software/embergen/download/free-vdb-animations',
            download_url=link,kind='animated_VDB_volume',commercial_copyright_permission=True,
            condition='CC0 applies to these published free VDB simulations; no EmberGen software/trial licence inferred.'))
    (OUT/'download_plan.json').write_text(json.dumps(assets,indent=2),encoding='utf-8')
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(download,assets))
    (OUT/'sources.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    evidence=OUT/'licence_evidence';evidence.mkdir(exist_ok=True)
    for name in ('charge_fire.html','charge_materials.html','charge_muzzle.html','jangafx_licence.html','blender_terms.html','cc_ai_guidance.html','evidence.json'):
        (evidence/name).write_bytes((AUDIT/name).read_bytes())
    print('All selected free sources downloaded; source/render validation remains separate.',flush=True)


if __name__=='__main__':main()
