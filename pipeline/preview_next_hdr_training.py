"""Overview and 1:1 crop diagnostics for the new 1080p pairs, no training."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw
from pipeline.prepare_next_hdr_training import OUT,old,dump,sha

def display(rgb):
    return np.uint8(np.rint(np.clip(rgb,0,1)*255))

def srgb(linear):
    linear=np.maximum(linear,0)
    return np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)

def main():
    records=[json.loads(p.read_text()) for p in sorted((OUT/'records').glob('*.json'))]
    chosen={}
    for r in records:
        if r['variant']=='clean':chosen.setdefault((r['source_id'],r['category']),r)
    folder=OUT/'visual_diagnostics';folder.mkdir(exist_ok=True)
    checks=[]
    for index,r in enumerate(chosen.values()):
        sdr=cv2.imread(r['sdr_path'],cv2.IMREAD_UNCHANGED)[...,::-1].astype(np.float32)/65535
        slinear=np.where(sdr<=.04045,sdr/12.92,((sdr+.055)/1.055)**2.4)
        m=np.linalg.inv(old.render.M_709_2020).T
        sview=srgb(slinear@m)
        encoded=cv2.imread(r['hdr_path'],cv2.IMREAD_UNCHANGED)[...,::-1]
        hdr=old.decode_hdr_u16(encoded,old.render.STORAGE)
        hlinear=np.maximum(hdr@m,0)
        hview=srgb(hlinear/(1+hlinear))
        linear_path=Path(r.get('linear_exr_path',r['source_path']))
        source=old.read_exr(linear_path)
        if 'linear_exr_path' not in r:source*=r['radiometric_scale']
        source=np.maximum(source,0);lview=srgb(source/(1+source))
        panels=[display(x) for x in (sview,hview,lview)]
        canvas=Image.new('RGB',(1536,850),(20,20,20));draw=ImageDraw.Draw(canvas)
        for col,(a,label) in enumerate(zip(panels,('SDR display preview','HDR diagnostic compression','Scene-linear diagnostic compression'))):
            canvas.paste(Image.fromarray(a).resize((512,288)),(col*512,0))
            canvas.paste(Image.fromarray(a[284:796,704:1216]),(col*512,320))
            draw.text((col*512+4,292),label,fill='white')
            draw.text((col*512+4,835),'Centre crop: 1 source pixel = 1 preview pixel',fill='white')
        path=folder/f'{index:02d}_{r["category"]}.jpg';canvas.save(path)
        checks.append(dict(asset_id=r['asset_id'],category=r['category'],preview=str(path),preview_sha256=sha(path),
            hdr_peak_nits=float(hdr.max()*203),source_max=float(source.max()),finite=bool(np.isfinite(source).all()),
            note='Independent diagnostic compression for HDR/linear source; not display matching, not calibrated HDR review.'))
    dump(folder/'diagnostic_records.json',dict(count=len(checks),checks=checks))
    print(json.dumps({'diagnostic_categories':len(checks)}))

if __name__=='__main__':main()
