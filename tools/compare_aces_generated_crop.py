"""Fixed previously selected failure crop, comparing actual saved EXR display views."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.rudra_aces_export import video_frames,sha
from rudra.delivery.exr import read_exr
from training.aces_reference import reference_view


def main():
    old=Path('G:/rudra_training/aces_v1_generated_review_20261024/cogvideox_res2')
    new=Path('G:/rudra_training/aces_constrained_generated_review_20261024/cogvideox_res2')
    selection=json.loads((old/'highlight_colour_crop.json').read_text());frame=selection['frame'];x,y=selection['xy']
    report=json.loads((new/'report.json').read_text());out=new/'previous_failure_comparison.png'
    if out.exists():raise ValueError('new output required')
    frames=video_frames(report['provenance']['input'],max_frames=frame+1)
    for source in frames:pass
    view=reference_view();images=[('Source',source)];hashes={};metrics={}
    for name,root in (('Previous v7',old),('Constrained head',new)):
        r=json.loads((root/'report.json').read_text());row=next(r for r in r['frames'] if r['frame']==frame)
        if sha(row['exr'])!=row['sha256']:raise ValueError('EXR identity')
        rgb,_=read_exr(row['exr']);shown=view.forward(rgb);images.append((name,shown));hashes[name]=row['sha256']
        metrics[name]={'selected_pixel_max_error_codes':float(np.abs(shown[y,x]-source[y,x]).max()*255)}
    box=(slice(max(0,y-48),y+48),slice(max(0,x-48),x+48))
    sheet=Image.new('RGB',(384*3,410),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,rgb) in enumerate(images):
        panel=Image.fromarray(np.rint(np.clip(rgb[box],0,1)*255).astype(np.uint8)).resize((384,384),Image.Resampling.NEAREST)
        sheet.paste(panel,(384*i,26));draw.text((384*i+8,6),name,fill='black')
    sheet.save(out)
    out.with_suffix('.json').write_text(json.dumps({'frame':frame,'xy':[x,y],'metrics':metrics,'exr_sha256':hashes,
        'scope':'Same fixed worst-case crop previously selected for v7, not representative average; native 96px crop enlarged nearest-neighbour4x'},indent=2))
    print(metrics)


if __name__=='__main__':main()
