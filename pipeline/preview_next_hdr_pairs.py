"""Read-only source diagnostics: simulated SDR beside exposure-normalized HDR.

These previews do not establish the target colour/exposure mapping.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from pipeline.review_next_hdr_sources import ROOT, read_exr
from pipeline.collect_next_hdr_sources import dump

def main():
    out = ROOT / 'sihdr'
    base = out / 'unpacked/sihdr'
    refs = sorted((base / 'reference').glob('*.exr'))
    chosen = {refs[i] for i in np.linspace(0, len(refs)-1, 9, dtype=int)}
    chosen.update(base / 'reference' / (n + '.exr') for n in ('080', '194', '195'))
    chosen = sorted(chosen)
    records = []
    for offset in range(0, len(chosen), 4):
        subset = chosen[offset:offset+4]
        canvas = Image.new('RGB', (960, 248*len(subset)), (20,20,20))
        draw = ImageDraw.Draw(canvas)
        for row, p in enumerate(subset):
            a, _ = read_exr(p)
            a = np.maximum(a, 0)
            a *= .18 / max(float(np.median(a[::8,::8])), 1e-8)
            a = (a / (1+a)) ** (1/2.2)
            images = [Image.fromarray((np.clip(a,0,1)*255).astype('uint8'))]
            inputs = [base/'input'/clip/(p.stem+'.png') for clip in ('clip_95','clip_97')]
            for src in inputs:
                with Image.open(src) as im:
                    images.append(im.convert('RGB'))
            for col, (im, label) in enumerate(zip(images, ('HDR diagnostic exposure', 'SDR clip_95', 'SDR clip_97'))):
                im.thumbnail((320,220))
                canvas.paste(im,(col*320,row*248))
                draw.text((col*320+4,row*248+224), p.stem+' '+label, fill='white')
            records.append(dict(reference=str(p),inputs=[str(x) for x in inputs]))
        canvas.save(out/f'paired_diagnostic_{offset//4+1:02d}.jpg')
    dump(out/'paired_diagnostic_manifest.json',dict(pairs=records,
        scope='Representative montage including the three ceiling-plateau flags; original SDR display values and independently normalized HDR diagnostic preview.',
        not_a_colour_transform_or_full_resolution_review=True))
    print(json.dumps({'sampled_references':len(records),'sheets':(len(records)+3)//4}))

if __name__ == '__main__':
    main()
