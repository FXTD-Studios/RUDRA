"""Sampled colour fidelity audit including regions excluded by ordinary-only R."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.rudra_aces_export import video_frames,sha
from rudra.delivery.exr import read_exr
from training.aces_reference import reference_view


def audit(path):
    report=json.loads(Path(path).read_text());saved={r['frame']:r for r in report['frames'] if r.get('exr')}
    view=reference_view();rows=[]
    for i,source in enumerate(video_frames(report['provenance']['input'],max_frames=report['checks']['frames'])):
        if i not in saved:continue
        row=saved[i]
        if sha(row['exr'])!=row['sha256']:raise ValueError('EXR hash mismatch')
        rgb,attrs=read_exr(row['exr']);mask=attrs['reconstruction_mask']>0
        error=np.abs(view.forward(rgb)-source).max(-1)*255
        masks={'unmarked':~mask,'reconstructed':mask,
               'partial_channel_highlight':mask&(source.min(-1)<254.5/255),
               'fully_clipped_highlight':mask&(source.min(-1)>=254.5/255)}
        regions={}
        for name,region in masks.items():
            e=error[region]
            regions[name]={'pixels':int(e.size),'over1code_pixels':int((e>1).sum()),
                           'over4codes_pixels':int((e>4).sum()),
                           'max_code_error':float(e.max()) if e.size else None,
                           'p99_code_error':float(np.percentile(e,99)) if e.size else None}
        rows.append({'frame':i,'regions':regions})
    return {'clip':Path(path).parent.name,'report_sha256':sha(path),'samples':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    rows=[]
    for path in sorted(a.root.glob('*/report.json')):
        row=audit(path);rows.append(row);print(row['clip'],len(row['samples']),flush=True)
    a.out.write_text(json.dumps({'clips':rows,'scope':'saved EXR sample frames only; includes masked highlight fidelity; no HDR ground truth and no modification of registered acceptance'},indent=2))


if __name__=='__main__':main()
