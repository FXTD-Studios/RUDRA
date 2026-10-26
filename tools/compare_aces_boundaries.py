"""Compare TRAIN target joins; boundary metrics are diagnostics, not quality gates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rudra.aces_view import ACES2View
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from training.aces_model import reconstruction_weight
from tools.evaluate_run05_dequant import sha
import torch


def boundary_error(codes,baseline,mask):
    values=[]
    for axis in (0,1):
        edge=np.diff(mask.astype(np.int8),axis=axis)!=0
        delta=np.abs(np.diff(codes,axis=axis)-np.diff(baseline,axis=axis)).max(-1)*255
        values.append(delta[edge])
    values=np.concatenate(values)
    return {'edges':len(values),'mean_excess_code':float(values.mean()) if len(values) else 0,
            'p99_excess_code':float(np.quantile(values,.99)) if len(values) else 0}


def main():
    torch.set_num_threads(2)
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('old','new','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output directory required')
    old=json.loads(a.old.read_text());new=json.loads(a.new.read_text())
    assert old['source_manifest_sha256']==new['source_manifest_sha256']
    assert len(old['rows'])==len(new['rows'])
    a.out.mkdir(parents=True);view=ACES2View();results=[]
    for i,(o,n) in enumerate(zip(old['rows'],new['rows'])):
        assert o['asset_id']==n['asset_id'] and o['sdr_sha256']==n['sdr_sha256']
        for row in (o,n):
            assert sha(row['target_path'])==row['target_sha256']
        assert sha(o['sdr_path'])==o['sdr_sha256']
        source=canonical_to_source(load_rgb(o['sdr_path'],hdr=False))
        baseline=view.inverse(source);before=np.load(o['target_path']);after=np.load(n['target_path'])
        weight=reconstruction_weight(torch.from_numpy(source).permute(2,0,1)[None])[0,0].numpy()
        metrics={};sheet=Image.new('RGB',(1440,3*294),'white');draw=ImageDraw.Draw(sheet)
        for j,ev in enumerate((-3,0,3)):
            codes=[view.forward(x*2.**ev) for x in (baseline,before,after)]
            metrics[str(ev)]={'old':boundary_error(codes[1],codes[0],weight>0),
                              'new':boundary_error(codes[2],codes[0],weight>0)}
            if i in (0,2,7,9):
                for k,(label,rgb) in enumerate(zip(('analytic','scalar target','boundary target'),codes)):
                    draw.text((480*k+5,294*j+4),f'{label} {ev:+}EV',fill='black')
                    im=Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).resize((480,270))
                    sheet.paste(im,(480*k,294*j+24))
        if i in (0,2,7,9):sheet.save(a.out/f'review_{i:02d}.png')
        results.append({'asset_id':o['asset_id'],'boundary_metrics':metrics})
    (a.out/'report.json').write_text(json.dumps({'old_sha256':sha(a.old),'new_sha256':sha(a.new),
        'rows':results,'scope':'TRAIN boundary discrepancy against analytic baseline; not true HDR accuracy or release evidence'},indent=2))


if __name__=='__main__':main()
