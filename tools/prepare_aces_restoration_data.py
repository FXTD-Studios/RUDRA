"""Original procedural restoration pairs; no external media or HDR-truth claim."""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.run05_corpus import canonical
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from training.aces_model import reconstruction_weight
from rudra.aces_view import ACES2View
from tools.evaluate_run05_dequant import sha


def clean_frame(rng,index,size=256,coverage='legacy'):
    y,x=np.mgrid[-1:1:complex(size),-1:1:complex(size)]
    angle=rng.uniform(0,2*np.pi);u=x*np.cos(angle)+y*np.sin(angle)
    family=index%4
    if family==0:field=u
    elif family==1:field=np.sin(u*rng.uniform(.5,3)+rng.uniform(-2,2))
    elif family==2:field=np.exp(-((x-rng.uniform(-.5,.5))**2+(y-rng.uniform(-.5,.5))**2)*rng.uniform(.5,4))
    else:field=u+.5*(x>rng.uniform(-.5,.5))
    field=(field-field.min())/(field.max()-field.min())
    low=rng.uniform(.03,.65,3);span=rng.uniform(.03,.32,3)
    if coverage=='balanced':
        group=(index//4)%6
        if group==1:low=np.repeat(rng.uniform(.03,.4),3);span=np.repeat(rng.uniform(.03,.55),3)
        elif group==2:low=np.repeat(rng.uniform(.02,.05),3);span=np.repeat(rng.uniform(.01,.06),3)
        elif group==3:low=np.repeat(rng.uniform(.76,.85),3);span=np.repeat(rng.uniform(.01,.1),3)
        elif group==4:low=rng.uniform(.02,.055,3);span=rng.uniform(.01,.05,3)
        elif group==5:low=rng.uniform(.76,.85,3);span=rng.uniform(.01,.09,3)
    elif coverage!='legacy':raise ValueError('unknown coverage')
    return (low+field[...,None]*span).astype(np.float32),family


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--coverage',choices=('legacy','balanced'),default='legacy')
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    torch.set_num_threads(2);view=ACES2View();a.out.mkdir(parents=True)
    reports={}
    splits=(('train',96,20261027),('val',24,20261028)) if a.coverage=='balanced' else (('train',32,20261025),('val',8,20261026))
    for split,count,seed in splits:
        rng=np.random.default_rng(seed);folder=a.out/split;folder.mkdir();rows=[]
        for i in range(count):
            clean,family=clean_frame(rng,i,coverage=a.coverage);quantized=np.rint(clean*255)/255
            path=folder/f'{i:03d}_sdr.png'
            encoded=np.rint(canonical(quantized)*65535).astype(np.uint16)
            if not cv2.imwrite(str(path),encoded[...,::-1]):raise ValueError('PNG write failed')
            source=canonical_to_source(load_rgb(path,hdr=False))
            bounded=np.minimum(np.maximum(clean,source-.49/255),source+.49/255)
            target=view.inverse(bounded);target_path=folder/f'{i:03d}_target.npy';np.save(target_path,target)
            clean_path=folder/f'{i:03d}_clean_codes.npy';np.save(clean_path,clean)
            if reconstruction_weight(torch.from_numpy(source).permute(2,0,1)[None]).any():raise ValueError('restoration-only support expected')
            error=float(np.abs(view.forward(target)-source).max()*255)
            if error>1:raise ValueError('target roundtrip failed')
            asset=f'procedural_restoration_{seed}_{i:03d}'
            rows.append({'asset_id':asset,'scene_id':asset,'split':split,'family':family,'seed':seed,
                         'coverage':a.coverage,'colour_group':(i//4)%6 if a.coverage=='balanced' else None,
                         'sdr_path':str(path),'sdr_sha256':sha(path),'target_path':str(target_path),'target_sha256':sha(target_path),
                         'clean_codes_path':str(clean_path),'clean_codes_sha256':sha(clean_path),'max_source_code_error':error,
                         'commercial_ok':True,'provenance':'original procedural generation, no third-party media; no public licence dedication',
                         'target_role':'restoration only, exact ACES inverse of bounded continuous display codes; not recovered physical HDR'})
        report={'rows':rows,'view':view.identity,'builder_sha256':sha(Path(__file__)),
                'scope':'procedural restoration supplement; val is synthetic development only, not independent product holdout'}
        (folder/'report.json').write_text(json.dumps(report,indent=2));reports[split]=report
    if {r['sdr_sha256'] for r in reports['train']['rows']}&{r['sdr_sha256'] for r in reports['val']['rows']}:raise ValueError('content overlap')
    (a.out/'summary.json').write_text(json.dumps({'train':len(reports['train']['rows']),'development':len(reports['val']['rows']),'external_media':False,
        'coverage':a.coverage,
        'max_target_source_code_error':max(r['max_source_code_error'] for p in reports.values() for r in p['rows']),
        'scene_and_source_content_disjoint':True,'used_in_current_experiment':False},indent=2))
    print((a.out/'summary.json').read_text())


if __name__=='__main__':main()
