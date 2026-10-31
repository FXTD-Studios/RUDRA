"""Original linear-light shadow surfaces with exact ACES/8-bit input formation."""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.run05_corpus import canonical
from rudra.aces_view import ACES2View
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from training.aces_model import reconstruction_weight
from tools.evaluate_run05_dequant import sha


def shadow_surface(rng,index,size=256):
    y,x=np.mgrid[-1:1:complex(size),-1:1:complex(size)]
    angle=rng.uniform(0,2*np.pi);u=(x*np.cos(angle)+y*np.sin(angle));u=(u-u.min())/(u.max()-u.min())
    family=index%4
    if family==0:field=u
    elif family==1:field=.5+.5*np.sin(u*rng.uniform(2,7)+rng.uniform(-2,2))
    elif family==2:field=np.exp(-((x-rng.uniform(-.5,.5))**2+(y-rng.uniform(-.5,.5))**2)*rng.uniform(1,5))
    else:field=u*(.15+.85*(x>rng.uniform(-.5,.5)))
    field=(field-field.min())/(field.max()-field.min())
    low=10**rng.uniform(-7,-5);high=10**rng.uniform(-3,-1.8)
    value=low+(high-low)*field
    if index%3==0:value=low*np.power(high/low,field)
    colour=np.ones(3) if index%2==0 else rng.uniform(.35,1.4,3)
    target=value[...,None]*colour
    if index%8==7:target[x<-.8]=0
    return target.astype(np.float32),family


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    torch.set_num_threads(2);view=ACES2View();a.out.mkdir(parents=True);reports={}
    for split,count,seed in (('train',96,20261031),('val',24,20261101)):
        folder=a.out/split;folder.mkdir();rng=np.random.default_rng(seed);rows=[]
        for i in range(count):
            target,family=shadow_surface(rng,i);clean=view.forward(target)
            quantized=np.rint(np.clip(clean,0,1)*255)/255;path=folder/f'{i:03d}_sdr.png'
            encoded=np.rint(canonical(quantized)*65535).astype(np.uint16)
            if not cv2.imwrite(str(path),encoded[...,::-1]):raise ValueError('PNG write failed')
            source=canonical_to_source(load_rgb(path,hdr=False));target_path=folder/f'{i:03d}_target.npy';np.save(target_path,target)
            tensor=torch.from_numpy(source).permute(2,0,1)[None];mask=(reconstruction_weight(tensor)>0)[0,0].numpy()
            error=float(np.abs(clean-source).max()*255)
            if error>1 or not mask.any():raise ValueError('invalid shadow pair coverage/roundtrip')
            asset=f'original_shadow_{seed}_{i:03d}'
            rows.append({'asset_id':asset,'scene_id':asset,'split':split,'seed':seed,'family':family,'neutral':i%2==0,
                         'sdr_path':str(path),'sdr_sha256':sha(path),'target_path':str(target_path),'target_sha256':sha(target_path),
                         'shadow_pixels':int(mask.sum()),'exact_black_source_pixels':int((source.max(-1)==0).sum()),
                         'max_target_source_code_error':error,'commercial_ok':True,
                         'provenance':'original analytic linear-light surface; no external media/code/weights; no public licence dedication',
                         'target_role':'unmodified known analytic ACEScg values, arbitrary scene exposure, not physical nits'})
        report={'rows':rows,'view':view.identity,'builder_sha256':sha(Path(__file__)),
                'scope':'original static shadow training/synthetic development supplement; no real/generator/video or release coverage'}
        (folder/'report.json').write_text(json.dumps(report,indent=2));reports[split]=report
    train=reports['train']['rows'];val=reports['val']['rows']
    for key in ('scene_id','sdr_sha256','target_sha256'):
        if {r[key] for r in train}&{r[key] for r in val}:raise ValueError('split overlap')
    summary={'train':len(train),'development':len(val),'train_shadow_pixels':sum(r['shadow_pixels'] for r in train),
             'train_black_source_pixels':sum(r['exact_black_source_pixels'] for r in train),
             'max_target_source_code_error':max(r['max_target_source_code_error'] for r in train+val),
             'source_target_scene_disjoint':True,'external_media':False,'used_for_optimization':False}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
