"""Fresh grading set from FXTD camera plates (registration claude_fresh_fxtd_grade_20261026.json).

One eligible frame per scene (selection file), area-downsampled to longest side 512 in
linear light, BT.2020 -> AP1, at the plate's native camera exposure plus 0/+1/+2 stops
(v2; v1 auto-exposed every frame nearly white). Uses only the reference, never a model output."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.run05_corpus import load_plate,NETWORK_PER_SCENE
from rudra.delivery.colorspace import convert
from training.aces_model import reconstruction_weight
from training.aces_reference import reference_view
from training.sdr2hdr_dataset import storage_for

CLIP=254.5/255
OFFSETS=(0,1,2)   # v2: native camera exposure plus 0/+1/+2 stops (registration claude_fresh_fxtd_grade_v2_20261026)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clip_fraction(view,ref,ev):
    return float((view.forward(ref*np.float32(2.**ev)).max(-1)>=CLIP).mean())


def solve_ev(view,ref,target):
    lo,hi=-8.,12.
    for _ in range(40):
        mid=(lo+hi)/2
        if clip_fraction(view,ref,mid)<target:lo=mid
        else:hi=mid
    return (lo+hi)/2


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--plates',type=Path,required=True)
    ap.add_argument('--registration',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2);view=reference_view();view.set_threads(4)
    rows=[];sel=json.loads(a.selection.read_text())
    for k,s in enumerate(sel):
        path=a.plates/Path(s['hdr_path'].replace('\\','/')).name
        if storage_for(path) is None:raise ValueError(f'no ingest sentinel (_ingest_config.json) above {path}: the log2_extended plate would be read as linear')
        plate=load_plate({**s,'hdr_path':str(path)})/NETWORK_PER_SCENE
        h,w=plate.shape[:2];f=512/max(h,w)
        plate=cv2.resize(plate,(round(w*f),round(h*f)),interpolation=cv2.INTER_AREA)
        base=np.ascontiguousarray(convert(plate,'rec2020','ap1').astype(np.float32))
        for offset in OFFSETS:
            ev=float(offset);ref=(base*np.float32(2.**ev)).astype(np.float32)
            src=(np.rint(np.clip(view.forward(ref),0,1)*255)/255).astype(np.float32)
            wgt=reconstruction_weight(torch.from_numpy(src).permute(2,0,1)[None])[0,0].numpy()
            high=(wgt>0)&(src.max(-1)>.5);full=high&(src.min(-1)>=CLIP)
            cid=f"{s['asset_id']}__ev{int(offset):+d}"
            median_code=float(np.median(src.max(-1)))
            row={'case_id':cid,'asset_id':s['asset_id'],'scene_id':s['scene_id'],'project':s['project'],'clip_target':offset,'ev':ev,
                 'measured_clip_fraction':float((src.max(-1)>=CLIP).mean()),'median_max_code':median_code,'shape':list(src.shape),
                 'plate_sha256':sha(path),'highlight_pixels':int(high.sum()),'full_pixels':int(full.sum()),'partial_pixels':int((high&~full).sum()),
                 'kept':bool(high.sum()>=100 and median_code<=0.9)}
            if row['kept']:
                sp=a.out/f'{len(rows):03d}_source.npy';rp=a.out/f'{len(rows):03d}_reference.npy'
                np.save(sp,src);np.save(rp,ref);row.update(source_path=str(sp),source_sha256=sha(sp),reference_path=str(rp),reference_sha256=sha(rp))
            rows.append(row)
        print(k+1,s['asset_id'],[r['highlight_pixels'] for r in rows[-2:]],flush=True)
    kept=[r for r in rows if r['kept']]
    report={'rows':rows,'kept_cases':len(kept),'scenes':len({r['scene_id'] for r in kept}),'view':view.identity if isinstance(view.identity,dict) else str(view.identity),
            'registration_sha256':sha(a.registration),'selection_sha256':sha(a.selection),'builder_sha256':sha(__file__),
            'scope':'fresh grading set, FXTD own footage, never used by ACES training/selection; release_authorized false'}
    (a.out/'report.json').write_text(json.dumps(report,indent=1,default=str))
    print('kept',len(kept),'cases from',report['scenes'],'scenes')


if __name__=='__main__':main()
