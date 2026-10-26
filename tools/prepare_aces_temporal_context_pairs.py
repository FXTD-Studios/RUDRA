"""Build registered exact-view CC0 pairs with known subpixel planar motion."""
import json,sys
from pathlib import Path
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.run05_corpus import load_plate,NETWORK_PER_SCENE
from rudra.delivery.colorspace import convert
from training.aces_model import reconstruction_weight
from training.aces_reference import reference_view
from tools.prepare_aces_consistent_pairs import save_source
from tools.summarize_aces_generated import sha


def main():
    regpath=Path('configs/experiments/aces_temporal_context_pairs_20261024.json');reg=json.loads(regpath.read_text())
    manifestpath=Path(reg['source_manifest']);manifest=json.loads(manifestpath.read_text())
    root=Path('G:/datasets/corpora/aces_temporal_context_pairs_20261024');root.mkdir(exist_ok=False)
    (root/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for split in ('train','val'):(root/split).mkdir()
    torch.set_num_threads(2);view=reference_view();view.set_threads(4);reports={'train':[],'val':[]}
    height,width=reg['height'],reg['width'];gy,gx=np.mgrid[:height,:width].astype(np.float32)
    for scene_index,origin in enumerate(manifest['rows']):
        if not origin['commercial_ok'] or origin['licence_licence']!='CC0' or sha(origin['hdr_path'])!=origin['origin_sha256']:raise ValueError('origin identity/rights')
        linear=convert(load_plate(origin)/NETWORK_PER_SCENE,'rec2020','ap1').astype(np.float32)
        h,w=linear.shape[:2];scale=max(height/h,width/w)
        linear=cv2.resize(linear,(max(width,round(w*scale)),max(height,round(h*scale))),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
        yy=(linear.shape[0]-height)//2;xx=(linear.shape[1]-width)//2
        linear=linear[yy:yy+height,xx:xx+width].copy()
        for exposure_index,fraction in enumerate(reg['fully_clipped_fractions']):
            low,high=-12.,20.
            for _ in range(14):
                mid=(low+high)/2
                actual=float((view.forward(linear*2.**mid).min(-1)>=254.5/255).mean())
                if actual<fraction:low=mid
                else:high=mid
            ev=high;group=f'{origin["asset_id"]}__context_{exposure_index}'
            for frame,(dx,dy) in enumerate(reg['translations_xy']):
                reference=cv2.remap(linear,gx-dx,gy-dy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)*np.float32(2.**ev)
                if not np.isfinite(reference).all():raise ValueError('nonfinite reference')
                clean=np.clip(view.forward(reference),0,1);quantized=np.rint(clean*255).astype(np.float32)/255
                split=origin['split'];index=len(reports[split]);folder=root/split
                sp=folder/f'{index:04d}_source.png';tp=folder/f'{index:04d}_target.npy';rp=folder/f'{index:04d}_reference.npy'
                source,storage_error=save_source(sp,quantized)
                weight=reconstruction_weight(torch.from_numpy(source).permute(2,0,1)[None])[0,0].numpy()
                hi=(weight>0)&(source.max(-1)>.5);shadow=(weight>0)&~hi
                bounded=np.clip(clean,source-.49/255,source+.49/255).clip(0,1)
                target=view.inverse(bounded);target[hi]=reference[hi]
                if shadow.any():target[shadow]=view.inverse(source[shadow])
                if not np.isfinite(target).all():raise ValueError('nonfinite target')
                np.save(tp,target);np.save(rp,reference)
                error=np.abs(view.forward(target)-source).max(-1)*255
                full=source.min(-1)>=254.5/255
                row={'asset_id':group+f'_f{frame}','scene_id':origin['scene_id'],'parent_asset_id':origin['asset_id'],
                    'split':split,'group_id':group,'frame':frame,'translation_xy':[dx,dy],'valid_border':4,
                    'commercial_ok':True,'licence':'CC0','origin_path':origin['hdr_path'],'origin_sha256':origin['origin_sha256'],
                    'sdr_path':str(sp),'sdr_sha256':sha(sp),'target_path':str(tp),'target_sha256':sha(tp),
                    'reference_path':str(rp),'reference_sha256':sha(rp),'render_ev':ev,
                    'desired_fully_clipped_fraction':fraction,'fully_clipped_fraction':float(full.mean()),
                    'highlight_pixels':int(hi.sum()),'shadow_pixels':int(shadow.sum()),
                    'saved_source_encoding_error_codes':storage_error,'target_display_gt1_pixels':int((error>1).sum()),
                    'target_display_max_error_codes':float(error.max())}
                reports[split].append(row)
        print('scene',scene_index+1,origin['scene_id'],flush=True)
    for split,rows in reports.items():
        (root/split/'report.json').write_text(json.dumps({'rows':rows,'view':view.identity,'source_manifest_sha256':sha(manifestpath),'scope':reg['scope']},indent=2))
    train,val=reports['train'],reports['val'];allrows=train+val
    checks={'counts':len(train)==288 and len(val)==72,
        'scene_disjoint':not ({r['scene_id'] for r in train}&{r['scene_id'] for r in val}),
        'original_content_disjoint':not ({r['origin_sha256'] for r in train}&{r['origin_sha256'] for r in val}),
        'saved_source_disjoint':not ({r['sdr_sha256'] for r in train}&{r['sdr_sha256'] for r in val}),
        'target_display_within1':all(r['target_display_gt1_pixels']==0 for r in allrows),
        'train_full_scene_coverage':len({r['scene_id'] for r in train if r['fully_clipped_fraction']>0})>=8,
        'val_full_scene_coverage':len({r['scene_id'] for r in val if r['fully_clipped_fraction']>0})>=3}
    summary={'checks':checks,'build_pass':all(checks.values()),'used_for_optimization':False,'registration_sha256':sha(regpath),
        'source_manifest_sha256':sha(manifestpath),'builder_sha256':sha(__file__),'scope':reg['scope'],'release_authorized':False}
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
