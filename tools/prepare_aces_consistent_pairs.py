"""Original CC0 references rendered through the same view used to evaluate targets."""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.run05_corpus import load_plate,NETWORK_PER_SCENE,canonical
from rudra.delivery.colorspace import convert
from training.aces_model import reconstruction_weight
from training.aces_reference import reference_view
from tools.summarize_aces_generated import sha
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def save_source(path,quantized):
    encoded=np.rint(canonical(quantized)*65535).astype(np.uint16)
    if not cv2.imwrite(str(path),cv2.cvtColor(encoded,cv2.COLOR_RGB2BGR)):
        raise ValueError('PNG write failed')
    decoded=canonical_to_source(load_rgb(path,hdr=False))
    error=float(np.abs(decoded-quantized).max()*255)
    if error>.1:raise ValueError(f'saved source encoding drift: {error} codes')
    return decoded,error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('G:/datasets/corpora/aces_consistent_pairs_20261024'))
    parser.add_argument('--exposure-offsets',type=float,nargs='+',default=[0.])
    parser.add_argument('--unique-scenes',action='store_true')
    parser.add_argument('--min-highlight-scenes',type=int,default=1)
    parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_consistent_pairs_20261024.json'))
    args=parser.parse_args()
    torch.set_num_threads(2)
    out=args.out;out.mkdir(exist_ok=False)
    (out/'builder_snapshot.py').write_bytes(Path(__file__).read_bytes())
    manifest_path=Path('G:/datasets/corpora/rudra_run05_observed_clip_expansion_20261022/manifest.jsonl')
    sources_path=Path('G:/datasets/corpora/rudra_run05_sources_20261022/sources.jsonl')
    manifest={r['asset_id']:r for r in map(json.loads,manifest_path.read_text().splitlines())}
    origins={r['asset_id']:r for r in map(json.loads,sources_path.read_text().splitlines())}
    view=reference_view();view.set_threads(4);reports={}
    for split,oldpath in (
        ('train','G:/datasets/corpora/run05_aces_spatial_train48_v2_20261023/report.json'),
        ('val','G:/datasets/corpora/run05_aces_spatial_dev24_v2_20261023/report.json')):
        old=json.loads(Path(oldpath).read_text())
        assert old['source_manifest_sha256']==sha(manifest_path) and old['sources_sha256']==sha(sources_path)
        folder=out/split;folder.mkdir();rows=[]
        selected=[];seen=set()
        for row in old['rows']:
            if args.unique_scenes and row['scene_id'] in seen:continue
            seen.add(row['scene_id'])
            selected.extend((row,offset) for offset in args.exposure_offsets)
        for i,(row,offset) in enumerate(selected):
            parent=manifest[row['asset_id']];origin=origins[parent['parent_asset_id']]
            assert origin['commercial_ok'] and origin['licence_licence']=='CC0'
            assert sha(origin['hdr_path'])==parent['parent_hdr_sha256'] and parent['split']==split
            render_ev=parent['render_ev']+offset
            reference=convert(load_plate(origin)/NETWORK_PER_SCENE*(2.**render_ev),'rec2020','ap1').astype(np.float32)
            clean=view.forward(reference).clip(0,1)
            quantized=(np.rint(clean*255)/255).astype(np.float32)
            sdr=folder/f'{i:03d}_source.png';tp=folder/f'{i:03d}_target.npy';rp=folder/f'{i:03d}_reference.npy'
            source,encoding_error=save_source(sdr,quantized)
            w=reconstruction_weight(torch.from_numpy(source).permute(2,0,1)[None])[0,0].numpy()
            hi=(w>0)&(source.max(-1)>.5);shadow=(w>0)&~hi
            corrected=np.clip(clean,source-.49/255,source+.49/255).clip(0,1).astype(np.float32)
            target=view.inverse(corrected);target[hi]=reference[hi]
            target[shadow]=view.inverse(source[shadow]) if shadow.any() else target[shadow]
            np.save(tp,target);np.save(rp,reference)
            error=np.abs(view.forward(target)-source).max(-1)*255
            record={'asset_id':row['asset_id']+f'__consistent_ev{offset:+g}','parent_asset_id':row['asset_id'],
                'scene_id':row['scene_id'],'split':split,'commercial_ok':True,'licence':'CC0',
                'origin_path':origin['hdr_path'],'origin_sha256':parent['parent_hdr_sha256'],
                'render_ev':render_ev,'exposure_offset':offset,'sdr_path':str(sdr),'sdr_sha256':sha(sdr),
                'target_path':str(tp),'target_sha256':sha(tp),'reference_path':str(rp),'reference_sha256':sha(rp),
                'highlight_pixels':int(hi.sum()),'shadow_pixels':int(shadow.sum()),
                'target_display_max_error_codes':float(error.max()),'target_display_gt1_pixels':int((error>1).sum())}
            record['saved_source_encoding_error_codes']=encoding_error
            record['source_storage']='canonical Rec.2020 sRGB uint16; decode through canonical_to_source'
            rows.append(record);print(split,i+1,record['highlight_pixels'],record['target_display_gt1_pixels'],flush=True)
        reports[split]={'rows':rows,'view':view.identity,'parent_pilot_sha256':sha(oldpath),
            'sources_sha256':sha(sources_path),'source_manifest_sha256':sha(manifest_path),
            'scope':'exact-view paired development corpus; original linear reference retained; not independent release holdout'}
        (folder/'report.json').write_text(json.dumps(reports[split],indent=2))
    train,val=[reports[k]['rows'] for k in ('train','val')]
    checks={'scene_disjoint':not ({r['scene_id'] for r in train}&{r['scene_id'] for r in val}),
            'source_disjoint':not ({r['sdr_sha256'] for r in train}&{r['sdr_sha256'] for r in val}),
            'target_display_within1':all(r['target_display_gt1_pixels']==0 for r in train+val),
            'train_highlight_coverage':sum(r['highlight_pixels'] for r in train)>=1000,
            'dev_highlight_coverage':sum(r['highlight_pixels'] for r in val)>=100,
            'train_highlight_scene_coverage':len({r['scene_id'] for r in train if r['highlight_pixels']})>=args.min_highlight_scenes,
            'dev_highlight_scene_coverage':len({r['scene_id'] for r in val if r['highlight_pixels']})>=args.min_highlight_scenes}
    summary={'checks':checks,'build_pass':all(checks.values()),'used_for_optimization':False,
             'registration_sha256':sha(args.registration),
             'exposure_offsets':args.exposure_offsets,'unique_scenes':args.unique_scenes,
             'builder_sha256':sha(__file__),'highlight_pixels':{k:sum(r['highlight_pixels'] for r in reports[k]['rows']) for k in reports},
             'release_authorized':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
