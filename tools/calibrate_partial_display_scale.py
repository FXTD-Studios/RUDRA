"""Fixed analytic display-error denominators from the exact TRAIN exposure schedule."""
import json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_reference import reference_view
from training.aces_model import reconstruction_weight
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def main():
    root=Path('G:/rudra_training/partial_temporal1_400_20261025')
    out=Path('G:/rudra_training/partial_display_fixed_scale_20261025');out.mkdir(exist_ok=False)
    pilots=['G:/datasets/corpora/aces_consistent_exposures_v2_20261024/train/report.json',
            'G:/datasets/corpora/aces_temporal_context_pairs_20261024/train/report.json']
    paths=[root/'crop_pool.json',root/'motion_pool.json',root/'schedule.json',*map(Path,pilots)]
    reg={'input_hashes':{str(p):sha(p) for p in paths},'policy':'Sum analytic code-squared error over all partial pixels in the saved400-update/1600-frame TRAIN schedule, separately perEV; divide by total supervised pixels. Floor1e-4 retained. No model prediction or DEV input.', 'release_authorized':False}
    (out/'registration.json').write_text(json.dumps(reg,indent=2));(out/'calibrator.py').write_bytes(Path(__file__).read_bytes())
    rows={r['asset_id']:r for p in pilots for r in json.loads(Path(p).read_text())['rows']}
    pools=[json.loads((root/name).read_text()) for name in ('crop_pool.json','motion_pool.json')]
    schedule=json.loads((root/'schedule.json').read_text());requests={}
    for i,indices in enumerate(schedule):
        motion=i%2==0
        for index in indices:
            sample=pools[int(motion)][index];assets=sample['assets'] if motion else [sample['asset_id']]
            for asset in assets:
                key=(asset,*sample['origin']);requests[key]=requests.get(key,0)+1
    view=reference_view();view.set_threads(4);torch.set_num_threads(2)
    records=[];totals={ev:0. for ev in (-3,0,3)};pixels=0;cache_asset=None
    for (asset,y,x),frequency in sorted(requests.items()):
        row=rows[asset]
        if row['split']!='train':raise ValueError('TRAIN only')
        if cache_asset!=asset:
            for key in ('sdr','target'):
                if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('input identity')
            source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path']);cache_asset=asset
        s=source[y+64:y+128,x+64:x+128].copy();t=target[y+64:y+128,x+64:x+128].copy()
        weight=reconstruction_weight(torch.from_numpy(s).permute(2,0,1)[None])[0,0].numpy()
        valid=(weight>0)&(s.max(-1)>.5)&(s.min(-1)<254.5/255);n=int(valid.sum())
        if not n:raise ValueError('empty scheduled supervision')
        baseline=view.inverse(s);errors={}
        for ev in totals:
            e=np.square((view.forward(baseline*2.**ev)-view.forward(t*2.**ev)).astype(np.float64)*255).mean(-1)
            total=float(e[valid].sum());totals[ev]+=frequency*total;errors[str(ev)]=total/n
        pixels+=frequency*n;records.append({'asset_id':asset,'origin':[y,x],'exposures':frequency,'pixels':n,'analytic_mse':errors})
    denom={str(ev):max(total/pixels,1e-4) for ev,total in totals.items()}
    result={'denominators':denom,'effective_coefficients':{ev:.5/3/v for ev,v in denom.items()},'scheduled_frames':sum(requests.values()),
            'unique_asset_crops':len(requests),'supervised_pixel_exposures':pixels,'rows':records,'registration_sha256':sha(out/'registration.json'),
            'calibrator_sha256':sha(__file__),'release_authorized':False}
    assert result['scheduled_frames']==1600
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
