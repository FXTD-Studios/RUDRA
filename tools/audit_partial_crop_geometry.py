"""Fixed TRAIN crop/full-frame comparison for partial-route geometry."""
import json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint,reconstruction_weight
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def main():
    root=Path('G:/rudra_training/partial_crop_geometry_20261025');root.mkdir(exist_ok=False)
    checkpoint='G:/rudra_training/claude_partial_finetune_20261025/ft_rel3_w05/step400.pt'
    pilots=['G:/datasets/corpora/aces_consistent_exposures_v2_20261024/train/report.json',
            'G:/datasets/corpora/aces_temporal_context_pairs_20261024/train/report.json']
    registration={'checkpoint':checkpoint,'checkpoint_sha256':sha(checkpoint),'pilots':{p:sha(p) for p in pilots},
                  'selection':'First four eligible distinct TRAIN scenes per pilot in stored order; median partial pixel inside80px border; crop192; aligned origin floor4 and +1/+1 control; borders48 and64.',
                  'scope':'Geometry diagnosis only; no training, quality selection or holdout claim.'}
    (root/'registration.json').write_text(json.dumps(registration,indent=2));(root/'auditor.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2);reference_view().set_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model_from_checkpoint(torch.load(checkpoint,map_location='cpu',weights_only=True)).eval().cuda()
    tensor=lambda a:torch.from_numpy(np.ascontiguousarray(a)).permute(2,0,1)[None].cuda()
    output=lambda x:model(tensor(x)).acescg[0].permute(1,2,0).cpu().numpy()
    rows=[]
    for pilot in pilots:
        seen=set()
        for row in json.loads(Path(pilot).read_text())['rows']:
            if row['split']!='train':raise ValueError('TRAIN only')
            if row['scene_id'] in seen:continue
            if sha(row['sdr_path'])!=row['sdr_sha256']:raise ValueError('source identity')
            source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));h,w=source.shape[:2]
            if min(h,w)<224:continue
            weight=reconstruction_weight(tensor(source))[0,0].cpu().numpy()
            mask=(weight>0)&(source.max(-1)>.5)&(source.min(-1)<254.5/255)
            eligible=mask.copy();eligible[:80]=False;eligible[-80:]=False;eligible[:,:80]=False;eligible[:,-80:]=False
            points=np.argwhere(eligible)
            if len(points)<64:continue
            cy,cx=points[len(points)//2];y=int(np.clip(cy-96,0,h-196))//4*4;x=int(np.clip(cx-96,0,w-196))//4*4
            with torch.inference_mode():full=output(source)
            measurements=[]
            for phase in (0,1):
                yy,xx=y+phase,x+phase
                with torch.inference_mode():crop=output(source[yy:yy+192,xx:xx+192])
                reference=full[yy:yy+192,xx:xx+192]
                for border in (48,64):
                    valid=mask[yy:yy+192,xx:xx+192].copy();valid[:border]=False;valid[-border:]=False;valid[:,:border]=False;valid[:,-border:]=False
                    if not valid.any():raise ValueError('empty comparison')
                    error=np.abs(np.arcsinh(crop[valid])-np.arcsinh(reference[valid]))
                    measurements.append({'phase':phase,'border':border,'pixels':int(valid.sum()),'max_asinh_error':float(error.max()),'mean_asinh_error':float(error.mean()),'exact':bool(np.array_equal(crop[valid],reference[valid]))})
            rows.append({'asset_id':row['asset_id'],'scene':row['scene_id'],'shape':[h,w],'origin':[y,x],'measurements':measurements})
            seen.add(row['scene_id']);print('audited',len(rows),flush=True)
            if len(seen)==4:break
        if len(seen)!=4:raise ValueError('insufficient coverage')
    summary={f'phase{p}_border{b}':{'max_asinh_error':max(m['max_asinh_error'] for r in rows for m in r['measurements'] if m['phase']==p and m['border']==b),
             'mean_image_asinh_error':float(np.mean([m['mean_asinh_error'] for r in rows for m in r['measurements'] if m['phase']==p and m['border']==b]))} for p in (0,1) for b in (48,64)}
    result={'rows':rows,'summary':summary,'registration_sha256':sha(root/'registration.json'),'tool_sha256':sha(__file__),'release_authorized':False}
    (root/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
