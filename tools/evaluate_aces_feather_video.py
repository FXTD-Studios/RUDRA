"""Registered full Hunyuan48 diagnostic for feather8; no video transport claim."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.rudra_aces_export import video_frames,load_model,sha
from tools.evaluate_aces_temporal import LUMA_AP1,cell_luminance,temporal_change
from training.aces_reference import reference_view
from training.aces_reliable_highlights import feather_full
from training.aces_colour_backtrack import constrain_colour


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_feather8_video_20261024.json'))
    parser.add_argument('--out',type=Path,default=Path('G:/rudra_training/aces_feather8_ablation_20261024/hunyuan48.json'))
    args=parser.parse_args();regpath=args.registration;reg=json.loads(regpath.read_text())
    suite=json.loads(Path(reg['source_registration']).read_text());source=next(x for x in suite['expected_inputs'] if x['clip']==reg['clip'])
    out=args.out;checkpoint=reg.get('candidate',suite['candidate']);radius=reg.get('feather_radius',8)
    if out.exists():raise ValueError('new output required')
    if sha(source['path'])!=source['sha256'] or sha(checkpoint)!=reg['checkpoint_sha256']:raise ValueError('identity')
    torch.set_num_threads(2);view=reference_view();view.set_threads(4);model,_=load_model(checkpoint,'cuda')
    rows=[];previous=None
    for i,source_rgb in enumerate(video_frames(source['path'],max_frames=reg['frames'],decode_bits=reg.get('decode_bits',8))):
        with torch.inference_mode():result=model(torch.from_numpy(source_rgb).permute(2,0,1)[None].to('cuda'))
        pred=result.acescg[0].permute(1,2,0).cpu().numpy();base=result.baseline[0].permute(1,2,0).cpu().numpy();w=result.reconstruction_weight[0,0].cpu().numpy()
        if reg.get('clipped_smoothing'):
            from training.aces_clipped_smoothing import smooth_clipped
            pred,_=smooth_clipped(pred,source_rgb,**reg['clipped_smoothing'])
        rgb,diag=feather_full(source_rgb,pred,base,w,view,radius) if radius else constrain_colour(source_rgb,pred,base,w,view,include_fully_clipped=reg.get('include_fully_clipped',False),repair_inverse_boundary=reg.get('repair_inverse_boundary',False),forbid_new_negative_y=reg.get('forbid_new_negative_y',False))
        shown=view.forward(rgb);error=np.abs(shown-source_rgb).max(-1)*255
        shadow=(w>0)&(source_rgb.max(-1)<=.5);current=(cell_luminance(rgb),cell_luminance(base))
        row={'frame':i,'diagnostic':diag,'all_max_code_error':float(error.max()),
             'highlight_over1':int(((error>1)&(w>0)&(source_rgb.max(-1)>.5)).sum()),
             'ordinary_within1':float((error[w==0]<=1).mean()) if (w==0).any() else 1.,
             'shadow_lift_gt1':int((((shown-source_rgb).max(-1)*255>1)&shadow).sum()),
             'new_negativeY':int((((rgb@LUMA_AP1)<0)&((base@LUMA_AP1)>=0)).sum())}
        if previous is not None:row['transition']=temporal_change(previous[0],current[0],previous[1],current[1])
        previous=current;rows.append(row);print('frame',i,flush=True)
    changes=[r['transition']['mean_abs_excess_ev'] for r in rows[1:]]
    checks={'complete':len(rows)==reg['frames'],'temporal':all(v is not None and v<=.02 for v in changes),
            'ordinary':all(r['ordinary_within1']>=.999 for r in rows),
            'colour':all(r['diagnostic']['remaining_display_errors']==0 for r in rows),
            'no_shadow_lift':all(r['shadow_lift_gt1']==0 for r in rows),
            'no_new_negativeY':all(r['new_negativeY']==0 for r in rows),
            'outside_support':all(r['diagnostic']['outside_support_bit_exact'] for r in rows)}
    if reg.get('all_highlight_appearance'):checks['all_highlight_appearance']=all(r['highlight_over1']==0 for r in rows)
    report={'rows':rows,'checks':checks,'numerical_pass':all(checks.values()),'max_transition_mean_ev':max(changes),'decode_bits':reg.get('decode_bits',8),
            'source_sha256':source['sha256'],'checkpoint_sha256':reg['checkpoint_sha256'],'registration_sha256':sha(regpath),
            'tool_sha256':sha(__file__),'policy_sha256':sha('training/aces_reliable_highlights.py' if radius else 'training/aces_colour_backtrack.py'),'scope':reg['scope'],'release_authorized':False}
    out.write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ('checks','max_transition_mean_ev')},indent=2))


if __name__=='__main__':main()
