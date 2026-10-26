"""Selected failed Hunyuan transition: isolate causal spatial branches, no gate claim."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.rudra_aces_export import video_frames,load_model,sha
from tools.evaluate_aces_temporal import cell_luminance,temporal_change
from training.aces_reference import reference_view
from training.aces_colour_backtrack import constrain_colour


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--feather',action='store_true');args=parser.parse_args()
    reg=json.loads(Path('configs/experiments/aces_constrained_generated_review_20261024.json').read_text())
    source=next(x for x in reg['expected_inputs'] if x['clip']=='hunyuan_demo')
    out=Path('G:/rudra_training/aces_constrained_generated_review_20261024')/('hunyuan_transition40_feather8.json' if args.feather else 'hunyuan_transition40_diagnostic.json')
    if out.exists():raise ValueError('new output required')
    if sha(source['path'])!=source['sha256']:raise ValueError('source identity')
    frames=[rgb for i,rgb in enumerate(video_frames(source['path'],max_frames=41)) if i in (39,40)]
    torch.set_num_threads(2);view=reference_view();view.set_threads(4);methods={};stats=[]
    for name,checkpoint in (('v7','G:/rudra_training/aces_v1_analytic_shadow_20261024/candidate.pt'),('equal_head',reg['candidate'])):
        model,_=load_model(checkpoint,'cuda')
        for i,rgb in enumerate(frames):
            with torch.inference_mode():pred=model(torch.from_numpy(rgb).permute(2,0,1)[None].to('cuda'))
            baseline=pred.baseline[0].permute(1,2,0).cpu().numpy();value=pred.acescg[0].permute(1,2,0).cpu().numpy()
            w=pred.reconstruction_weight[0,0].cpu().numpy();high=(w>0)&(rgb.max(-1)>.5);full=high&(rgb.min(-1)>=254.5/255)
            corrected,diag=constrain_colour(rgb,value,baseline,w,view)
            variations={'raw':value,'constrained':corrected,
                'constrained_partial_only':np.where(full[...,None],baseline,corrected),
                'ordinary_shadow_only':np.where(high[...,None],baseline,corrected),
                'highlights_only':np.where(high[...,None],corrected,baseline),
                'fully_clipped_only':np.where(full[...,None],corrected,baseline)}
            if args.feather:
                from training.aces_reliable_highlights import feather_full
                variations['feather8']=feather_full(rgb,value,baseline,w,view,8)[0]
            basecells=cell_luminance(baseline)
            for variant,values in variations.items():methods.setdefault(name+'_'+variant,[]).append((cell_luminance(values),basecells))
            stats.append({'model':name,'frame':39+i,'partial_pixels':int((high&~full).sum()),'fully_clipped_pixels':int(full.sum()),'constraint':diag})
        del model
    changes={name:temporal_change(pair[0][0],pair[1][0],pair[0][1],pair[1][1]) for name,pair in methods.items()}
    result={'changes':changes,'frames':stats,'input_sha256':source['sha256'],'tool_sha256':sha(__file__),
        'scope':'Selected failure at frames39->40, causal branch substitutions; diagnostic only, not a registered pass or independent holdout'}
    out.write_text(json.dumps(result,indent=2));print(json.dumps(changes,indent=2))


if __name__=='__main__':main()
