"""Original constant-colour patches; displayed fidelity, not HDR-detail truth."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.rudra_aces_export import load_model,infer,contact_sheet,sha
from training.aces_reference import reference_view


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--colour-backtrack',action='store_true');a=p.parse_args()
    a.out.mkdir(exist_ok=False);torch.set_num_threads(2);model,state=load_model(a.checkpoint,'cpu');view=reference_view()
    colours={'red':[1,0,0],'green':[0,1,0],'blue':[0,0,1],'cyan':[0,1,1],'magenta':[1,0,1],'yellow':[1,1,0],'white':[1,1,1],'black':[0,0,0]};rows=[]
    for name,c in colours.items():
        source=np.broadcast_to(np.array(c,np.float32),(96,128,3)).copy()
        value,mask,stats,base=infer(model,source,'cpu')
        if a.colour_backtrack:
            from training.aces_model import reconstruction_weight
            from training.aces_colour_backtrack import constrain_colour
            weight=reconstruction_weight(torch.from_numpy(source).permute(2,0,1)[None])[0,0].numpy()
            value,diagnostic=constrain_colour(source,value,base,weight,view)
            stats={'unconstrained_inference_stats':stats,'constraint':diagnostic}
        stats.update(name=name,source_rgb=c,centre_prediction=value[48,64].tolist(),centre_baseline=base[48,64].tolist(),
            centre_display_error_codes=((view.forward(value)-source)[48,64]*255).tolist(),
            negative3ev_model_rgb=view.forward(value/8)[48,64].tolist(),negative3ev_analytic_rgb=view.forward(base/8)[48,64].tolist())
        rows.append(stats);contact_sheet(a.out/(name+'.png'),source,value,base,mask)
    report={'checkpoint_sha256':sha(a.checkpoint),'contract':state['contract'],'auditor_sha256':sha(__file__),'rows':rows,
            'colour_backtrack':a.colour_backtrack,
            'scope':'constant-colour stress diagnostics previously examined in research; not HDR detail ground truth or independent release holdout'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({r['name']:r['centre_display_error_codes'] for r in rows},indent=2))


if __name__=='__main__':main()
