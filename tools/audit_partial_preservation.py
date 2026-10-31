"""Independent partial-route preservation audit; does not replace quality gates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint
from training.aces_reference import reference_view
from training.aces_clipped_smoothing import smooth_clipped
from training.aces_colour_backtrack import constrain_colour
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--pilot',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    states=[torch.load(path,map_location='cpu',weights_only=True) for path in (a.parent,a.candidate)]
    if states[0]['contract']!=states[1]['contract']:raise ValueError('contract mismatch')
    if states[0]['model'].keys()!=states[1]['model'].keys():raise ValueError('state keys')
    changed=[k for k in states[0]['model'] if not torch.equal(states[0]['model'][k],states[1]['model'][k])]
    frozen=all(k.startswith('highlight_context.') for k in changed)
    torch.set_num_threads(2);view=reference_view();view.set_threads(4)
    models=[model_from_checkpoint(s).eval().cuda() for s in states]
    rows=[]
    for row in json.loads(Path(a.pilot).read_text())['rows']:
        if sha(row['sdr_path'])!=row['sdr_sha256']:raise ValueError('source identity')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False))
        x=torch.from_numpy(source).permute(2,0,1)[None].cuda()
        outputs=[]
        for model in models:
            with torch.inference_mode():value=model(x)
            array=lambda t:t[0].permute(1,2,0).cpu().numpy()
            raw=array(value.acescg);base=array(value.baseline);w=value.reconstruction_weight[0,0].cpu().numpy()
            smooth,_=smooth_clipped(raw,source,sigma=64,mode='hybrid')
            constrained,_=constrain_colour(source,smooth,base,w,view,include_fully_clipped=True)
            outputs.append({'raw':raw,'smooth':smooth,'constrained':constrained})
        full=(w>0)&(source.min(-1)>=254.5/255)
        outside=~((w>0)&(source.max(-1)>.5))
        checks={stage+'_'+region:bool(np.array_equal(outputs[0][stage][mask],outputs[1][stage][mask]))
                for stage in outputs[0] for region,mask in [('full',full),('outside_highlight',outside)]}
        rows.append({'asset_id':row['asset_id'],'full_pixels':int(full.sum()),'outside_highlight_pixels':int(outside.sum()),'checks':checks})
        print('audited',len(rows),flush=True)
    checks={'frozen_weights':frozen,'partial_weights_changed':bool(changed),
            'nonempty_full_coverage':sum(r['full_pixels'] for r in rows)>0,
            'nonempty_outside_coverage':sum(r['outside_highlight_pixels'] for r in rows)>0,
            'all_stages_preserved':all(all(r['checks'].values()) for r in rows)}
    result={'checks':checks,'rows':rows,'changed_tensors':changed,
            'parent_sha256':sha(a.parent),'candidate_sha256':sha(a.candidate),'pilot_sha256':sha(a.pilot),
            'code_hashes':{f:sha(f) for f in [__file__,'training/aces_clipped_smoothing.py','training/aces_colour_backtrack.py','training/aces_model.py','training/aces_context_model.py','training/aces_full_context_model.py']},
            'scope':'Preservation on supplied development frames only; no grading tolerance adoption or model qualification.', 'release_authorized':False}
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
