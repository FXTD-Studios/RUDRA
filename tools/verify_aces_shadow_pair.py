"""Verify checkpoint/data/sampler matching for the registered shadow-scale pair."""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('control','treatment','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new report required')
    c=torch.load(a.control,map_location='cpu',weights_only=True);t=torch.load(a.treatment,map_location='cpu',weights_only=True)
    cr=c['recipe'].copy();tr=t['recipe'].copy()
    cw=cr.pop('shadow_weight');tw=tr.pop('shadow_weight')
    checks={'recipe_except_weight':cr==tr,'weights_registered':cw==0 and tw==1,
            'steps300':c['step']==t['step']==300,'contract':c['contract']==t['contract'],
            'data':c['pilot_sha256']==t['pilot_sha256'],
            'numpy_sampler_rng':c['numpy_rng']==t['numpy_rng'],
            'torch_rng':torch.equal(c['torch_rng'],t['torch_rng']),
            'cuda_rng':len(c['cuda_rng'])==len(t['cuda_rng']) and all(torch.equal(x,y) for x,y in zip(c['cuda_rng'],t['cuda_rng'])),
            'finite_weights':all(torch.isfinite(x).all().item() for state in (c,t) for x in state['model'].values())}
    result={'checks':checks,'pass':all(checks.values()),'control_sha256':sha(a.control),'treatment_sha256':sha(a.treatment),
            'scope':'matched recipe/code/data/RNG, CUDA numerical nondeterminism still possible'}
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['pass']:raise SystemExit(1)


if __name__=='__main__':main()
