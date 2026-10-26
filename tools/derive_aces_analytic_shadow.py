"""Derive the v1 analytic-shadow candidate from a trained shared-shadow ACES checkpoint.

No weights change and nothing is trained: the predictor and restorer are copied
bit-exactly and only the shadow policy in the contract changes.
"""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import ACESModel,model_from_checkpoint
from tools.evaluate_run05_dequant import sha


def derive(state):
    parent=model_from_checkpoint(state)
    if parent.shadow!='shared':raise ValueError('parent must use the shared shadow path')
    child=ACESModel(state['model']['predictor.stem.weight'].shape[0],parent.normalization,parent.restoration,'analytic')
    child.load_state_dict(state['model'],strict=True)
    out={k:v for k,v in state.items() if k not in ('model','contract','optimizer')}
    out.update(model=child.state_dict(),contract=child.contract(),research_only=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    state=torch.load(a.parent,map_location='cpu',weights_only=True);child=derive(state)
    child['derivation']={'parent':str(a.parent),'parent_sha256':sha(a.parent),'weights_bit_exact':True,'trained':False,
                         'change':'shadow policy shared -> analytic','release_authorized':False}
    a.out.parent.mkdir(parents=True,exist_ok=True);torch.save(child,a.out)
    loaded=model_from_checkpoint(torch.load(a.out,map_location='cpu',weights_only=True))
    if any(not torch.equal(v,state['model'][k]) for k,v in loaded.state_dict().items()):raise ValueError('weights changed')
    (a.out.parent/'derivation.json').write_text(json.dumps({**child['derivation'],'output':str(a.out),'output_sha256':sha(a.out),'contract':child['contract']},indent=2))
    print(json.dumps(child['derivation'],indent=2))


if __name__=='__main__':main()
