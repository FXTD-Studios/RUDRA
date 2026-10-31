"""Verify diagnostic replay identity and summarize measured TRAIN step losses."""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.summarize_aces_generated import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,default=Path('G:/rudra_training/partial_pcgrad400_20261025'))
    parser.add_argument('--replay',type=Path,default=Path('G:/rudra_training/partial_pcgrad_replay400_20261025'))
    args=parser.parse_args();original=args.original;replay=args.replay
    a=torch.load(original/'step400.pt',map_location='cpu',weights_only=True)['model']
    b=torch.load(replay/'step400.pt',map_location='cpu',weights_only=True)['model']
    checks={'final_model_tensor_identity':a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)}
    for name in ('crop_pool.json','motion_pool.json','schedule.json'):
        checks[name+'_identity']=sha(original/name)==sha(replay/name)
    old=json.loads((original/'report.json').read_text());new=json.loads((replay/'report.json').read_text())
    checks['complete_history_identity']=old['history']==new['history']
    differences={k:float((a[k]-b[k]).abs().max()) for k in a if not torch.equal(a[k],b[k])}
    steps=[1,2,101,102,201,202,399,400]
    probes=[json.loads((replay/f'step_probe_{s:04d}.json').read_text()) for s in steps]
    for s,p in zip(steps,probes):
        assert p['step']==s and p['actual_weights_restored']
        assert sha(replay/f'step_probe_{s:04d}.pt')==p['snapshot_sha256']
    methods=list(probes[0]['deltas']);terms=sorted({n for p in probes for n in p['baseline']})
    summary={}
    for method in methods:
        summary[method]={}
        for term in terms:
            rows=[p for p in probes if term in p['baseline']]
            changes=[p['deltas'][method][term] for p in rows]
            summary[method][term]={'evaluations':len(rows),'increases':sum(d>0 for d in changes),
                                   'decreases':sum(d<0 for d in changes),'sum_loss_delta':sum(changes),
                                   'sum_baseline_loss':sum(p['baseline'][term] for p in rows)}
    result={'checks':checks,'changed_tensor_max_abs':differences,'summary':summary,'probes':probes,
            'original_checkpoint_sha256':sha(original/'step400.pt'),'replay_checkpoint_sha256':sha(replay/'step400.pt'),
            'scope':'Eight preregistered TRAIN updates; measured weighted losses. Exact replay only if all identity checks pass. No DEV result or generalization claim.'}
    (replay/'step_probe_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'checks':checks,'summary':summary},indent=2))
    if not all(checks.values()):raise SystemExit('Replay identity FAILED; measurements describe this diagnostic run only.')


if __name__=='__main__':main()
