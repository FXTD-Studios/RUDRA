"""Summarize a completed registered retention pilot without changing its gates."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.summarize_aces_generated import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    reg=json.loads((a.root/'registration.json').read_text());report=json.loads((a.root/'report.json').read_text())
    if report['registration_sha256']!=sha(a.root/'registration.json'):raise ValueError('registration identity')
    summary={}
    for method,run in report['runs'].items():
        result={k:run[k] for k in ('accepted','attempts','seconds','evaluation_batches')}
        result['anchor_gradient_batches']=run.get('anchor_gradient_batches',0)
        result['rejected_panel_objectives']=dict(Counter(key for step in run['steps'] for trial in step['trials'] for key in trial.get('panel_failures',[])))
        projections=[step['projection'] for step in run['steps'] if step.get('projection')]
        if projections:
            result['projection_norm_ratio']={'min':min(x['norm_ratio'] for x in projections),'max':max(x['norm_ratio'] for x in projections)}
        result['local_pass_panel_rejected']=sum(t['current_pass'] and t['panel_pass'] is False for s in run['steps'] for t in s['trials'])
        result['rejected_steps_rollback_verified']=all(s['rollback_verified'] for s in run['steps'] if not s['accepted'])
        result['by_role']={}
        for role in ('update','anchor','monitor'):
            ids=[str(i) for i,s in enumerate(reg['selection']) if s['role']==role];metrics={}
            for metric in ('photo','detail','temporal','log_detail'):
                pairs=[(run['initial'][i][metric],run['final'][i][metric]) for i in ids if run['initial'][i][metric] is not None and run['final'][i][metric] is not None]
                metrics[metric]={'members':len(pairs),'initial_mean':float(np.mean([b for b,f in pairs])), 'final_mean':float(np.mean([f for b,f in pairs])),
                                 'worse_members':sum(f>b+reg['atol']+reg['rtol']*abs(b) for b,f in pairs)}
            for region in ('full_boundary','full_near','full_deep'):
                pairs=[(run['initial'][i]['regions'][region]['gradient_mse'],run['final'][i]['regions'][region]['gradient_mse']) for i in ids]
                pairs=[(b,f) for b,f in pairs if b is not None and f is not None]
                metrics[region+'_gradient']={'members':len(pairs),'worse_members':sum(f>b+reg['atol']+reg['rtol']*abs(b) for b,f in pairs)}
            result['by_role'][role]=metrics
        summary[method]=result
    out=a.root/'summary.json'
    if out.exists():raise ValueError('new output required')
    out.write_text(json.dumps({'summary':summary,'report_sha256':sha(a.root/'report.json'),'tool_sha256':sha(__file__),'scope':'Equal panel-member means; TRAIN-only, monitor unused for acceptance. No inference checkpoint saved.'},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
