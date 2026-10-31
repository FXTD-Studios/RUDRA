"""Compare all preregistered cyclic TRAIN retention folds, with no fold selection."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.summarize_aces_generated import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    reg=json.loads(a.registration.read_text());paths=[Path(reg['base_report'])]+[Path('G:/rudra_training')/('aces_anchor_rotation_fold'+str(f)+'_20261024')/'report.json' for f in (1,2)]
    if sha(paths[0])!=reg['base_report_sha256']:raise ValueError('base report changed')
    folds=[];monitor_scenes=[];base_selection=None;initials=None
    for fold,path in enumerate(paths):
        report=json.loads(path.read_text());rrpath=path.parent/'registration.json';rr=json.loads(rrpath.read_text())
        if report['registration_sha256']!=sha(rrpath):raise ValueError('report registration identity')
        if fold and sha(rrpath)!=sha(reg['folds'][fold-1]):raise ValueError('registered fold identity')
        if fold==0:base_selection=rr['selection'];initials=report['runs']['current_only']['initial']
        roles=['update','anchor','monitor']
        for i,row in enumerate(rr['selection']):
            expected=dict(base_selection[i]);expected['role']=roles[(roles.index(expected['role'])+fold)%3]
            if row!=expected:raise ValueError('not exact cyclic rotation')
        ids=[str(i) for i,row in enumerate(rr['selection']) if row['role']=='monitor'];monitor_scenes.extend(rr['selection'][int(i)]['scene'] for i in ids)
        result={'fold':fold,'report_sha256':sha(path),'methods':{}}
        for method in ('current_only','anchor_direction'):
            run=report['runs'][method]
            for member,initial in run['initial'].items():
                for metric in ('photo','detail','temporal','log_detail'):
                    before=initials[member][metric];now=initial[metric]
                    if before is None or now is None:
                        if before is not now:raise ValueError('initial eligibility changed')
                    elif abs(now-before)>1e-6+1e-5*abs(before):raise ValueError('initial values changed')
            rows=[]
            for i in ids:
                rows.append({'scene':rr['selection'][int(i)]['scene'],'domain':rr['selection'][int(i)]['domain'],'initial':run['initial'][i],'final':run['final'][i]})
            result['methods'][method]={'accepted':run['accepted'],'attempts':run['attempts'],'monitor_rows':rows,'seconds':run['seconds']}
        folds.append(result)
    if len(monitor_scenes)!=18 or len(set(monitor_scenes))!=18:raise ValueError('monitor coverage')
    summary={}
    for method in ('current_only','anchor_direction'):
        rr=[r for fold in folds for r in fold['methods'][method]['monitor_rows']];stats={}
        for metric in ('photo','detail','temporal','log_detail'):
            pairs=[(r['initial'][metric],r['final'][metric]) for r in rr if r['initial'][metric] is not None and r['final'][metric] is not None]
            stats[metric]={'members':len(pairs),'initial_mean':float(np.mean([b for b,f in pairs])),'final_mean':float(np.mean([f for b,f in pairs])),
                           'worse_members':sum(f>b+1e-9+1e-6*abs(b) for b,f in pairs)}
        summary[method]={'accepted':sum(f['methods'][method]['accepted'] for f in folds),'attempts':sum(f['methods'][method]['attempts'] for f in folds),'monitor':stats}
    a.out.parent.mkdir(exist_ok=True,parents=True)
    a.out.write_text(json.dumps({'folds':folds,'summary':summary,'registration_sha256':sha(a.registration),'tool_sha256':sha(__file__),'scope':'All18 scenes monitored once across three independently reset TRAIN role rotations. Reused TRAIN, not independent generalization or product qualification. No inference weights saved.'},indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
