"""Evaluate frozen candidates on the fresh FXTD grading set (claude_fresh_fxtd_grade_20261026.json)."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint
from training.aces_reference import reference_view
from training.aces_clipped_smoothing import smooth_clipped
from training.aces_colour_backtrack import constrain_colour

LUMA=np.array([.2722287168,.6740817658,.0536895174])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus',type=Path,required=True);ap.add_argument('--registration',type=Path,required=True)
    ap.add_argument('--candidate',action='append',required=True,help='name=checkpoint');ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise ValueError('new output required')
    reg=json.loads(a.registration.read_text());corpus=json.loads((a.corpus/'report.json').read_text())
    if corpus['registration_sha256']!=sha(a.registration):raise ValueError('registration changed after build')
    torch.set_num_threads(2);view=reference_view();view.set_threads(4)
    models={}
    for spec in a.candidate:
        n,p=spec.split('=',1);st=torch.load(p,map_location='cpu',weights_only=True);models[n]=(model_from_checkpoint(st).eval(),sha(p))
    rows=[]
    for c in [r for r in corpus['rows'] if r['kept']]:
        if sha(c['source_path'])!=c['source_sha256'] or sha(c['reference_path'])!=c['reference_sha256']:raise ValueError('case identity')
        src=np.load(c['source_path']);ref=np.load(c['reference_path']).astype(np.float64)
        x=torch.from_numpy(src).permute(2,0,1)[None]
        preds={}
        for n,(m,_) in models.items():
            with torch.inference_mode():v=m(x)
            raw=v.acescg[0].permute(1,2,0).numpy().copy();base=v.baseline[0].permute(1,2,0).numpy();w=v.reconstruction_weight[0,0].numpy()
            sm,_=smooth_clipped(raw,src,sigma=64,mode='hybrid',ramp=1,start=0)
            out,diag=constrain_colour(src,sm,base,w,view,include_fully_clipped=True,forbid_new_negative_y=True)
            preds[n]=(out,diag)
        analytic=base.astype(np.float64)  # exact ACES 2.0 inverse (stage 1), same for all candidates
        high=(w>0)&(src.max(-1)>.5);full=high&(src.min(-1)>=254.5/255);part=high&~full
        row={'case_id':c['case_id'],'scene_id':c['scene_id'],'project':c['project'],'clip_target':c['clip_target'],
             'highlight':int(high.sum()),'partial':int(part.sum()),'full':int(full.sum()),'methods':{}}
        cands={'analytic':(analytic,None),**{n:(p.astype(np.float64),d) for n,(p,d) in preds.items()}}
        for n,(arr,d) in cands.items():
            m={}
            for ev in (-3,0,3):
                t=view.forward((ref*2.**ev).astype(np.float32));sh=view.forward((arr*2.**ev).astype(np.float32))
                e=((sh-t)*255.)**2
                m[f'grade_{ev}']=float(e[high].mean());m[f'grade_{ev}_partial']=float(e[part].mean()) if part.any() else None
            m['H']=float((np.mean((np.arcsinh(arr)-np.arcsinh(ref))**2,-1))[high].mean())
            m['new_negative_y']=int((((arr@LUMA)<0)&((analytic@LUMA)>=0)).sum())
            if d is not None:m['remaining_display_errors']=d['remaining_display_errors']
            row['methods'][n]=m
        rows.append(row);print(len(rows),c['case_id'],{n:round(v['grade_3'],5) for n,v in row['methods'].items()},flush=True)
    names=list(rows[0]['methods'])
    mean=lambda n,k:float(np.mean([r['methods'][n][k] for r in rows]))
    summary={n:{k:mean(n,k) for k in ('grade_-3','grade_0','grade_3','H')} for n in names}
    for n in names[1:]:
        summary[n]['new_negative_y']=sum(r['methods'][n]['new_negative_y'] for r in rows)
        summary[n]['remaining_display_errors']=sum(r['methods'][n]['remaining_display_errors'] for r in rows)
    # bootstrap over scenes
    rng=np.random.default_rng(20261026);scenes=sorted({r['scene_id'] for r in rows});idx={s:[i for i,r in enumerate(rows) if r['scene_id']==s] for s in scenes}
    ci={}
    for n in names[1:]:
        ci[n]={}
        for k in ('grade_-3','grade_0','grade_3','H'):
            d=np.array([r['methods'][n][k]-r['methods']['analytic'][k] for r in rows])
            bs=[]
            for _ in range(10000):
                pick=np.concatenate([idx[s] for s in rng.choice(scenes,len(scenes))]);bs.append(d[pick].mean())
            ci[n][k]={'mean_diff':float(d.mean()),'ci95':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],'cases_better':int((d<0).sum()),'cases_worse':int((d>0).sum())}
    projects={}
    for p in sorted({r['project'] for r in rows}):
        rr=[r for r in rows if r['project']==p]
        projects[p]={n:{k:float(np.mean([r['methods'][n][k] for r in rr])) for k in ('grade_-3','grade_0','grade_3','H')} for n in names};projects[p]['cases']=len(rr)
    primary=names[1]
    gates={'grade_-3':summary[primary]['grade_-3']<=summary['analytic']['grade_-3'],
           'grade_0':summary[primary]['grade_0']<=summary['analytic']['grade_0'],
           'grade_+3':summary[primary]['grade_3']<=summary['analytic']['grade_3'],
           'highlight_H':summary[primary]['H']<=summary['analytic']['H'],
           'no_new_negativeY':summary[primary]['new_negative_y']==0,
           'appearance':summary[primary]['remaining_display_errors']==0}
    result={'primary':primary,'gates':gates,'numerical_pass':all(gates.values()),'summary':summary,'bootstrap_vs_analytic':ci,'by_project':projects,
            'cases':len(rows),'scenes':len(scenes),'candidates':{n:s for n,(m,s) in models.items()},'rows':rows,
            'registration_sha256':sha(a.registration),'corpus_report_sha256':sha(a.corpus/'report.json'),'evaluator_sha256':sha(__file__),
            'scope':reg['limits'],'release_authorized':False}
    a.out.write_text(json.dumps(result,indent=1));print(json.dumps({k:result[k] for k in ('gates','summary','bootstrap_vs_analytic','by_project')},indent=1))


if __name__=='__main__':main()
