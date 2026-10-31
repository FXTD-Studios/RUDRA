"""Original continuous motion/texture patterns for constrained dequantization."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_curvature_dequant import dequantize
from training.aces_model import model_from_checkpoint
from training.aces_reference import reference_view
from tools.summarize_aces_generated import sha
from tools.evaluate_aces_temporal import cell_luminance,temporal_change
from tools.audit_aces_learned_detail import common_detail


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_curvature_motion_20261024.json'));parser.add_argument('--out',type=Path,default=Path('G:/rudra_training/aces_curvature_dequant_20261024/motion.json'))
    args=parser.parse_args();regpath=args.registration;reg=json.loads(regpath.read_text());out=args.out
    method=dequantize
    if reg.get('coherent_guard'):
        from training.aces_coherent_dequant import coherent_dequantize
        method=coherent_dequantize
    if out.exists() or sha(reg['checkpoint'])!=reg['checkpoint_sha256']:raise ValueError('identity/new output')
    torch.set_num_threads(2);view=reference_view();view.set_threads(4)
    model=model_from_checkpoint(torch.load(reg['checkpoint'],map_location='cpu',weights_only=True)).to('cuda').eval()
    yy,xx=np.mgrid[:reg['shape'][0],:reg['shape'][1]].astype(np.float32);rows=[];checks={};summary={}
    for case in reg['cases']:
        previous=None;cr=[]
        for frame in range(reg['frames']):
            x=xx-frame*.5;y=yy-frame*.25
            if case=='bright_linear':v=230+23*x/256+1*y/64
            elif case=='radial':v=230+22*np.exp(-((x-128)**2+(y-32)**2)/4000)
            elif case=='low_contrast_texture':v=140+5*x/256+.8*np.sin(x*1.3)+.7*np.cos(y*1.1)
            else:v=np.where(x<128,80+4*x/256,180+4*x/256)
            truth=np.stack((v,v*.998,v*.996),axis=-1).astype(np.float32)/255
            source=np.round(truth*255)/255;reference=view.inverse(truth)
            with torch.inference_mode():value=model(torch.from_numpy(source).permute(2,0,1)[None].to('cuda'))
            learned=value.acescg[0].permute(1,2,0).cpu().numpy();smooth,allowed=method(source,reg['iterations'])
            hybrid=learned.copy();hybrid[allowed]=view.inverse(smooth)[allowed]
            candidates={'analytic':view.inverse(source),'learned':learned,'curvature':hybrid}
            mask=np.zeros(source.shape[:2],bool);mask[8:-8,8:-8]=True
            detail=common_detail(candidates,reference,mask)
            code={k:float(np.mean(((view.forward(v)[mask]-truth[mask])*255)**2)) for k,v in candidates.items()}
            grades={str(ev):{k:float(np.mean(((view.forward(v*2.**ev)[mask]-view.forward(reference*2.**ev)[mask])*255)**2)) for k,v in candidates.items()} for ev in (-3,0,3)}
            current={k:cell_luminance(v[8:-8,8:-8]) for k,v in {**candidates,'reference':reference}.items()}
            temporal=None if previous is None else {k:temporal_change(previous[k],current[k],previous['reference'],current['reference'])['mean_abs_excess_ev'] for k in candidates}
            previous=current
            row={'case':case,'frame':frame,'codes':code,'grades':grades,'detail':{k:detail['methods'][k]['rmse_ev'] for k in candidates},'temporal':temporal,
                 'max_delta_codes':float(np.max(np.abs(smooth-source))*255),'guarded_exact':bool(np.array_equal(hybrid[~allowed],learned[~allowed]))}
            cr.append(row);rows.append(row)
        summary[case]={kind:{k:float(np.mean([r[kind][k] for r in cr if r[kind] is not None])) for k in candidates} for kind in ('codes','detail','temporal')}
        summary[case]['grades']={ev:{k:float(np.mean([r['grades'][ev][k] for r in cr])) for k in candidates} for ev in ('-3','0','3')}
        for kind in ('codes','detail','temporal'):checks[case+'_'+kind]=summary[case][kind]['curvature']<=summary[case][kind]['learned']
        for ev in ('-3','0','3'):checks[case+'_grade_'+ev]=summary[case]['grades'][ev]['curvature']<=summary[case]['grades'][ev]['learned']
        print(case,summary[case],flush=True)
    checks['bounded']=all(r['max_delta_codes']<=.49002 for r in rows);checks['guarded_exact']=all(r['guarded_exact'] for r in rows)
    out.write_text(json.dumps({'summary':summary,'checks':checks,'numerical_pass':all(checks.values()),'rows':rows,'registration_sha256':sha(regpath),
        'tool_sha256':sha(__file__),'method_sha256':sha('training/aces_curvature_dequant.py'),'guard_sha256':sha('training/aces_coherent_dequant.py') if reg.get('coherent_guard') else None,'scope':reg['checks'],'release_authorized':False},indent=2,allow_nan=False))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
