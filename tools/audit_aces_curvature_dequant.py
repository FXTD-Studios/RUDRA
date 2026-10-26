"""Registered crop audit of bounded source-code dequantization; no model promotion."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_curvature_dequant import dequantize
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_curvature_dequant_20261024.json'));parser.add_argument('--out',type=Path,default=Path('G:/rudra_training/aces_curvature_dequant_20261024'))
    args=parser.parse_args();regpath=args.registration;reg=json.loads(regpath.read_text());root=args.out;root.mkdir(exist_ok=False)
    (root/'auditor_snapshot.py').write_bytes(Path(__file__).read_bytes())
    method=dequantize
    if reg.get('coherent_guard'):
        from training.aces_coherent_dequant import coherent_dequantize
        method=coherent_dequantize
    view=reference_view();view.set_threads(4);rows=[]
    for label,paths in reg['datasets'].items():
        pilot=json.loads(Path(paths['pilot']).read_text());report=json.loads(Path(paths['predictions']).read_text())
        if report['pilot_sha256']!=sha(paths['pilot']):raise ValueError('pilot identity')
        predictions={r['asset_id']:r for r in report['rows']}
        for r in pilot['rows']:
            if label=='new_motion' and r['frame']!=0:continue
            if r['split']!='val':raise ValueError('DEV only')
            for k in ('sdr','reference'):
                if sha(r[k+'_path'])!=r[k+'_sha256']:raise ValueError('source identity')
            p=predictions[r['asset_id']]
            if sha(p['prediction_path'])!=p['prediction_sha256']:raise ValueError('prediction identity')
            source=canonical_to_source(load_rgb(r['sdr_path'],hdr=False));h,w=source.shape[:2]
            cy=(h-min(h,320))//2;cx=(w-min(w,512))//2;crop=np.s_[cy:cy+320,cx:cx+512]
            source=source[crop].copy();reference=np.load(r['reference_path'])[crop];learned=np.load(p['prediction_path'])[crop]
            smooth,allowed=method(source,reg['iterations']);ordinary=(source.max(-1)<254.5/255)&(source.max(-1)>3/255)
            valid=ordinary.copy();valid[:32]=False;valid[-32:]=False;valid[:,:32]=False;valid[:,-32:]=False
            # Preserve learned output wherever the source-defined dequantizer is guarded.
            hybrid=learned.copy();hybrid[allowed]=view.inverse(smooth)[allowed]
            candidates={'analytic':view.inverse(source),'learned':learned,'curvature':hybrid}
            truth=view.forward(reference);code={k:float(np.mean(((view.forward(v)[valid]-truth[valid])*255)**2)) if valid.any() else None for k,v in candidates.items()}
            grades={str(ev):{k:float(np.mean(((view.forward(v*2.**ev)[valid]-view.forward(reference*2.**ev)[valid])*255)**2)) if valid.any() else None for k,v in candidates.items()} for ev in (-3,0,3)}
            rows.append({'dataset':label,'asset_id':r['asset_id'],'code_mse':code,'grade_mse':grades,'allowed_pixels':int((allowed&valid).sum()),'ordinary_pixels':int(valid.sum()),
                         'max_delta_codes':float(np.abs(smooth-source).max()*255),'guarded_exact':bool(np.array_equal(hybrid[~allowed],learned[~allowed]))})
            print('audited',label,len(rows),flush=True)
    summary={};checks={'bounded':all(r['max_delta_codes']<=.49002 for r in rows),'guarded_exact':all(r['guarded_exact'] for r in rows)}
    for label in reg['datasets']:
        selected=[r for r in rows if r['dataset']==label and r['code_mse']['curvature'] is not None]
        summary[label]={'codes':{k:float(np.mean([r['code_mse'][k] for r in selected])) for k in candidates},
                        'grades':{str(ev):{k:float(np.mean([r['grade_mse'][str(ev)][k] for r in selected])) for k in candidates} for ev in (-3,0,3)}}
        checks[label+'_codes']=summary[label]['codes']['curvature']<=summary[label]['codes']['learned']
        for ev in ('-3','0','3'):checks[label+'_grade_'+ev]=summary[label]['grades'][ev]['curvature']<=summary[label]['grades'][ev]['learned']
    result={'summary':summary,'checks':checks,'numerical_pass':all(checks.values()),'rows':rows,'registration_sha256':sha(regpath),'tool_sha256':sha(__file__),
            'method_sha256':sha('training/aces_curvature_dequant.py'),'guard_sha256':sha('training/aces_coherent_dequant.py') if reg.get('coherent_guard') else None,'input_hashes':{p:sha(p) for v in reg['datasets'].values() for p in v.values()},'scope':reg['selection'],'release_authorized':False}
    (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'summary':summary,'checks':checks},indent=2))


if __name__=='__main__':main()
