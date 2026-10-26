"""Paired old/new development evaluation for the isolated highlight-head trial."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.rudra_aces_export import load_model,sha
from tools.evaluate_aces_smoke import validate_evaluation_rows
from tools.audit_aces2_sources import scores
from tools.compare_aces_development import assess
from tools.evaluate_aces_temporal import LUMA_AP1
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('G:/rudra_training/aces_consistent_head600_20261024'))
    parser.add_argument('--consistent-pilot',type=Path,default=Path('G:/datasets/corpora/aces_consistent_exposures_20261024/val/report.json'))
    parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_consistent_head_20261024.json'));args=parser.parse_args()
    root=args.root;reg_path=args.registration;reg=json.loads(reg_path.read_text())
    checkpoint=root/'step600.pt';out=root/'paired_development';out.mkdir(exist_ok=False)
    torch.set_num_threads(2);reference_view().set_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent,_=load_model(Path(reg['parent']),'cuda');candidate,state=load_model(checkpoint,'cuda')
    ancestry=json.loads((root/'training_ancestry.json').read_text())
    assert state['pilot_sha256']==sha(root/'training_ancestry.json')
    view=reference_view();results={};diagnostics={}
    for label,pilot_path in (
        ('original_natural',Path('G:/datasets/corpora/run05_aces_spatial_dev24_v2_20261023/report.json')),
        ('consistent',args.consistent_pilot)):
        pilot=json.loads(pilot_path.read_text());validate_evaluation_rows(ancestry,pilot)
        folder=out/label;folder.mkdir();rows={'parent':[],'candidate':[]};diag=[]
        for i,row in enumerate(pilot['rows']):
            assert sha(row['sdr_path'])==row['sdr_sha256'] and sha(row['target_path'])==row['target_sha256']
            source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
            x=torch.from_numpy(source).permute(2,0,1)[None].to('cuda');predictions={}
            for name,model in (('parent',parent),('candidate',candidate)):
                with torch.inference_mode():
                    value=model(x);prediction=value.acescg[0].permute(1,2,0).cpu().numpy().copy()
                    baseline=value.baseline[0].permute(1,2,0).cpu().numpy().copy()
                    weight=value.reconstruction_weight[0,0].cpu().numpy().copy()
                predictions[name]=prediction
                shadow=(weight>0)&(source.max(-1)<=.5);hi=(weight>0)&~shadow
                partial=hi&(source.min(-1)<254.5/255);loss={}
                for method,array in (('analytic',baseline),('model',prediction)):
                    error=np.mean((np.arcsinh(array)-np.arcsinh(target))**2,-1)
                    loss[method]={'all_mse':float(error.mean()),'highlight_mse':float(error[hi].mean()) if hi.any() else None,
                                  'shadow_mse':float(error[shadow].mean()) if shadow.any() else None}
                codes=view.forward(prediction);delta=(codes-source).max(-1)*255
                path=folder/f'{i:03d}_{name}.npy';np.save(path,prediction)
                rows[name].append({'asset_id':row['asset_id'],'prediction_path':str(path),'prediction_sha256':sha(path),
                    'roundtrip':scores(codes,source),'target_errors':loss,'shadow_lift_gt1code':int((delta[shadow]>1).sum()),
                    'partial_highlight_pixels':int(partial.sum()),
                    'partial_highlight_over1_pixels':int((np.abs(codes-source).max(-1)[partial]*255>1).sum()),
                    'new_negative_luminance_pixels':int((((prediction@LUMA_AP1)<0)&((baseline@LUMA_AP1)>=0)).sum())})
            diag.append({'asset_id':row['asset_id'],'outside_highlight_bit_exact':bool(np.array_equal(predictions['parent'][~hi],predictions['candidate'][~hi]))})
            print(label,i+1,flush=True)
        reports={name:{'rows':rr,'evaluation_role':'scene-disjoint development','pilot_sha256':sha(pilot_path),
                       'checkpoint_sha256':sha(checkpoint if name=='candidate' else reg['parent'])} for name,rr in rows.items()}
        for name,report in reports.items():(folder/(name+'_report.json')).write_text(json.dumps(report,indent=2))
        results[label]=reports;diagnostics[label]=diag
    old=assess(results['original_natural']['candidate']);old_parent=assess(results['original_natural']['parent'])
    new=results['consistent'];errors={}
    for name,report in new.items():
        errors[name]={method:float(np.mean([r['target_errors'][method]['highlight_mse'] for r in report['rows'] if r['target_errors'][method]['highlight_mse'] is not None])) for method in ('analytic','model')}
    rr=new['candidate']['rows'];count=sum(r['partial_highlight_pixels'] for r in rr);bad=sum(r['partial_highlight_over1_pixels'] for r in rr)
    checks={**old['checks'],'new_highlight_improvement_5pct':errors['candidate']['model']<=.95*errors['parent']['model'],
        'new_highlight_no_regression_vs_analytic':errors['candidate']['model']<=errors['candidate']['analytic'],
        'partial_highlight_appearance':count>0 and 1-bad/count>=.999,
        'outside_highlight_bit_exact':all(r['outside_highlight_bit_exact'] for d in diagnostics.values() for r in d),
        'no_new_negative_luminance':all(r['new_negative_luminance_pixels']==0 for result in results.values() for r in result['candidate']['rows'])}
    checks['frozen_weights_bit_exact']=json.loads((root/'report.json').read_text())['frozen_weights_bit_exact']
    if reg.get('control'):
        control_root=Path(reg['control']).parent/'paired_development'
        control_report=json.loads((control_root/'consistent/candidate_report.json').read_text())
        if control_report['checkpoint_sha256']!=sha(reg['control']):raise ValueError('pooled control identity mismatch')
        control_acceptance=json.loads((control_root/'acceptance.json').read_text())
        checks['no_regression_vs_pooled_control']=errors['candidate']['model']<=control_acceptance['consistent_highlight_errors']['candidate']['model']
    acceptance={'checks':checks,'numerical_pass':all(checks.values()),'original_candidate':old,'original_parent':old_parent,
        'consistent_highlight_errors':errors,'partial_highlight_within1_fraction':1-bad/count if count else None,
        'registration_sha256':sha(reg_path),'isolation':diagnostics,'release_authorized':False,
        'scope':'registered isolated head development trial; motion, colour stress, grading, detail and release remain separate'}
    (out/'acceptance.json').write_text(json.dumps(acceptance,indent=2));print(json.dumps({k:acceptance[k] for k in ('checks','numerical_pass','consistent_highlight_errors','partial_highlight_within1_fraction')},indent=2))


if __name__=='__main__':main()
