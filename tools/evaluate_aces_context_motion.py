"""Paired v8 development audit, retaining raw/constrained and historical scores."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_erosion
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_context_model import ACESContextModel,aligned_pair_residual_loss
from training.aces_model import model_from_checkpoint
from training.aces_full_context_model import fully_clipped_support
from training.aces_reference import reference_view
from training.aces_colour_backtrack import constrain_colour
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.summarize_aces_generated import sha
from tools.evaluate_aces_smoke import validate_evaluation_rows
from tools.compare_aces_development import assess
from tools.audit_aces2_sources import scores
from tools.audit_aces_learned_detail import common_detail
from tools.evaluate_aces_temporal import LUMA_AP1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,default=Path('configs/experiments/aces_context_motion600_20261024.json'))
    parser.add_argument('--root',type=Path,default=Path('G:/rudra_training/aces_context_motion600_20261024'))
    args=parser.parse_args();regpath=args.registration;reg=json.loads(regpath.read_text())
    root=args.root;out=root/'paired_development';out.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    view=reference_view();view.set_threads(4)
    ckpt=root/f'step{reg["steps"]}.pt';state=torch.load(ckpt,map_location='cpu',weights_only=True)
    ancestry_path=root/'training_ancestry.json';ancestry=json.loads(ancestry_path.read_text())
    if state['pilot_sha256']!=sha(ancestry_path):raise ValueError('ancestry identity')
    model=model_from_checkpoint(state).to('cuda').eval()
    parent=model_from_checkpoint(torch.load(reg['parent'],map_location='cpu',weights_only=True)).to('cuda').eval()
    if sha(reg['parent'])!=reg['parent_sha256']:raise ValueError('parent identity')
    all_reports={};isolation=[];full_isolation=[];detail_rows=[];motion=[]
    for label,pilotpath in (
        ('new_motion',Path('G:/datasets/corpora/aces_temporal_context_pairs_20261024/val/report.json')),
        ('old_coherent',Path('G:/datasets/corpora/aces_consistent_exposures_v2_20261024/val/report.json')),
        ('legacy_natural',Path('G:/datasets/corpora/run05_aces_spatial_dev24_v2_20261023/report.json'))):
        pilot=json.loads(pilotpath.read_text());validate_evaluation_rows(ancestry,pilot)
        folder=out/label;folder.mkdir();rows={k:[] for k in ('parent','candidate_raw','candidate')};previous=None
        for i,row in enumerate(pilot['rows']):
            for k in ('sdr','target'):
                if sha(row[k+'_path'])!=row[k+'_sha256']:raise ValueError('input identity')
            source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
            x=torch.from_numpy(source).permute(2,0,1)[None].to('cuda');predictions={}
            for name,network in (('parent',parent),('candidate_raw',model)):
                with torch.inference_mode():value=network(x)
                predictions[name]=value.acescg[0].permute(1,2,0).cpu().numpy().copy()
            baseline=value.baseline[0].permute(1,2,0).cpu().numpy();w=value.reconstruction_weight[0,0].cpu().numpy()
            constrained_input=predictions['candidate_raw']
            if reg.get('clipped_smoothing'):
                from training.aces_clipped_smoothing import smooth_clipped
                constrained_input,_=smooth_clipped(constrained_input,source,**reg['clipped_smoothing'])
            predictions['candidate'],diagnostic=constrain_colour(source,constrained_input,baseline,w,view,include_fully_clipped=reg.get('include_fully_clipped',False),forbid_new_negative_y=reg.get('forbid_new_negative_y',False))
            high=(w>0)&(source.max(-1)>.5);shadow=(w>0)&~high
            isolation.append(bool(np.array_equal(predictions['parent'][~high],predictions['candidate_raw'][~high])))
            if model.contract().get('full_context'):
                full=fully_clipped_support(x,value.reconstruction_weight)[0,0].cpu().numpy()
                full_isolation.append(bool(np.array_equal(predictions['parent'][~full],predictions['candidate_raw'][~full])))
            for name,rgb in predictions.items():
                if not np.isfinite(rgb).all():raise ValueError('nonfinite output')
                losses={}
                for method,array in (('analytic',baseline),('model',rgb)):
                    error=np.mean((np.arcsinh(array)-np.arcsinh(target))**2,-1)
                    losses[method]={'all_mse':float(error.mean()),'highlight_mse':float(error[high].mean()) if high.any() else None,
                                    'shadow_mse':float(error[shadow].mean()) if shadow.any() else None}
                shown=view.forward(rgb);path=folder/f'{i:03d}_{name}.npy';np.save(path,rgb)
                rows[name].append({'asset_id':row['asset_id'],'prediction_path':str(path),'prediction_sha256':sha(path),
                    'roundtrip':scores(shown,source),'target_errors':losses,
                    'shadow_lift_gt1code':int((((shown-source).max(-1)*255>1)&shadow).sum()),
                    'new_negative_luminance':int((((rgb@LUMA_AP1)<0)&((baseline@LUMA_AP1)>=0)).sum()),
                    'constraint':diagnostic if name=='candidate' else None})
            if label=='new_motion':
                if sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('reference identity')
                reference=np.load(row['reference_path'])
                candidates={'analytic':baseline,**predictions}
                detail_rows.append({'asset_id':row['asset_id'],'highlight':common_detail(candidates,reference,binary_erosion(high,iterations=2)),
                    'fully_clipped':common_detail(candidates,reference,binary_erosion(source.min(-1)>=254.5/255,iterations=2))})
                current={'group':row['group_id'],'predictions':predictions,'target':target,'high':high}
                if previous is not None and previous['group']==current['group']:
                    masks=torch.from_numpy(np.stack([previous['high'],high])[:,None])
                    truth=torch.from_numpy(np.stack([previous['target'],target])).permute(0,3,1,2)
                    errors={}
                    for name in predictions:
                        pair=torch.from_numpy(np.stack([previous['predictions'][name],predictions[name]])).permute(0,3,1,2)
                        loss,count=aligned_pair_residual_loss(pair,truth,masks,return_count=True)
                        errors[name]=float(loss) if count>0 else None
                    motion.append({'group':row['group_id'],'frame':row['frame'],'valid_pixels':int(count),'aligned_residual_mse':errors})
                previous=current
            print(label,i+1,flush=True)
        reports={name:{'rows':rr,'evaluation_role':'scene-disjoint development','pilot_sha256':sha(pilotpath),
                    'checkpoint_sha256':sha(reg['parent'] if name=='parent' else ckpt),'colour_backtrack':name=='candidate',
                    'include_fully_clipped':name=='candidate' and reg.get('include_fully_clipped',False),
                    'clipped_smoothing':reg.get('clipped_smoothing') if name=='candidate' else None} for name,rr in rows.items()}
        for name,report in reports.items():(folder/(name+'_report.json')).write_text(json.dumps(report,indent=2))
        all_reports[label]=reports
    def high_error(report):
        values=[r['target_errors']['model']['highlight_mse'] for r in report['rows'] if r['target_errors']['model']['highlight_mse'] is not None]
        return float(np.mean(values))
    primary=all_reports['new_motion'];new=assess(primary['candidate']);old=all_reports['old_coherent']
    checks={'new_H_improvement_5pct':high_error(primary['candidate'])<=.95*high_error(primary['parent']),
        'new_H_no_worse_analytic':new['errors']['highlight']['model']<=new['errors']['highlight']['analytic'],
        'old_coherent_H_preserved':high_error(old['candidate'])<=reg.get('old_coherent_control_max',high_error(old['parent'])),
        'outside_highlight_exact':all(isolation),'frozen_parent':json.loads((root/'report.json').read_text())['frozen_parent_bit_exact'],
        'partial_colour':all(r['constraint']['remaining_display_errors']==0 for reports in all_reports.values() for r in reports['candidate']['rows']),
        'no_new_negativeY':all(r['new_negative_luminance']==0 for reports in all_reports.values() for r in reports['candidate']['rows'])}
    def mean_or_none(values):return float(np.mean(values)) if values else None
    if full_isolation:checks['outside_full_exact']=all(full_isolation)
    detail={region:{name:mean_or_none([r[region]['methods'][name]['rmse_ev'] for r in detail_rows if r[region]['methods'][name]['edges']]) for name in ('analytic','parent','candidate_raw','candidate')} for region in ('highlight','fully_clipped')}
    aligned={name:mean_or_none([r['aligned_residual_mse'][name] for r in motion if r['valid_pixels']>0]) for name in ('parent','candidate_raw','candidate')}
    if reg.get('aligned_residual_control_max') is not None:
        checks['aligned_residual_preserved']=aligned['candidate'] is not None and aligned['candidate']<=reg['aligned_residual_control_max']
        for region in detail:
            checks[region+'_detail_no_worse_analytic']=detail[region]['candidate'] is not None and detail[region]['candidate']<=detail[region]['analytic']
    result={'checks':checks,'numerical_pass':all(checks.values()),'new_development':new,
        'H_error_by_dataset':{label:{name:high_error(report) for name,report in reports.items()} for label,reports in all_reports.items()},
        'legacy_natural':assess(all_reports['legacy_natural']['candidate']),
        'new_detail_gradient_rmse_ev':detail,'new_aligned_residual_mse':aligned,
        'detail_rows':detail_rows,'motion_rows':motion,'registration_sha256':sha(regpath),'evaluator_sha256':sha(__file__),
        'scope':'Previously specified research development checks only; legacy failures retained, real generated motion and grade/release checks remain separate','release_authorized':False}
    (out/'acceptance.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:result[k] for k in ('checks','H_error_by_dataset','new_detail_gradient_rmse_ev','new_aligned_residual_mse')},indent=2))


if __name__=='__main__':main()
