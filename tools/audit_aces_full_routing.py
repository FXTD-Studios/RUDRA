"""Verify identical full predictions and measure boundary errors for routing comparison."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.diagnose_aces_full_branch import boundary_error
from tools.summarize_aces_generated import sha
from training.aces_model import reconstruction_weight
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'routing_boundaries.json'
    if out.exists():raise ValueError('new output required')
    (a.root/'routing_boundary_auditor.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2);rows=[];hashes={}
    for label,pilot in [('new_motion','aces_temporal_context_pairs_20261024'),('old_coherent','aces_consistent_exposures_v2_20261024')]:
        path=Path('G:/datasets/corpora')/pilot/'val/report.json';manifest=json.loads(path.read_text())['rows'];reports={};hashes[str(path)]=sha(path)
        for name,root,version in [('parent',a.root,'parent'),('isolated_raw',a.root,'candidate_raw'),('shared_raw',a.root/'shared_control','candidate_raw'),('isolated_constrained',a.root,'candidate'),('shared_constrained',a.root/'shared_control','candidate')]:
            rp=root/'paired_development'/label/(version+'_report.json');r=json.loads(rp.read_text());hashes[str(rp)]=sha(rp)
            if r['pilot_sha256']!=sha(path):raise ValueError('pilot identity')
            reports[name]={x['asset_id']:x for x in r['rows']}
            if set(reports[name])!={x['asset_id'] for x in manifest}:raise ValueError('complete report required')
        for row in manifest:
            for key in ('sdr','target'):
                if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('source identity')
            source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path']);x=torch.from_numpy(source).permute(2,0,1)[None]
            high=(reconstruction_weight(x)>0)&(x.amax(1,keepdim=True)>.5);full=high&(x.amin(1,keepdim=True)>=254.5/255);mask=full[0,0].numpy()
            valid=high.clone();valid[:,:,:4]=False;valid[:,:,-4:]=False;valid[:,:,:,:4]=False;valid[:,:,:,-4:]=False
            arrays={}
            for name,rr in reports.items():
                item=rr[row['asset_id']]
                if sha(item['prediction_path'])!=item['prediction_sha256']:raise ValueError('prediction identity')
                arrays[name]=np.load(item['prediction_path'])
            full_equal=np.array_equal(arrays['isolated_raw'][mask],arrays['shared_raw'][mask]);outside_equal=np.array_equal(arrays['isolated_raw'][~mask],arrays['parent'][~mask])
            if not full_equal or not outside_equal:raise ValueError('routing invariants')
            truth=torch.from_numpy(target).permute(2,0,1)[None]
            errors={k:boundary_error(torch.from_numpy(v).permute(2,0,1)[None],truth,valid,full) for k,v in arrays.items()}
            rows.append({'dataset':label,'asset_id':row['asset_id'],'full_raw_identical':full_equal,'outside_full_raw_identical':outside_equal,'boundary_mse':errors})
        print(label,'done',flush=True)
    summary={}
    for label in ('new_motion','old_coherent'):
        rr=[r for r in rows if r['dataset']==label and r['boundary_mse']['parent'] is not None]
        summary[label]={'images_with_boundary':len(rr),'mean_mse':{k:float(np.mean([r['boundary_mse'][k] for r in rr])) if rr else None for k in reports},
            'isolated_raw_worse_than_parent_images':sum(r['boundary_mse']['isolated_raw']>r['boundary_mse']['parent'] for r in rr)}
    out.write_text(json.dumps({'summary':summary,'rows':rows,'input_report_hashes':hashes,'auditor_sha256':sha(__file__),'scope':'Fixed coherent96 DEV; equal-image signed-asinh boundary residual, raw parent comparison. Descriptive seam diagnostic, not display/video acceptance.','release_authorized':False},indent=2,allow_nan=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
