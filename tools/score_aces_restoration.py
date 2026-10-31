"""Score saved predictions against fixed ordinary-pixel display-code targets."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import reconstruction_weight
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from rudra.aces_view import ACES2View
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pilot','evaluation','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    pilot=json.loads(a.pilot.read_text());evaluation=json.loads(a.evaluation.read_text())
    if evaluation['pilot_sha256']!=sha(a.pilot):raise ValueError('dataset mismatch')
    predicted={r['asset_id']:r for r in evaluation['rows']}
    if set(predicted)!={r['asset_id'] for r in pilot['rows']}:raise ValueError('prediction set mismatch')
    torch.set_num_threads(2);view=ACES2View();rows=[]
    for row in pilot['rows']:
        pred=predicted[row['asset_id']]
        for path,expected in ((row['sdr_path'],row['sdr_sha256']),(row['target_path'],row['target_sha256']),
                              (pred['prediction_path'],pred['prediction_sha256'])):
            if sha(path)!=expected:raise ValueError('content mismatch')
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False))
        target=view.forward(np.load(row['target_path']));model=view.forward(np.load(pred['prediction_path']))
        ordinary=reconstruction_weight(torch.from_numpy(source).permute(2,0,1)[None])[0,0].numpy()==0
        methods={}
        for name,code in (('analytic',view.forward(view.inverse(source))),('model',model)):
            error=np.mean(((code-target)*255)**2,axis=-1)
            methods[name]=float(error[ordinary].mean()) if ordinary.any() else None
        rows.append({'asset_id':row['asset_id'],'ordinary_pixels':int(ordinary.sum()),'code_mse':methods})
    valid=[r for r in rows if r['ordinary_pixels']]
    means={name:float(np.mean([r['code_mse'][name] for r in valid])) for name in ('analytic','model')}
    result={'checkpoint_sha256':evaluation['checkpoint_sha256'],'pilot_sha256':sha(a.pilot),'rows':rows,
            'equal_image_mean_code_mse':means,'relative_improvement':1-means['model']/means['analytic'] if means['analytic']>0 else None,
            'scope':'ordinary source-defined pixels only; fixed target display-code MSE, not reconstruction/temporal/grade acceptance'}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
