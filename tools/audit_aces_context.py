"""Compare identical learned pixels under crop versus full-frame context."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import ACESModel
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','training_report','pilot','full_report','out'):p.add_argument('--'+key.replace('_','-'),dest=key,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new report required')
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
    training=json.loads(a.training_report.read_text());full=json.loads(a.full_report.read_text())
    assert training['checkpoint_sha256']==full['checkpoint_sha256']==sha(a.checkpoint)
    assert training['pilot_sha256']==sha(a.pilot)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=ACESModel(state['model']['predictor.stem.weight'].shape[0],normalization=state['contract'].get('normalization','group')).to(device).eval()
    assert model.contract()==state['contract'];model.load_state_dict(state['model'])
    rows={r['asset_id']:r for r in json.loads(a.pilot.read_text())['rows']}
    predictions={r['asset_id']:r for r in full['rows']};result=[]
    for crop in training['crops']:
        row=rows[crop['asset_id']];pred=predictions[crop['asset_id']]
        assert sha(row['sdr_path'])==row['sdr_sha256'] and sha(row['target_path'])==row['target_sha256']
        assert sha(pred['prediction_path'])==pred['prediction_sha256']
        x,y=crop['xy'];size=crop['size'];sl=np.s_[y:y+size,x:x+size]
        src=canonical_to_source(load_rgb(row['sdr_path'],hdr=False))[sl]
        target=np.load(row['target_path'])[sl];full_pred=np.load(pred['prediction_path'])[sl]
        with torch.inference_mode():
            output=model(torch.from_numpy(src.copy()).permute(2,0,1)[None].to(device))
            own=output.acescg[0].permute(1,2,0).cpu().numpy()
        metrics={}
        for label,region in (('all',np.s_[:,:]),('inner64',np.s_[32:-32,32:-32])):
            t=target[region];c=own[region];f=full_pred[region]
            metrics[label]={'crop_target_asinh_mse':float(np.mean((np.arcsinh(c)-np.arcsinh(t))**2)),
                            'full_target_asinh_mse':float(np.mean((np.arcsinh(f)-np.arcsinh(t))**2)),
                            'context_asinh_rmse':float(np.sqrt(np.mean((np.arcsinh(c)-np.arcsinh(f))**2))),
                            'crop_negative_pixels':int(np.any(c<0,axis=-1).sum()),
                            'full_negative_pixels':int(np.any(f<0,axis=-1).sum())}
        result.append({'crop':crop,'metrics':metrics})
    a.out.write_text(json.dumps({'checkpoint_sha256':sha(a.checkpoint),'rows':result,
        'scope':'same TRAIN pixels, crop/full context; includes normalization and convolution context, not causal attribution'},indent=2))


if __name__=='__main__':main()
