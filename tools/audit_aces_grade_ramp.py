"""Known smooth-ramp dequantization diagnostic at -3/0/+3 EV; not a release suite."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import ACESModel,model_from_checkpoint
from rudra.aces_view import ACES2View
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output directory required')
    torch.set_num_threads(2);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=model_from_checkpoint(state).eval()
    if model.contract()!=state['contract']:raise ValueError('contract mismatch')
    model.load_state_dict(state['model'],strict=True);view=ACES2View();rows=[]
    a.out.mkdir(parents=True)
    for label,lo,hi in (('broad',.03,.97),('dark',.02,.12),('bright',.75,.98)):
        clean=np.broadcast_to(np.linspace(lo,hi,1024,dtype=np.float32)[None,:,None],(128,1024,3)).copy()
        source=np.rint(clean*255)/255;truth=view.inverse(clean)
        with torch.inference_mode():out=model(torch.from_numpy(source).permute(2,0,1)[None])
        prediction=out.acescg[0].permute(1,2,0).numpy();baseline=out.baseline[0].permute(1,2,0).numpy()
        np.save(a.out/f'{label}_prediction.npy',prediction)
        grades={}
        for ev in (-3,0,3):
            true_code=view.forward(truth*2.**ev)[64,64:-64].mean(-1)*255
            reference_gradient=np.diff(true_code.astype(np.float64));valid=reference_gradient>1e-7
            methods={}
            for name,values in (('analytic',baseline),('model',prediction)):
                line=view.forward(values*2.**ev)[64,64:-64].mean(-1)*255
                gradient=np.diff(line.astype(np.float64))
                methods[name]={'code_rmse':float(np.sqrt(np.mean((line-true_code)**2))),
                    'gradient_rmse_code':float(np.sqrt(np.mean((gradient-reference_gradient)**2))),
                    'plateau_fraction':float(np.mean(np.abs(gradient[valid])<.01*reference_gradient[valid])) if valid.any() else None,
                    'reversed_gradient_fraction':float(np.mean(gradient[valid]<-.01*reference_gradient[valid])) if valid.any() else None,
                    'max_reversal_code':float(np.maximum(-gradient,0).max())}
            grades[str(ev)]={'evaluated_edges':int(valid.sum()),'methods':methods}
        rows.append({'ramp':label,'source_bounds':[lo,hi],'grades':grades});print(json.dumps(rows[-1]),flush=True)
    (a.out/'report.json').write_text(json.dumps({'checkpoint_sha256':sha(a.checkpoint),'view':view.identity,'rows':rows,
       'scope':'three synthetic neutral ramps, exact inverse of continuous display ramp as truth, source rounded8bit; centre row excluding64px borders; plateau and reversal use1% true positive step; diagnostic only, not complete G gate'},indent=2,allow_nan=False))


if __name__=='__main__':main()
