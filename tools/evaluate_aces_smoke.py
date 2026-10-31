"""Full-frame TRAIN evaluation of experimental ACES learnability checkpoint."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import ACESModel,model_from_checkpoint
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha
from tools.audit_aces2_sources import scores
from rudra.delivery.exr import write_exr,read_exr
from rudra.delivery.colorspace import AP1_CHROMATICITIES


def validate_evaluation_rows(training,evaluation):
    if not training['rows'] or not evaluation['rows']:raise ValueError('nonempty datasets required')
    if any(r['split']!='train' for r in training['rows']) or any(r['split']!='val' for r in evaluation['rows']):
        raise ValueError('development evaluation requires train/val roles')
    if any(not r.get('scene_id') for r in training['rows']+evaluation['rows']):
        raise ValueError('scene identity required')
    if {r['scene_id'] for r in training['rows']}&{r['scene_id'] for r in evaluation['rows']}:
        raise ValueError('scene overlap')
    if {r['sdr_sha256'] for r in training['rows']}&{r['sdr_sha256'] for r in evaluation['rows']}:
        raise ValueError('source content overlap')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','pilot','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--training-pilot',type=Path,help='required for separate development evaluation')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output directory required')
    torch.set_num_threads(2);device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    pilot=json.loads(a.pilot.read_text());development=state['pilot_sha256']!=sha(a.pilot)
    if development:
        if a.training_pilot is None or state['pilot_sha256']!=sha(a.training_pilot):
            raise ValueError('checkpoint training dataset identity required')
        validate_evaluation_rows(json.loads(a.training_pilot.read_text()),pilot)
    channels=state['model']['predictor.stem.weight'].shape[0]
    model=model_from_checkpoint(state).to(device).eval()
    assert state['contract']==model.contract()
    model.load_state_dict(state['model'],strict=True)
    view=reference_view();a.out.mkdir(parents=True)
    results=[]
    for i,row in enumerate(pilot['rows']):
        assert row['split']==('val' if development else 'train') and sha(row['sdr_path'])==row['sdr_sha256']
        assert sha(row['target_path'])==row['target_sha256']
        source=canonical_to_source(load_rgb(row['sdr_path'],hdr=False));target=np.load(row['target_path'])
        start=time.time()
        with torch.inference_mode():
            out=model(torch.from_numpy(source).permute(2,0,1)[None].to(device))
            prediction=out.acescg[0].permute(1,2,0).cpu().numpy()
            baseline=out.baseline[0].permute(1,2,0).cpu().numpy()
            weight=out.reconstruction_weight[0,0].cpu().numpy()
        if not np.isfinite(prediction).all():raise ValueError('nonfinite prediction')
        predicted_codes=view.forward(prediction);top=source.max(-1)
        shadow=(weight>0)&(top<=.5);highlight=(weight>0)&(top>.5)
        loss={}
        for name,array in (('analytic',baseline),('model',prediction)):
            error=np.mean((np.arcsinh(array)-np.arcsinh(target))**2,axis=-1)
            loss[name]={'all_mse':float(error.mean())}
            for region,mask in (('shadow',shadow),('highlight',highlight)):
                loss[name][region+'_mse']=float(error[mask].mean()) if mask.any() else None
        delta=(predicted_codes-source).max(-1)*255
        path=a.out/f'{i:03d}_prediction.npy';np.save(path,prediction)
        record={'asset_id':row['asset_id'],'prediction_path':str(path),'prediction_sha256':sha(path),
                'seconds':time.time()-start,'roundtrip':scores(predicted_codes,source),'target_errors':loss,
                'shadow_pixels':int(shadow.sum()),'shadow_lift_gt1code':int((delta[shadow]>1).sum()),
                'shadow_max_lift_code':float(delta[shadow].max()) if shadow.any() else None,
                'ap1_min':float(prediction.min()),'ap1_max':float(prediction.max())}
        if i==0:
            exr=a.out/'learned_float32_probe.exr'
            changed=np.any(prediction!=baseline,axis=-1).astype(np.float32)
            write_exr(exr,prediction,half=False,chromaticities=AP1_CHROMATICITIES,
                      reconstruction_mask=changed,attributes={'rudra:provenance':json.dumps({
                          'checkpoint_sha256':sha(a.checkpoint),'contract':model.contract(),
                          'role':'research learnability checkpoint, not release quality',
                          'mask':'any exact float change from analytic inverse; ordinary R uses source support'})})
            decoded,attrs=read_exr(exr)
            np.testing.assert_array_equal(decoded,prediction)
            np.testing.assert_array_equal(attrs['reconstruction_mask'],changed)
            record['float32_exr']={'path':str(exr),'sha256':sha(exr),'bit_exact':True,
                                  'changed_mask_fraction':float(changed.mean())}
        if i in (0,6,8,9):
            sheet=Image.new('RGB',(1440,882),'white');draw=ImageDraw.Draw(sheet)
            for j,ev in enumerate((-3,0,3)):
                for k,(label,array) in enumerate((('analytic',baseline),('target',target),('model',prediction))):
                    draw.text((480*k+5,294*j+4),f'{label} {ev:+}EV',fill='black')
                    rgb=np.rint(np.clip(view.forward(array*2.**ev),0,1)*255).astype(np.uint8)
                    sheet.paste(Image.fromarray(rgb).resize((480,270)),(480*k,294*j+24))
            sheet.save(a.out/f'review_{i:02d}.png')
        results.append(record);print('[full frame]',i+1,record['shadow_lift_gt1code'],flush=True)
    (a.out/'report.json').write_text(json.dumps({'checkpoint_sha256':sha(a.checkpoint),
        'pilot_sha256':sha(a.pilot),'contract':model.contract(),'device':device,'rows':results,
        'evaluation_role':'scene-disjoint development' if development else 'training',
        'training_pilot_sha256':state['pilot_sha256'],
        'scope':'full-frame untiled FP32; development validation was used in earlier research, not independent release holdout'},indent=2))


if __name__=='__main__':main()
