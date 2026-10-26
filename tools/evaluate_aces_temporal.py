"""Signed AP1 camera-motion diagnostic; no release-temporal claim."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import ACESModel,model_from_checkpoint
from training.aces_reference import reference_view
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha
from tools.audit_aces2_sources import scores

LUMA_AP1=np.array([.2722287168,.6740817658,.0536895174],np.float64)


def cell_luminance(rgb):
    y=np.asarray(rgb,dtype=np.float64)@LUMA_AP1
    return torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(y)[None,None],(36,64))[0,0].numpy()


def temporal_change(previous,current,baseline_previous,baseline_current):
    arrays=[np.asarray(x,dtype=np.float64) for x in (previous,current,baseline_previous,baseline_current)]
    if len({x.shape for x in arrays})!=1 or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('matching finite cell arrays required')
    valid=np.logical_and.reduce([x>0 for x in arrays])
    if not valid.any():return {'valid_cells':0,'excluded_cells':int(valid.size),'mean_abs_excess_ev':None,'max_abs_excess_ev':None}
    p,c,bp,bc=[np.log2(x[valid]) for x in arrays]
    delta=(c-p)-(bc-bp)
    return {'valid_cells':int(valid.sum()),'excluded_cells':int((~valid).sum()),
            'mean_abs_excess_ev':float(np.abs(delta).mean()),'max_abs_excess_ev':float(np.abs(delta).max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','checkpoint','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--colour-backtrack',action='store_true',help='Research-only partial-highlight appearance constraint')
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output directory required')
    torch.set_num_threads(2);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=model_from_checkpoint(state).eval()
    if state['contract']!=model.contract():raise ValueError('contract mismatch')
    model.load_state_dict(state['model'],strict=True)
    manifest=json.loads(a.manifest.read_text())
    if not manifest['clips']:raise ValueError('clips required')
    for clip in manifest['clips']:
        if len(clip['frames'])<2:raise ValueError('multiple frames required')
        for row in clip['frames']:
            if sha(row['sdr_path'])!=row['sdr_sha256']:raise ValueError('source identity mismatch')
    a.out.mkdir(parents=True);clips=[];view=reference_view()
    for ci,clip in enumerate(manifest['clips']):
        rows=[];transitions=[];previous=None
        for frame in clip['frames']:
            source=canonical_to_source(load_rgb(frame['sdr_path'],hdr=False))
            with torch.inference_mode():out=model(torch.from_numpy(source).permute(2,0,1)[None])
            rgb=out.acescg[0].permute(1,2,0).numpy();base=out.baseline[0].permute(1,2,0).numpy()
            colour_diagnostic=None
            if a.colour_backtrack:
                from training.aces_colour_backtrack import constrain_colour
                rgb,colour_diagnostic=constrain_colour(source,rgb,base,out.reconstruction_weight[0,0].numpy(),view)
            if not np.isfinite(rgb).all():raise ValueError('nonfinite prediction')
            coded=view.forward(rgb);shadow=(out.reconstruction_weight[0,0].numpy()>0)&(source.max(-1)<=.5)
            path=a.out/f'clip{ci:02d}_{frame["frame"]:05d}.npy';np.save(path,rgb)
            rows.append({'frame':frame['frame'],'prediction_path':str(path),'prediction_sha256':sha(path),
                         'colour_constraint':colour_diagnostic,
                         'roundtrip':scores(coded,source),'shadow_lift_gt1code':int((((coded-source).max(-1)*255>1)&shadow).sum()),
                         'negative_luminance_pixels':int(((rgb@LUMA_AP1)<0).sum()),
                         'baseline_negative_luminance_pixels':int(((base@LUMA_AP1)<0).sum()),
                         'new_negative_luminance_pixels':int((((rgb@LUMA_AP1)<0)&((base@LUMA_AP1)>=0)).sum())})
            current=(cell_luminance(rgb),cell_luminance(base))
            if previous is not None:
                transitions.append({'frame':frame['frame'],**temporal_change(previous[0],current[0],previous[1],current[1])})
            previous=current;print('clip/frame',ci,frame['frame'],flush=True)
        values=[x['mean_abs_excess_ev'] for x in transitions]
        clips.append({'scene_id':clip['scene_id'],'variant':clip.get('variant','unspecified'),'fps':clip['fps'],'frames':rows,'transitions':transitions,
                      'mean_cell_excess_ev':float(np.mean(values)) if all(x is not None for x in values) else None,
                      'max_transition_cell_excess_ev':max(values) if all(x is not None for x in values) else None})
    result={'checkpoint_sha256':sha(a.checkpoint),'manifest_sha256':sha(a.manifest),'contract':model.contract(),
            'colour_backtrack':a.colour_backtrack,'evaluator_sha256':sha(__file__),
            'constraint_sha256':sha('training/aces_colour_backtrack.py') if a.colour_backtrack else None,
            'device':'cpu','inference':'full-frame FP32, no temporal smoothing','clips':clips,
            'source_manifest_scope':manifest.get('scope','unspecified'),
            'scope':'fixed36x64 cell log2 AP1 luminance changes relative to analytic; nonpositive cells excluded and counted, no floor; motion confounded, not optical-flow aligned; input coverage is specified separately; not release proof',
            'release_authorized':False}
    (a.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
