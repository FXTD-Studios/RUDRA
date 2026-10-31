import json,subprocess,sys
from pathlib import Path
import numpy as np
import cv2
import torch
from training.aces_model import ACESModel
from rudra.delivery.exr import read_exr

ROOT=Path(__file__).resolve().parents[1]


def checkpoint(tmp_path):
    torch.manual_seed(3);model=ACESModel(8,'pixel_rms','local_codes','analytic')
    with torch.no_grad():
        for p in model.parameters():p.add_(torch.randn_like(p)*.02)
    path=tmp_path/'model.pt';torch.save({'model':model.state_dict(),'contract':model.contract()},path);return path


def test_still_export_writes_float32_acescg_master_with_mask(tmp_path):
    img=np.broadcast_to(np.linspace(0,1,64,dtype=np.float32)[None,:,None],(48,64,3)).copy()
    img[:10]=.004;img[-8:,-8:]=1
    src=tmp_path/'shot.png';cv2.imwrite(str(src),np.rint(img*255).astype(np.uint8))
    out=tmp_path/'out'
    subprocess.run([sys.executable,str(ROOT/'tools/rudra_aces_export.py'),str(src),'--checkpoint',str(checkpoint(tmp_path)),
                    '--out',str(out),'--device','cpu'],check=True,cwd=ROOT)
    report=json.loads((out/'report.json').read_text())
    assert report['checks']['ordinary_within1_min']>=.999 and report['checks']['max_shadow_lift_code']<=1
    rgb,attrs=read_exr(out/'exr/shot.exr')
    assert rgb.shape==(48,64,3) and rgb.dtype==np.float32
    mask=attrs['reconstruction_mask']
    assert mask[-8:,-8:].max()>0 and mask[:10].max()==0      # clipped corner marked, analytic shadows not
    assert json.loads(attrs['rudra:provenance'])['contract']['shadow']=='analytic'


def test_capped_video_decode_finishes_without_pipe_error(tmp_path):
    from tools.rudra_aces_export import video_frames
    src=tmp_path/'input.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=24',
                    '-frames:v','12','-c:v','libx264',str(src)],check=True)
    frames=list(video_frames(src,max_frames=3))
    assert len(frames)==3 and frames[0].shape==(48,64,3)


def test_generated_review_rejects_short_successful_export():
    from tools.summarize_aces_generated import clip_checks
    r={'checks':{'frames':2,'all_exr_bit_exact_readback':True,'ordinary_within1_min':1.,
                 'max_shadow_lift_code':0.,'new_negative_luminance_pixels':0,'max_transition_excess_ev':0.},
       'frames':[{},{}],'video':{'path':'example.mkv'}}
    assert not clip_checks(r)['frames_processed']


def test_constrained_export_changes_only_partial_highlights(tmp_path):
    from tools.rudra_aces_export import load_model,infer
    from training.aces_reference import reference_view
    model,_=load_model(checkpoint(tmp_path),'cpu')
    source=np.zeros((48,64,3),np.float32)
    source[:16]=[.3,.4,.5];source[16:32]=[0,0,1];source[32:40]=1
    original,mask,_,base=infer(model,source,'cpu')
    constrained,mask2,stats,base2=infer(model,source,'cpu',colour_backtrack=True)
    support=(mask>0)&(source.min(-1)<254.5/255)
    np.testing.assert_array_equal(constrained[~support],original[~support])
    np.testing.assert_array_equal(mask,mask2);np.testing.assert_array_equal(base,base2)
    assert np.abs(reference_view().forward(constrained)[support]-source[support]).max()*255<=1
    assert stats['colour_constraint']['remaining_display_errors']==0
    all_high,_,stats,_=infer(model,source,'cpu',include_fully_clipped=True)
    assert stats['colour_constraint']['include_fully_clipped'] is True
    assert np.abs(reference_view().forward(all_high)[mask>0]-source[mask>0]).max()*255<=1
