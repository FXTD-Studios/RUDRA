"""Verify two research native masters and their derived RGB16 PQ transport."""
import json
import sys
import subprocess
from pathlib import Path
import numpy as np
import OpenEXR
import Imath
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.rudra_aces_export import load_model, read_still, infer
from tools.summarize_aces_generated import sha
from training.aces_reference import reference_view
from rudra.delivery.aces_video import encode_acescg_rgb16, ACES2HDR1000View


def read_exr(path):
    f=OpenEXR.InputFile(str(path));h=f.header();w=h['dataWindow']
    shape=(w.max.y-w.min.y+1,w.max.x-w.min.x+1)
    names=['R','G','B','rudra.reconstructed']
    assert set(h['channels'])==set(names)
    assert all(h['channels'][n].type==Imath.PixelType(Imath.PixelType.FLOAT) for n in names)
    assert h['rudra:sourceSpace']==b'ap1' and h['rudra:exposureScale']==b'1.0'
    rgb=np.stack([np.frombuffer(f.channel(n),np.float32).reshape(shape) for n in names[:3]],-1)
    mask=np.frombuffer(f.channel(names[-1]),np.float32).reshape(shape);f.close()
    assert np.isfinite(rgb).all() and np.isfinite(mask).all()
    return rgb,mask


def main():
    root=Path('reports/native_aces_v2_setup_20261025')
    checkpoint=Path('G:/rudra_training/aces_guarded_projected600_20261024/step600.pt')
    torch.set_num_threads(2);reference_view().set_threads(2)
    model,_=load_model(str(checkpoint),'cpu');records=[];frames=[]
    for i in range(2):
        source=read_still(root/'frames'/f'{i}.png')
        expected,mask,_,_=infer(model,source,'cpu',include_fully_clipped=True,clipped_smoothing='hybrid64')
        cpu,cm=read_exr(root/'native_master_cpu'/f'{i}.exr')
        dml,dm=read_exr(root/'native_master_directml'/f'{i}.exr')
        code=float(np.max(np.abs(reference_view().forward(dml)-reference_view().forward(cpu)))*255)
        rel=float(np.max(np.abs(dml.astype(np.float64)-cpu)/(np.abs(cpu)+1e-3)))
        row={'frame':i,'cpu_python_exact':bool(np.array_equal(cpu,expected)),
             'cpu_mask_exact':bool(np.array_equal(cm,mask)),'dml_mask_exact':bool(np.array_equal(dm,mask)),
             'dml_max_display_code_difference':code,'dml_max_relative_difference':rel,
             'dml_tolerance_pass':code<=.25 and rel<=.005,
             'cpu_min':float(cpu.min()),'cpu_max':float(cpu.max()),
             'source_sha256':sha(root/'frames'/f'{i}.png')}
        records.append(row);frames.append(cpu);print(row,flush=True)
    video=root/'native_derived_pq16.mkv'
    if not video.exists():
        video=encode_acescg_rgb16(frames,video,fps=24,ocio_threads=2)
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(video)]))['streams'][0]
    decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(video),'-f','rawvideo','-pix_fmt','rgb48le','pipe:1'])
    view=ACES2HDR1000View(threads=2)
    # Encoder quantizes float32 view codes in float64; float32 multiply can
    # round near half-code boundaries differently even for lossless transport.
    expected=np.stack([(view.forward(f).astype(np.float64)*65535+.5).astype(np.uint16) for f in frames])
    actual=np.frombuffer(decoded,dtype='<u2').reshape(expected.shape)
    video_checks={'rgb16':meta['pix_fmt']=='gbrp16le','codec':meta['codec_name']=='ffv1',
                  'pq':meta['color_transfer']=='smpte2084','primaries':meta['color_primaries']=='bt2020',
                  'full_range':meta['color_range']=='pc','decoded_codes_exact':bool(np.array_equal(actual,expected))}
    result={'rows':records,'video_checks':video_checks,'video_metadata':meta,'checkpoint_sha256':sha(checkpoint),
            'video_sha256':sha(video),'scope':'Two research DEV frames; nativeCPU/DirectML plus derived Python encoder transport, not native video inference or release qualification.'}
    (root/'master_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    assert all(r['cpu_python_exact'] and r['cpu_mask_exact'] and r['dml_mask_exact'] and r['dml_tolerance_pass'] for r in records)
    assert all(video_checks.values())
    print(json.dumps(video_checks),flush=True)


if __name__=='__main__':main()
