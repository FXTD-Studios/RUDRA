import subprocess
import numpy as np
import pytest
import json,sys
from pathlib import Path
import torch
from tools.rudra_aces_export import video_frames


def test_rgb16_decode_preserves_codes_and_legacy_remains_eight_bit(tmp_path):
    codes=np.arange(1024,dtype=np.uint16)*64
    rgb=np.broadcast_to(codes[None,:,None],(8,1024,3)).copy()
    src=tmp_path/'rgb16.mkv'
    subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb48le','-s','1024x8','-r','1','-i','pipe:0','-frames:v','1','-c:v','ffv1','-level','3','-pix_fmt','gbrp16le',str(src)],input=rgb.astype('<u2').tobytes(),check=True)
    decoded=next(video_frames(src,matrix='auto',decode_bits=16))
    np.testing.assert_array_equal(np.rint(decoded*65535).astype(np.uint16),rgb)
    legacy=next(video_frames(src,matrix='auto'))
    assert np.unique(legacy).size<=256 and np.unique(decoded).size==1024


@pytest.mark.parametrize('range_name,low,high',[('tv',64,940),('pc',0,1023)])
def test_10bit_yuv_ranges_preserve_more_than_eight_bit_precision(tmp_path,range_name,low,high):
    width=high-low+1;yy=np.broadcast_to(np.arange(low,high+1,dtype='<u2'),(8,width)).copy();uv=np.full_like(yy,512)
    src=tmp_path/(range_name+'.mkv')
    subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','yuv444p10le','-s',f'{width}x8','-r','1','-i','pipe:0','-frames:v','1','-c:v','ffv1','-level','3','-pix_fmt','yuv444p10le','-color_range',range_name,'-colorspace','bt709',str(src)],input=yy.tobytes()+uv.tobytes()+uv.tobytes(),check=True)
    decoded=next(video_frames(src,decode_bits=16));line=decoded[0,:,0]
    assert np.unique(line).size>800
    assert np.all(np.diff(line)>=0) and line[0]<.003 and line[-1]>.995
    legacy=next(video_frames(src,decode_bits=8));assert np.unique(legacy[0,:,0]).size<=256


def test_export_cli_defaults_to_high_precision_and_writes_verified_masters(tmp_path):
    from training.aces_model import ACESModel
    from rudra.delivery.exr import read_exr
    src=tmp_path/'input.mkv';width=64;height=32
    codes=np.arange(width*height,dtype=np.uint16).reshape(height,width)*16
    rgb=np.repeat(codes[...,None],3,axis=2)
    subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb48le','-s',f'{width}x{height}','-r','24','-i','pipe:0','-frames:v','1','-c:v','ffv1','-level','3','-pix_fmt','gbrp16le',str(src)],input=rgb.astype('<u2').tobytes(),check=True)
    model=ACESModel(8,'pixel_rms','local_codes','analytic');checkpoint=tmp_path/'test_model.pt'
    torch.save({'model':model.state_dict(),'contract':model.contract()},checkpoint)
    out=tmp_path/'out';root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'tools/rudra_aces_export.py'),str(src),'--checkpoint',str(checkpoint),'--out',str(out),'--matrix','auto','--device','cpu','--video'],check=True,capture_output=True)
    report=json.loads((out/'report.json').read_text());assert report['provenance']['video_decode_bits']==16
    assert report['provenance']['video_source_metadata']['pix_fmt']=='gbrp16le'
    frame,attrs=read_exr(out/'exr/frame_000000.exr');assert frame.dtype==np.float32 and report['checks']['all_exr_bit_exact_readback']
    info=json.loads(subprocess.run(['ffprobe','-v','error','-show_streams','-of','json',report['video']['path']],capture_output=True,text=True,check=True).stdout)['streams'][0]
    assert info['pix_fmt']=='gbrp16le' and info['bits_per_raw_sample']=='16'
