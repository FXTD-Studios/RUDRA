"""Original analytic object-motion HDR fixtures with inter-frame H264 variants."""
import argparse,json,subprocess,sys
from pathlib import Path
import cv2
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rudra.aces_view import ACES2View
from pipeline.run05_corpus import canonical
from tools.evaluate_run05_dequant import sha


def render(index,scene,count=96):
    # Supersample silhouette/shadow edges; lighting is fixed across frames.
    h,w=432,768;y,x=np.mgrid[:h,:w].astype(np.float32);x/=2;y/=2
    phase=2*np.pi*index/(count-1);cx=64+128*(1-np.cos(phase));cy=108+22*np.sin(phase)
    dx=(x-cx)/30;dy=(y-cy)/30;inside=dx*dx+dy*dy<=1
    z=np.sqrt(np.maximum(1-dx*dx-dy*dy,0))
    background=.003+.06*x/384+.025*y/216
    if scene:background*=.04
    rgb=np.repeat(background[...,None],3,axis=-1)*np.array([.9,1.,1.1],np.float32)
    shadow=np.exp(-((x-cx-18)/42)**2-((y-cy-24)/12)**2)
    rgb*=1-.85*shadow[...,None]
    diffuse=.015+.24*np.maximum(.35*dx-.4*dy+.8*z,0)
    texture=1+.08*np.sin(dx*14)*np.cos(dy*12)
    specular=1800*np.exp(-((dx+.22)**2+(dy+.3)**2)*85)
    colour=np.array([1.,.55,.18] if scene==0 else [.25,.65,1.],np.float32)
    sphere=diffuse[...,None]*texture[...,None]*colour+specular[...,None]
    if scene:sphere*=.2
    rgb[inside]=sphere[inside]
    return cv2.resize(rgb.astype(np.float32),(384,216),interpolation=cv2.INTER_AREA),[float(cx),float(cy)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output required')
    torch.set_num_threads(2);a.out.mkdir(parents=True);view=ACES2View();clips=[]
    for scene in range(2):
        folder=a.out/f'scene{scene}';folder.mkdir();codes=[];truth=[]
        for index in range(96):
            hdr,position=render(index,scene);path=folder/f'{index:03d}_ap1.npy';np.save(path,hdr)
            codes.append(np.rint(np.clip(view.forward(hdr),0,1)*255).astype(np.uint8))
            truth.append({'frame':index,'reference_path':str(path),'reference_sha256':sha(path),'object_center':position})
        rgb=np.stack(codes);video=folder/'source_h264_crf28.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','384x216','-r','24','-i','pipe:0',
            '-vf','scale=out_color_matrix=bt709:out_range=tv','-c:v','libx264','-crf','28','-preset','medium','-g','48',
            '-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv',str(video)],
            input=rgb.tobytes(),capture_output=True,check=True)
        raw=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf','scale=in_color_matrix=bt709:in_range=tv:out_range=pc',
            '-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True).stdout
        decoded=np.frombuffer(raw,np.uint8).reshape(rgb.shape)
        for variant,frames in (('uncompressed',rgb),('h264_crf28',decoded)):
            out=folder/variant;out.mkdir();rows=[]
            for reference,frame in zip(truth,frames):
                path=out/f'{reference["frame"]:03d}.png'
                encoded=np.rint(canonical(frame.astype(np.float32)/255)*65535).astype(np.uint16)
                if not cv2.imwrite(str(path),encoded[...,::-1]):raise ValueError('PNG write failed')
                rows.append({**reference,'sdr_path':str(path),'sdr_sha256':sha(path)})
            clips.append({'scene_id':f'original_object_motion_{scene}','variant':variant,'fps':24,'frames':rows,
                          'codec_applied':variant,'vae_applied':False,'source_video':str(video) if variant!='uncompressed' else None,
                          'source_video_sha256':sha(video) if variant!='uncompressed' else None,
                          'provenance':'original analytic rendering; no external media; no public licence dedication'})
        print('scene completed',scene,flush=True)
    report={'clips':clips,'view':view.identity,'builder_sha256':sha(Path(__file__)),
            'reference_units':'scene-linear ACEScg, fixed arbitrary scene exposure; no inferred physical nits',
            'scope':'original analytic synthetic development: moving sphere, specular, shadow, occlusion over gradient; two96frame scenes plus H264 variants; no real people/generator/platform or independent release proof',
            'release_authorized':False}
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
