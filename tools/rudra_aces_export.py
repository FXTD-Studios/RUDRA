"""Run the v1 ACES model on SDR stills or video and write the grading master.

Outputs, per input frame:
  * 32-bit float scene-linear OpenEXR in ACEScg (AP1 primaries in the header), the
    grading master. Values are signed and unclamped.
  * channel ``rudra.reconstructed``: reconstruction weight of the highlight support
    (source pixels at the last 8-bit code), i.e. where radiance was estimated rather
    than taken from the source. Shadows (analytic in v1) and the sub-code banding
    restoration are 0. It is a support/blend mask, not a confidence map.
Optional: a derived display file, ACES 2.0 HDR 1000-nit Rec.2020 PQ, lossless 16-bit
RGB FFV1 in Matroska. It is display-referred; the EXR stays the master.

Source codes are read as Rec.709 display RGB (Rec.1886). Video is decoded by ffmpeg to
16-bit RGB by the export CLI with a BT.709 matrix unless --matrix auto. The legacy
8-bit decode remains explicit for historical comparisons. Nothing here recovers camera RAW.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.aces_model import model_from_checkpoint,shadow_support
from training.aces_reference import reference_view
from rudra.delivery.exr import write_exr,read_exr
from rudra.delivery.colorspace import AP1_CHROMATICITIES
from tools.evaluate_aces_temporal import LUMA_AP1,cell_luminance,temporal_change

STILLS={'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}


def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def load_model(checkpoint,device):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=model_from_checkpoint(state)
    if model.contract()!=state['contract']:raise ValueError('contract mismatch')
    return model.to(device).eval(),state


def read_still(path):
    import cv2
    image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if image is None:raise ValueError(f'cannot read {path}')
    if image.ndim==2:image=np.repeat(image[...,None],3,2)
    if image.shape[2]==4:image=image[...,:3]
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    if not np.issubdtype(image.dtype,np.integer):raise ValueError('integer SDR codes required (8 or 16 bit)')
    return image.astype(np.float32)/float(np.iinfo(image.dtype).max)


def probe_video(path):
    out=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                        'stream=width,height,r_frame_rate,pix_fmt,color_range,color_space,color_transfer,color_primaries',
                        '-of','json',str(path)],capture_output=True,check=True,text=True).stdout
    s=json.loads(out)['streams'][0];num,den=s['r_frame_rate'].split('/')
    s['fps']=float(num)/float(den);return s


def video_frames(path,matrix='bt709',max_frames=0,decode_bits=8):
    """Legacy API defaults to8 for benchmark reproducibility; export CLI selects16."""
    if decode_bits not in (8,16):raise ValueError('decode_bits must be8 or16')
    info=probe_video(path);w,h=info['width'],info['height']
    vf=[] if matrix=='auto' else ['-vf',f'scale=in_color_matrix={matrix}:flags=accurate_rnd+full_chroma_int']
    limit=['-frames:v',str(max_frames)] if max_frames else []
    cmd=['ffmpeg','-v','error','-i',str(path),*vf,*limit,'-f','rawvideo','-pix_fmt','rgb48le' if decode_bits==16 else 'rgb24','pipe:1']
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE)
    size=w*h*3*(decode_bits//8)
    completed=False
    try:
        while True:
            buf=proc.stdout.read(size)
            if not buf:
                completed=True
                break
            if len(buf)!=size:raise ValueError('truncated video frame')
            yield np.frombuffer(buf,np.dtype('<u2') if decode_bits==16 else np.uint8).reshape(h,w,3).astype(np.float32)/(65535 if decode_bits==16 else 255)
    finally:
        proc.stdout.close()
        if not completed and proc.poll() is None:proc.terminate()
        code=proc.wait()
        if completed and code!=0:raise RuntimeError('ffmpeg decode failed')


CLIPPED_SMOOTHING={'hybrid64':dict(sigma=64.0,ramp=1.0,start=0.0,mode='hybrid')}


def infer(model,source,device,colour_backtrack=False,include_fully_clipped=False,repair_inverse_boundary=False,clipped_smoothing='none',forbid_new_negative_y=False):
    """source HxWx3 display codes [0,1] -> (acescg HxWx3 float32, mask HxW, stats)."""
    x=torch.from_numpy(np.ascontiguousarray(source,dtype=np.float32)).permute(2,0,1)[None].to(device)
    with torch.inference_mode():
        out=model(x)
        weight=out.reconstruction_weight
        highlight=(weight>0)&(x.amax(1,keepdim=True)>.5)
        mask=torch.where(highlight,weight,torch.zeros_like(weight))[0,0].cpu().numpy().astype(np.float32)
        shadow=shadow_support(x,weight)[0,0].cpu().numpy()
        acescg=out.acescg[0].permute(1,2,0).cpu().numpy().astype(np.float32)
        baseline=out.baseline[0].permute(1,2,0).cpu().numpy().astype(np.float32)
    if clipped_smoothing!='none':
        # Fully clipped interiors: predicted texture is uncorrelated with truth, so flatten
        # it per connected component (log luminance + RGB mean); outside support bit-exact.
        from training.aces_clipped_smoothing import smooth_clipped
        acescg,_=smooth_clipped(acescg,source,**CLIPPED_SMOOTHING[clipped_smoothing])
    colour_diagnostic=None
    if colour_backtrack or include_fully_clipped:
        from training.aces_colour_backtrack import constrain_colour
        acescg,colour_diagnostic=constrain_colour(source,acescg,baseline,weight[0,0].cpu().numpy(),reference_view(),include_fully_clipped=include_fully_clipped,repair_inverse_boundary=repair_inverse_boundary,forbid_new_negative_y=forbid_new_negative_y)
    if not np.isfinite(acescg).all():raise ValueError('nonfinite model output')
    shown=reference_view().forward(acescg)
    delta=np.abs(shown-source).max(-1)*255;ordinary=(weight[0,0].cpu().numpy()==0)
    stats={'reconstructed_fraction':float((mask>0).mean()),'shadow_support_fraction':float(shadow.mean()),
           'colour_constraint':colour_diagnostic,
           'ordinary_within1_fraction':float((delta[ordinary]<=1).mean()) if ordinary.any() else None,
           'shadow_max_lift_code':float((shown-source).max(-1)[shadow].max()*255) if shadow.any() else None,
           'acescg_min':float(acescg.min()),'acescg_max':float(acescg.max()),
           'negative_luminance_pixels':int(((acescg@LUMA_AP1)<0).sum()),
           'new_negative_luminance_pixels':int((((acescg@LUMA_AP1)<0)&((baseline@LUMA_AP1)>=0)).sum())}
    return acescg,mask,stats,baseline


def contact_sheet(path,source,acescg,baseline,mask):
    """Source | ACES SDR view of master | master +2EV | analytic +2EV | reconstruction mask."""
    from PIL import Image,ImageDraw
    view=reference_view();h,w=source.shape[:2];tw=min(384,w);th=max(1,round(h*tw/w))
    panels=[('source',source),('master, SDR view',view.forward(acescg)),('master +2 EV',view.forward(acescg*4)),
            ('analytic +2 EV',view.forward(baseline*4)),('reconstructed mask',np.repeat(mask[...,None],3,-1))]
    sheet=Image.new('RGB',(tw*len(panels),th+20),'white');draw=ImageDraw.Draw(sheet)
    for k,(label,rgb) in enumerate(panels):
        draw.text((k*tw+4,3),label,fill='black')
        sheet.paste(Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).resize((tw,th)),(k*tw,20))
    sheet.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input',type=Path,help='still image, folder of stills, or video file')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True,help='new output folder')
    p.add_argument('--video',action='store_true',help='also write the derived 16-bit PQ HDR1000 FFV1 .mkv')
    p.add_argument('--matrix',default='bt709',choices=('bt709','bt601','auto'),help='YCbCr matrix for video decode')
    p.add_argument('--decode-bits',type=int,choices=(8,16),default=16,help='Video RGB decode precision;16 preserves high-bit-depth input,8 reproduces legacy review decoding')
    p.add_argument('--repair-inverse-boundary',action='store_true',help='Experimental measured near-white inverse fallback; requires colour-backtrack')
    p.add_argument('--forbid-new-negative-y',action='store_true',help='appearance constraint also backtracks new negative AP1 luminance')
    p.add_argument('--device',default='auto',choices=('auto','cpu','cuda'))
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--ocio-threads',type=int,choices=range(1,17),default=4)
    p.add_argument('--colour-backtrack',action='store_true',help='Experimental partial-highlight colour constraint; research use only')
    p.add_argument('--colour-backtrack-all',action='store_true',help='Experimental appearance constraint on all highlights, including fully clipped pixels')
    p.add_argument('--clipped-smoothing',default='none',choices=('none',)+tuple(CLIPPED_SMOOTHING),help='fully clipped interior smoothing applied before the appearance constraint (hybrid64 = guarded_projected600 release contract)')
    p.add_argument('--log',choices=('logc3','logc4'),help='also write ARRI LogC3 (EI800, AWG3) or LogC4 (AWG4) 16-bit TIFFs from the master (with --video: a ProRes 4444 XQ log .mov too)')
    p.add_argument('--log-exposure',type=float,default=0.0,help='stops applied to the master before log encoding (e.g. -2 to fit bright highlights into LogC3); recorded in the report')
    p.add_argument('--exr-every',type=int,default=1,help='write the EXR master for every Nth frame only (review runs); all frames still feed the video and checks')
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output folder required')
    if a.exr_every<1:raise ValueError('--exr-every must be >= 1')
    if a.max_frames<0:raise ValueError('--max-frames must be >= 0')
    if a.forbid_new_negative_y and not (a.colour_backtrack or a.colour_backtrack_all):raise ValueError('luminance guard requires appearance constraint')
    if a.repair_inverse_boundary and not (a.colour_backtrack or a.colour_backtrack_all):raise ValueError('inverse repair requires appearance constraint')
    reference_view().set_threads(a.ocio_threads)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    model,state=load_model(a.checkpoint,device)
    if state['contract'].get('output')!='ACEScg linear':raise ValueError('ACES model checkpoint required')
    fps=24.
    if a.input.is_dir():
        files=sorted(f for f in a.input.iterdir() if f.suffix.lower() in STILLS)
        if not files:raise ValueError('no stills in folder')
        names=[f.stem for f in files];frames=(read_still(f) for f in files)
    elif a.input.suffix.lower() in STILLS:
        names=[a.input.stem];frames=iter([read_still(a.input)])
    else:
        info=probe_video(a.input);fps=info['fps'];names=None;frames=video_frames(a.input,a.matrix,a.max_frames,a.decode_bits)
    a.out.mkdir(parents=True);(a.out/'exr').mkdir()
    provenance={'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),'contract':state['contract'],
                'video_decode_bits':a.decode_bits if names is None else None,
                'video_decode_matrix':a.matrix if names is None else None,
                'video_source_metadata':info if names is None else None,
                'repair_inverse_boundary':a.repair_inverse_boundary,'forbid_new_negative_y':a.forbid_new_negative_y,
                'clipped_smoothing':None if a.clipped_smoothing=='none' else {'name':a.clipped_smoothing,**CLIPPED_SMOOTHING[a.clipped_smoothing],
                    'sha256':sha(Path(__file__).resolve().parents[1]/'training/aces_clipped_smoothing.py')},
                'colour_backtrack':a.colour_backtrack or a.colour_backtrack_all,'include_fully_clipped':a.colour_backtrack_all,
                'colour_constraint_sha256':sha(Path(__file__).resolve().parents[1]/'training/aces_colour_backtrack.py') if a.colour_backtrack or a.colour_backtrack_all else None,
                'implementation_sha256':{name:sha(Path(__file__).resolve().parents[1]/name) for name in
                    ('tools/rudra_aces_export.py','training/aces_model.py','training/aces_reference.py','rudra/aces_view.py')+
                    (('training/aces_context_model.py',) if state['contract'].get('highlight_context') else ())},
                'input':str(a.input),'master':'ACEScg scene-linear, signed, unclamped; source appearance is evaluated through ACES 2.0 SDR100 Rec.709',
                'mask':'rudra.reconstructed = highlight reconstruction weight; 0 = from source (incl. analytic shadows)',
                'not':'not camera RAW; reconstructed highlights are estimates'}
    records=[];state_cells=[None]
    log=None;log_movie=[None];log_dir=None
    if a.log:
        from rudra.delivery.aces_log import LogEncoder,ProResLogWriter
        log=LogEncoder(a.log,a.log_exposure);log_dir=a.out/f'log_{a.log}';log_dir.mkdir()
        provenance['log_delivery']={**log.identity,'files':'16-bit TIFF per frame'+(', ProRes 4444 XQ .mov (lossy; TIFFs are lossless)' if a.video else ''),
                                    'note':'scene-referred camera log of the master; the reconstruction mask stays in the EXR'}

    def produce():
        for i,source in enumerate(frames):
            if a.max_frames and i>=a.max_frames:break
            start=time.time();acescg,mask,stats,baseline=infer(model,source,device,a.colour_backtrack,a.colour_backtrack_all,a.repair_inverse_boundary,a.clipped_smoothing,a.forbid_new_negative_y)
            record={'frame':i,'seconds':None,**stats}
            if i%a.exr_every==0:
                name=(names[i] if names else f'frame_{i:06d}')+'.exr';path=a.out/'exr'/name
                write_exr(path,acescg,half=False,chromaticities=AP1_CHROMATICITIES,reconstruction_mask=mask,
                          attributes={'rudra:provenance':json.dumps({**provenance,'frame':i})})
                decoded,attrs=read_exr(path)
                if not np.array_equal(decoded,acescg) or not np.array_equal(attrs['reconstruction_mask'],mask):raise ValueError('EXR readback mismatch')
                record.update(exr=str(path),sha256=sha(path))
            if log is not None:
                codes,clipped=log.encode(acescg);record['log_clipped_pixels']=clipped
                if i%a.exr_every==0:
                    import cv2
                    tif=log_dir/((names[i] if names else f'frame_{i:06d}')+'.tif')
                    if not cv2.imwrite(str(tif),cv2.cvtColor(log.to_uint16(codes),cv2.COLOR_RGB2BGR)):raise IOError(f'could not write {tif}')
                if a.video:
                    if log_movie[0] is None:log_movie[0]=ProResLogWriter(a.out/f'master_{a.log}_prores4444xq.mov',codes.shape[1],codes.shape[0],fps)
                    log_movie[0].write(codes)
            record['seconds']=time.time()-start
            cells=(cell_luminance(acescg),cell_luminance(baseline))
            if state_cells[0] is not None:record['transition']=temporal_change(state_cells[0][0],cells[0],state_cells[0][1],cells[1])
            state_cells[0]=cells
            if i==0:contact_sheet(a.out/'contact_first_frame.png',source,acescg,baseline,mask)
            records.append(record)
            print(json.dumps(records[-1]),flush=True)
            yield acescg

    movie=None
    if a.video:
        from rudra.delivery.aces_video import encode_acescg_rgb16
        movie=encode_acescg_rgb16(produce(),a.out/'display_hdr1000_pq_rgb16.mkv',fps,ocio_threads=a.ocio_threads)
    else:
        for _ in produce():pass
    if not records:raise ValueError('no frames decoded; export is incomplete')
    log_movie_path=log_movie[0].close() if log_movie[0] is not None else None
    report={'provenance':provenance,'device':device,'ocio_threads':a.ocio_threads,'fps':fps,'frames':records,
            'video':{'path':str(movie),'sha256':sha(movie),'kind':'ACES 2.0 HDR 1000 nit Rec.2020 PQ, FFV1 gbrp16le, display-referred'} if movie else None,
            'log':None if log is None else {**log.identity,'folder':str(log_dir),'prores':str(log_movie_path) if log_movie_path else None,
                                            'clipped_pixels_total':sum(r.get('log_clipped_pixels',0) for r in records)},
            'checks':{'all_exr_bit_exact_readback':True,
                      'appearance_constraint_remaining_pixels':sum(r['colour_constraint']['remaining_display_errors'] for r in records if r['colour_constraint'] is not None),
                      'appearance_constraint_pass':all(r['colour_constraint']['remaining_display_errors']==0 for r in records) if a.colour_backtrack or a.colour_backtrack_all else None,
                      'inverse_boundary_repaired_pixels':sum(r['colour_constraint'].get('inverse_boundary_repaired_pixels',0) for r in records if r['colour_constraint'] is not None),
                      'ordinary_within1_min':min((r['ordinary_within1_fraction'] for r in records if r['ordinary_within1_fraction'] is not None),default=None),
                      'max_shadow_lift_code':max((r['shadow_max_lift_code'] for r in records if r['shadow_max_lift_code'] is not None),default=None),
                      'new_negative_luminance_pixels':sum(r['new_negative_luminance_pixels'] for r in records),
                      'mean_transition_excess_ev':float(np.mean(t)) if (t:=[r['transition']['mean_abs_excess_ev'] for r in records if r.get('transition') and r['transition']['mean_abs_excess_ev'] is not None]) else None,
                      'max_transition_excess_ev':max(t) if t else None,
                      'frames':len(records)}}
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    if log is not None and report['log']['clipped_pixels_total']:
        print(f"warning: {report['log']['clipped_pixels_total']} pixel(s) exceed {log.colorspace}; use --log logc4 or a negative --log-exposure",flush=True)
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
