"""Export saved learned AP1 motion frames through ACES HDR and verify RGB16 decode."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rudra.delivery.aces_video import ACES2HDR1000View,encode_acescg_rgb16
from rudra.delivery.video import colour_tags,expected_tags
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--temporal-report',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('new output directory required')
    report=json.loads(a.temporal_report.read_text())
    if report['contract']['output']!='ACEScg linear':raise ValueError('ACEScg predictions required')
    a.out.mkdir(parents=True)
    view=ACES2HDR1000View();results=[]
    for ci,clip in enumerate(report['clips']):
        paths=[Path(r['prediction_path']) for r in clip['frames']]
        for path,row in zip(paths,clip['frames']):
            if sha(path)!=row['prediction_sha256']:raise ValueError('prediction identity mismatch')
        movie=encode_acescg_rgb16((np.load(path) for path in paths),a.out/f'clip{ci:02d}.mkv',clip['fps'])
        decoded=subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','rawvideo','-pix_fmt','rgb48le','pipe:1'],check=True,capture_output=True).stdout
        first=np.load(paths[0]);shape=first.shape
        count=len(paths)*int(np.prod(shape))
        if len(decoded)!=count*2:raise ValueError('decoded frame/shape count mismatch')
        frames=np.frombuffer(decoded,dtype='<u2').reshape(len(paths),*shape)
        max_error=0
        for path,frame in zip(paths,frames):
            expected=(view.forward(np.load(path)).astype(np.float64)*65535+.5).astype(np.uint16)
            max_error=max(max_error,int(np.abs(frame.astype(np.int32)-expected.astype(np.int32)).max()))
        tags=colour_tags(movie)
        probe=json.loads(subprocess.run(['ffprobe','-v','error','-show_streams','-of','json',str(movie)],capture_output=True,check=True).stdout)['streams'][0]
        row={'path':str(movie),'sha256':sha(movie),'frames':len(paths),'max_code_error_vs_official_transform':max_error,
             'pixel_format':probe['pix_fmt'],'bits_per_raw_sample':probe.get('bits_per_raw_sample'),
             'color_range':probe.get('color_range'),'tags':tags}
        if max_error!=0 or probe['pix_fmt']!='gbrp16le':raise ValueError('RGB16 transport mismatch')
        if probe.get('color_range')!='pc':raise ValueError('full range required')
        if any(tags.get(k)!=v for k,v in expected_tags('ffv1rgb16').items()):raise ValueError('colour tag mismatch')
        results.append(row);print(json.dumps(row),flush=True)
    (a.out/'verification.json').write_text(json.dumps({'temporal_report_sha256':sha(a.temporal_report),
        'checkpoint_sha256':report['checkpoint_sha256'],'display_transform':view.identity,'clips':results,
        'scope':'learned signed AP1 -> official HDR1000 Rec2020 PQ -> 16bit FFV1 -> exact decoded codes; display derivative, not scene-linear master; no player/native/release-quality claim'},indent=2))


if __name__=='__main__':main()
