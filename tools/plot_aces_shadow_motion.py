"""Hash-verified graded shadow contact against matched control and known truth."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rudra.aces_view import ACES2View
from training.contract_checks import canonical_to_source
from training.sdr2hdr_dataset import load_rgb
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','candidate','control','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--clip-index',type=int,default=2);a=p.parse_args()
    if a.out.exists() or a.out.with_suffix('.json').exists():raise ValueError('new contact required')
    manifest=json.loads(a.manifest.read_text());reports={k:json.loads(v.read_text()) for k,v in [('candidate',a.candidate),('control',a.control)]}
    for report in reports.values():
        if report['manifest_sha256']!=sha(a.manifest):raise ValueError('manifest identity mismatch')
    clip=manifest['clips'][a.clip_index];frame=clip['frames'][0];paths={frame['sdr_path']:frame['sdr_sha256'],frame['reference_path']:frame['reference_sha256']}
    predictions={}
    for name,report in reports.items():
        record=report['clips'][a.clip_index]
        if record['scene_id']!=clip['scene_id'] or record['variant']!=clip['variant']:raise ValueError('clip mismatch')
        pred=record['frames'][0]
        if pred['frame']!=frame['frame']:raise ValueError('frame mismatch')
        paths[pred['prediction_path']]=pred['prediction_sha256'];predictions[name]=pred['prediction_path']
    for path,expected in paths.items():
        if sha(path)!=expected:raise ValueError('input content mismatch')
    view=ACES2View();source=canonical_to_source(load_rgb(frame['sdr_path'],hdr=False))
    images={'analytic':view.inverse(source),'control':np.load(predictions['control']),'candidate':np.load(predictions['candidate']),'original reference':np.load(frame['reference_path'])}
    h,w=source.shape[:2];canvas=Image.new('RGB',(4*w,3*(h+24)),'white');draw=ImageDraw.Draw(canvas)
    for row,ev in enumerate((0,3,6)):
        for col,(name,array) in enumerate(images.items()):
            draw.text((col*w+4,row*(h+24)+4),f'{name} {ev:+}EV',fill='black')
            rgb=np.rint(np.clip(view.forward(array*2.**ev),0,1)*255).astype('uint8')
            canvas.paste(Image.fromarray(rgb),(col*w,row*(h+24)+24))
    canvas.save(a.out)
    a.out.with_suffix('.json').write_text(json.dumps({'source_hashes':paths,'manifest_sha256':sha(a.manifest),
        'candidate_report_sha256':sha(a.candidate),'control_report_sha256':sha(a.control),'plot_sha256':sha(a.out),
        'scope':'first frame of selected clip; +6EV additional stress, not product threshold; display contact not calibrated review'},indent=2))


if __name__=='__main__':main()
