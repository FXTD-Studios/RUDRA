"""Run the preregistered constrained generated review with exact input identities."""
import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.summarize_aces_generated import sha,summarize


def main():
    regpath=Path('configs/experiments/aces_constrained_generated_review_20261024.json')
    reg=json.loads(regpath.read_text())
    root=Path('G:/rudra_training/aces_constrained_generated_review_20261024')
    root.mkdir(exist_ok=True)
    if sha(reg['candidate'])!='4904f8c37766a3a8baa11ed4864807b09025df70a82c3d3fd98854f22fc7322a':raise ValueError('checkpoint identity')
    for source in reg['expected_inputs']:
        if sha(source['path'])!=source['sha256']:raise ValueError('input identity')
        out=root/source['clip']
        if out.exists():raise ValueError(f'Existing output requires inspection: {out}')
        print('Starting',source['clip'],flush=True)
        with (root/(source['clip']+'.log')).open('w') as log:
            subprocess.run([sys.executable,'tools/rudra_aces_export.py',source['path'],'--checkpoint',reg['candidate'],
                '--out',str(out),'--video','--max-frames',str(source['expected_frames']),'--exr-every','12',
                '--matrix','bt709','--colour-backtrack'],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed',source['clip'],flush=True)
    result=summarize(root,regpath)
    (root/'summary.json').write_text(json.dumps(result,indent=2))
    print('Suite pass:',result['numerical_pass'],result['failures'],flush=True)


if __name__=='__main__':main()
