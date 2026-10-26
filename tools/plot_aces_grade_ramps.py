"""Plot actual graded scanlines and errors; plateau counts alone can mislead."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rudra.aces_view import ACES2View
from tools.evaluate_run05_dequant import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ramps',type=Path,required=True)
    a=p.parse_args();output=a.ramps/'ramp_diagnostic.png'
    if output.exists():raise ValueError('new plot required')
    report=json.loads((a.ramps/'report.json').read_text());view=ACES2View()
    if report['view']!=view.identity:raise ValueError('view mismatch')
    fig,axes=plt.subplots(3,2,figsize=(13,9),constrained_layout=True);inputs={}
    for row,(name,lo,hi,ev) in enumerate((('broad',.03,.97,0),('dark',.02,.12,3),('bright',.75,.98,-3))):
        path=a.ramps/f'{name}_prediction.npy';inputs[str(path)]=sha(path)
        clean=np.broadcast_to(np.linspace(lo,hi,1024,dtype=np.float32)[None,:,None],(128,1024,3)).copy()
        source=np.rint(clean*255)/255
        truth=view.forward(view.inverse(clean)*2.**ev)[64].mean(-1)*255
        baseline=view.forward(view.inverse(source)*2.**ev)[64].mean(-1)*255
        model=view.forward(np.load(path)*2.**ev)[64].mean(-1)*255;x=np.arange(400,600)
        for label,values,color in (('Smooth reference',truth,'black'),('Analytic baseline',baseline,'#de8f05'),('Candidate',model,'#0173b2')):
            axes[row,0].plot(x,values[x],label=label,color=color,lw=1.3)
        for label,values,color in (('Analytic baseline',baseline,'#de8f05'),('Candidate',model,'#0173b2')):
            axes[row,1].plot(x,values[x]-truth[x],label=label,color=color,lw=1.2)
        axes[row,0].set_title(f'{name.title()} ramp, {ev:+}EV');axes[row,1].set_title('Error against smooth reference')
        for ax in axes[row]:ax.set_xlabel('Horizontal pixel');ax.set_ylabel('8-bit display-code units');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('Grading diagnostic: assess remaining steps and ripples, not only average error')
    fig.savefig(output,dpi=130)
    output.with_suffix('.json').write_text(json.dumps({'report_sha256':sha(a.ramps/'report.json'),'prediction_sha256':inputs,
        'plot_sha256':sha(output),'scope':'centre-row zoom400:600, actual candidate outputs through declared view'},indent=2))


if __name__=='__main__':main()
