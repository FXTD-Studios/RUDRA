"""Finish an already running duration experiment; never launches/restarts training."""
import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--training-pid',type=int,required=True)
    ap.add_argument('--steps',type=int,choices=(1600,6400),default=1600)
    ap.add_argument('--root',type=Path,help='Distinct output directory for a preserved failed-run retry')
    ap.add_argument('--fit-root',type=Path,help='Distinct TRAIN fit output directory')
    a=ap.parse_args()
    root=a.root or Path(f'G:/rudra_training/partial_pcgrad_duration{a.steps}_20261025')
    fit_root=a.fit_root or Path(f'G:/rudra_training/partial_duration{a.steps}_training_fit_20261025')
    prefix=400 if a.steps==1600 else 1600
    control=Path('G:/rudra_training/partial_pcgrad_deterministic_control400_20261025' if prefix==400 else 'G:/rudra_training/partial_pcgrad_duration1600_20261025')
    checkpoint=root/f'step{a.steps}.pt'
    status=root/'verification_status.json'
    def record(stage,**extra):
        status.write_text(json.dumps({'stage':stage,**extra},indent=2));print(stage,flush=True)
    try:
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x00100000,False,a.training_pid)
        if not handle:raise RuntimeError('Training PID is not live; inspect outcome manually')
        record('waiting_for_training',training_pid=a.training_pid)
        try:
            while True:
                value=kernel.WaitForSingleObject(handle,2000)
                if value==0:break
                if value!=258:raise RuntimeError('Training process wait failed')
        finally:kernel.CloseHandle(handle)
        train=json.loads((root/'report.json').read_text());old=json.loads((control/'report.json').read_text())
        checks={f'history_first{prefix}_exact':len(old['history'])==prefix and train['history'][:prefix]==old['history'],
                f'schedule_first{prefix}_exact':json.loads((root/'schedule.json').read_text())[:prefix]==json.loads((control/'schedule.json').read_text()),
                'crop_pool_exact':(root/'crop_pool.json').read_bytes()==(control/'crop_pool.json').read_bytes(),
                'motion_pool_exact':(root/'motion_pool.json').read_bytes()==(control/'motion_pool.json').read_bytes(),
                f'updates{a.steps}':len(train['history'])==a.steps,'frozen_parent_exact':train['frozen_parent_bit_exact']}
        (root/'duration_identity.json').write_text(json.dumps(checks,indent=2))
        if not all(checks.values()):raise RuntimeError('Duration comparison identity failed; no automatic evaluation')
        shutil.copyfile(control/'training_ancestry.json',root/'training_ancestry.json')
        configs=Path('configs/experiments');regs={}
        for kind in ('eval','grade','video'):
            r=json.loads((configs/f'partial_pcgrad_deterministic_control400_{kind}_20261025.json').read_text())
            if kind=='eval':r['steps']=a.steps
            if kind=='grade':
                r['evaluation']=(root/'paired_development/old_coherent/candidate_report.json').as_posix()
                r['parent']=(root/'paired_development/old_coherent/parent_report.json').as_posix()
            if kind=='video':r.update(candidate=str(checkpoint),checkpoint_sha256=train['checkpoint_sha256'])
            r['scope']=f'Duration{a.steps} comparison on reused DEV; all original limits retained. No release qualification.'
            path=configs/(f'{root.name}_{kind}.json' if a.root else f'partial_pcgrad_duration{a.steps}_{kind}_20261025.json')
            path.write_text(json.dumps(r,indent=2));regs[kind]=str(path)
        commands=[
            ('paired',['tools/evaluate_aces_context_motion.py','--registration',regs['eval'],'--root',str(root)]),
            ('grade',['tools/audit_aces_consistent_detail_grade.py','--registration',regs['grade'],'--out',str(root/'old_detail_grade.json')]),
            ('video',['tools/evaluate_aces_feather_video.py','--registration',regs['video'],'--out',str(root/'hunyuan48.json')]),
            ('preservation',['tools/audit_partial_preservation.py','--parent','G:/rudra_training/aces_guarded_projected600_20261024/step600.pt','--candidate',str(checkpoint),'--pilot','G:/datasets/corpora/aces_consistent_exposures_v2_20261024/val/report.json','--out',str(root/'preservation.json')]),
            ('train_fit',['tools/audit_partial_training_fit.py','--candidate',str(checkpoint),'--out',str(fit_root)])]
        for stage,argv in commands:
            record(stage,command=argv)
            with (root/(stage+'.log')).open('w') as log:
                subprocess.run([sys.executable,*argv],stdout=log,stderr=subprocess.STDOUT,check=True)
        record('complete',note='Evaluation finished; inspect individual gates before any promotion.')
    except Exception:
        record('failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
