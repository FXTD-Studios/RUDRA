"""Isolated scratch runner for reviewed v3 data. No resume or current-run edits."""
from pathlib import Path
import os
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.next_hdr_readiness import DEFAULT_OUT,check_ready
from training import run_highlight_commercial as adapter
from training import run_fxtd_commercial as runner

def active_trainers():
    import psutil
    found=[];own_tree={os.getpid(),*(p.pid for p in psutil.Process().parents())}
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            if proc.pid in own_tree or not (proc.info['name'] or '').lower().startswith('python'):continue
            args=proc.info['cmdline'] or []
            if '--prepare-only' in args:continue
            if any(re.fullmatch(r'(run_.*commercial|train_sdr2hdr)\.py',Path(a).name,re.I) for a in args):found.append(proc.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return found

def prepare_combined(run,root=DEFAULT_OUT):
    # Both adapters are modified only inside this new Python process.
    original=adapter.check_ready
    try:
        adapter.check_ready=check_ready
        return adapter.prepare_combined(run,root)
    finally:adapter.check_ready=original

def main():
    if '--prepare-only' not in sys.argv:
        active=active_trainers()
        if active:raise RuntimeError('Another HDR trainer is active: '+str(active)+'. Keep this next run stopped until it exits.')
    original_dump=runner.dump
    def next_dump(path,value):
        if Path(path).name=='recipe.json':
            value=dict(value,candidate_corpus=str(DEFAULT_OUT),meaning='Controlled commercial-development pilot: unchanged base plus 152 native-1080p CC0-derived pairs.',
                preparation_note='New pairs are 1920x1080; trainer uses 256x256 crops. Scratch only; no resume or automatic promotion.')
        original_dump(path,value)
    runner.dump=next_dump;runner.prepare=prepare_combined;runner.main()

if __name__=='__main__':main()
