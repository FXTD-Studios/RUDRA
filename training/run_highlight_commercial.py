"""Run the established scratch recipe on the reviewed v2 candidate, never resume."""
from pathlib import Path
import json
import shutil
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pipeline.fire_reflection_readiness import DEFAULT_OUT,check_ready,read,require
from pipeline.commercial_preflight import sha256
from training import run_fxtd_commercial as runner

def normalized_review(root):
    review=read(root/'commercial_review.json')
    for item in review['sources'].values():
        item['evidence_path']=str((root/item['evidence_path']).resolve())
    return review

def prepare_combined(run,root=DEFAULT_OUT):
    root=Path(root);check_ready(root)
    data=Path(run)/'data';data.mkdir(exist_ok=True)
    # Copy the frozen manifest and review snapshots. Absolute evidence references
    # keep relative paths valid after relocating the review into a run directory.
    for source,target in [(root/'combined_candidate_manifest.jsonl',data/'manifest.jsonl'),
        (root/'ready.json',data/'candidate_ready.json'),(root/'review_status.json',data/'visual_review.json'),
        (root/'visual_review_notes.md',data/'visual_review_notes.md')]:
        if target.exists():require(sha256(target)==sha256(source),'Run-local artifact changed; use a new RunName: '+str(target))
        else:shutil.copyfile(source,target)
    review=normalized_review(root);target=data/'review.json'
    if target.exists():require(read(target)==review,'Run-local evidence changed; use a new RunName')
    else:target.write_text(json.dumps(review,indent=2),encoding='utf-8')
    return data

if __name__=='__main__':
    original_dump=runner.dump
    def expanded_dump(path,value):
        if Path(path).name=='recipe.json':
            value=dict(value,meaning='Internal development candidate using owned camera pairs, documented CC0 sources and authored scenes; not approved for release.',
                candidate_corpus=str(DEFAULT_OUT))
        original_dump(path,value)
    runner.dump=expanded_dump
    runner.prepare=prepare_combined
    runner.main()
