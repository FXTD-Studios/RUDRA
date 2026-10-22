"""Bind actual review to the isolated v3 candidate; never trains."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import fire_reflection_readiness as base_gate
from pipeline.commercial_preflight import validate,sha256

DEFAULT_OUT=Path('G:/datasets/corpora/rudra_native1080_expansion_20261019_v3')
read,write,require=base_gate.read,base_gate.write,base_gate.require

def check_ready(root=DEFAULT_OUT):
    root=Path(root);ready=base_gate.check_ready(root)
    p=read(root/'plan.json');summary=read(root/'preparation_summary.json')
    require(ready['rows']==p['expected_total_pairs']==summary['total_pairs'],'Candidate row-count contract changed')
    require(summary['new_pairs']==p['expected_new_pairs'] and summary['new_resolution']==[1920,1080],'Unexpected expansion contract')
    require(summary['by_split']==ready['by_split'],'Split contract changed')
    require(read(root/'loader_smoke.json')['count']==summary['new_pairs'],'Actual-loader coverage incomplete')
    require(read(root/'similarity_review.json')['cross_split_flags']==[],'Similarity flags remain unresolved')
    for name,digest in read(root/'review_status.json').get('additional_visual_evidence',{}).items():
        require((root/name).is_file() and sha256(root/name)==digest,'Visual evidence changed: '+name)
    return ready

def finalize(root,reviewer,notes_file,accepted):
    root=Path(root)
    require(accepted,'Accept visual review only after inspecting the saved images')
    require(reviewer and reviewer.strip(),'Reviewer identity required')
    notes=Path(notes_file).read_text(encoding='utf-8-sig').strip();require(len(notes)>=100,'Record observations and limits')
    for name in base_gate.ARTIFACTS:
        if name not in ('review_status.json','visual_review_notes.md'):require((root/name).is_file(),'Preparation incomplete: '+name)
    p=read(root/'plan.json');summary=read(root/'preparation_summary.json')
    require(summary['stage']=='numeric_checks_passed_visual_review_pending','Numeric preparation is incomplete')
    base=Path(p['base_manifest']);require(sha256(base)==p['base_manifest_sha256'],'Base manifest changed')
    old=[json.loads(x) for x in base.read_text().splitlines() if x.strip()]
    combined=[json.loads(x) for x in (root/'combined_candidate_manifest.jsonl').read_text().splitlines() if x.strip()]
    require(len(old)==3079 and len(combined)==3231 and combined[:len(old)]==old,'Base rows changed or candidate incomplete')
    require(all(r['split']=='train' and (r['width'],r['height'])==(1920,1080) for r in combined[len(old):]),'Unexpected addition split/resolution')
    require(summary['new_pairs']==152 and summary['by_split']=={'train':2720,'val':251,'test':260},'Unexpected counts/splits')
    require(read(root/'loader_smoke.json')['count']==152,'Loader coverage incomplete')
    require(read(root/'similarity_review.json')['cross_split_flags']==[],'Similarity flags unresolved')
    base_gate.dependencies(read(root/'commercial_review.json'))
    result=validate(root/'combined_candidate_manifest.jsonl',root/'commercial_review.json',progress=lambda x:print(x,flush=True))
    require(result['manifest_sha256']==read(root/'preflight_result.json')['manifest_sha256'],'Preflight is stale')
    (root/'visual_review_notes.md').write_text(notes+'\n',encoding='utf-8')
    write(root/'review_status.json',dict(stage=base_gate.STAGE,manifest_sha256=result['manifest_sha256'],visual_review_accepted=True,
        reviewed_by=reviewer,reviewed_utc=datetime.now(timezone.utc).isoformat(),review_scope='All new SDR overview previews and representative HDR/full-resolution diagnostics; not calibrated HDR-display certification',
        notes_sha256=sha256(root/'visual_review_notes.md'),preview_hashes={n:sha256(root/n) for n in ('contact_sheet.jpg','degraded_contact_sheet.jpg')},
        additional_visual_evidence={str(f.relative_to(root)):sha256(f) for f in list(root.glob('preview_*.jpg'))+list((root/'visual_diagnostics').glob('*')) if f.is_file()},training_started=False,release_authorized=False))
    ready=dict(stage=base_gate.STAGE,corpus=str(root.resolve()),rows=len(combined),by_split=summary['by_split'],
        artifacts={n:sha256(root/n) for n in base_gate.ARTIFACTS},training_started=False,release_authorized=False)
    write(root/'ready.json',ready);return check_ready(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','finalize']);p.add_argument('--corpus',type=Path,default=DEFAULT_OUT)
    p.add_argument('--reviewer');p.add_argument('--notes-file',type=Path);p.add_argument('--accept-visual-review',action='store_true');a=p.parse_args()
    try:
        if a.action=='finalize':
            require(a.reviewer and a.notes_file,'Reviewer and notes file required')
            r=finalize(a.corpus,a.reviewer,a.notes_file,a.accept_visual_review)
        else:r=check_ready(a.corpus)
        print(json.dumps({k:r[k] for k in ('stage','rows','by_split','release_authorized')},indent=2))
    except (ValueError,OSError,KeyError) as e:
        print('NOT READY: '+str(e),file=sys.stderr);raise SystemExit(2)
