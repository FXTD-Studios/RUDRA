"""Bind an actual visual review to the completed v2 candidate; never trains.

Do not use finalize until a reviewer has inspected the full saved contact sheets
and representative frames. Neither numeric success nor this command is release
approval. check is read-only and rejects missing/stale evidence.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from pipeline.commercial_preflight import sha256, validate

DEFAULT_OUT=Path('G:/datasets/corpora/rudra_fire_reflection_expansion_20261018_v2')
STAGE='prepared_for_controlled_next_run_pilot'
ARTIFACTS=('combined_candidate_manifest.jsonl','commercial_review.json','preflight_result.json',
    'preparation_summary.json','loader_smoke.json','similarity_review.json','source_evidence.json',
    'plan.json','_ingest_config.json','contact_sheet.jpg','degraded_contact_sheet.jpg',
    'review_status.json','visual_review_notes.md')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)

def require(ok,message):
    if not ok:raise ValueError(message)

def dependencies(review):
    pinned=review.get('preparation_dependencies',{})
    require(bool(pinned),'Missing preparation dependency hashes')
    for path,digest in pinned.items():
        require(Path(path).is_file() and sha256(Path(path))==digest,'Preparation dependency changed: '+path)
    for path,digest in review.get('decoder_dependencies',{}).items():
        require(Path(path).is_file() and sha256(Path(path))==digest,'Decoder dependency changed: '+path)

def check_ready(root=DEFAULT_OUT):
    root=Path(root);ready_path=root/'ready.json'
    require(ready_path.is_file(),'Candidate not ready: finish preparation, inspect previews and record the visual review first.')
    ready=read(ready_path)
    require(ready.get('stage')==STAGE and ready.get('release_authorized') is False,'Unsupported readiness scope')
    require(set(ready.get('artifacts',{}))==set(ARTIFACTS),'Incomplete readiness artifact binding')
    for name,digest in ready['artifacts'].items():
        p=root/name
        require(p.is_file() and sha256(p)==digest,'Readiness artifact changed or missing: '+name)
    review=read(root/'commercial_review.json');visual=read(root/'review_status.json')
    manifest_hash=sha256(root/'combined_candidate_manifest.jsonl')
    require(review.get('manifest_sha256')==visual.get('manifest_sha256')==manifest_hash,'Manifest/review mismatch')
    require(visual.get('visual_review_accepted') is True and visual.get('reviewed_by'),'Visual review is not recorded')
    require(review.get('scope')=='internal-pilot' and review.get('release_authorized') is False,'Unsupported commercial evidence scope')
    dependencies(review)
    # Source approvals may point at older source-evidence files, kept on G:.
    for source in review['sources'].values():
        p=root/source['evidence_path']
        require(p.is_file() and sha256(p)==source['evidence_sha256'],'Source approval evidence changed: '+str(p))
    return ready

def finalize(root,reviewer,notes_file,accepted):
    root=Path(root)
    require(accepted,'Explicit --accept-visual-review is required after inspecting the previews')
    require(bool(reviewer.strip()),'Reviewer identity is required')
    notes=Path(notes_file).read_text(encoding='utf-8-sig').strip()
    require(len(notes)>=40,'Record substantive observations and limitations in the review notes')
    for name in ARTIFACTS:
        if name not in ('review_status.json','visual_review_notes.md'):
            require((root/name).is_file(),'Preparation incomplete; missing '+name)
    summary=read(root/'preparation_summary.json')
    require(summary.get('stage')=='numeric_checks_passed_visual_review_pending','Numerical preparation is incomplete')
    require(summary.get('total_pairs')==3079 and summary.get('new_pairs')==312,'Unexpected candidate counts')
    require(summary.get('by_split')=={'train':2568,'val':251,'test':260},'Unexpected splits')
    require(read(root/'loader_smoke.json').get('count')==26,'Actual-loader checks are incomplete')
    require(read(root/'similarity_review.json').get('cross_split_flags')==[],'Similarity flags remain unresolved')
    plan=read(root/'plan.json');base=Path(plan['base_manifest'])
    require(sha256(base)==plan['base_manifest_sha256'],'Prior candidate manifest changed')
    old=[json.loads(x) for x in base.read_text().splitlines() if x.strip()]
    all_rows=[json.loads(x) for x in (root/'combined_candidate_manifest.jsonl').read_text().splitlines() if x.strip()]
    require(len(old)==2767 and len(all_rows)==3079 and all_rows[:len(old)]==old,'Prior candidate rows changed')
    require(all(r['split']=='train' for r in all_rows[len(old):]),'New rows entered held-out sets')
    review=read(root/'commercial_review.json');dependencies(review)
    result=validate(root/'combined_candidate_manifest.jsonl',root/'commercial_review.json',progress=lambda x:print(x,flush=True))
    require(result['manifest_sha256']==read(root/'preflight_result.json')['manifest_sha256'],'Preflight is stale')
    # Never manufacture these files just to remove the gate. This is the review record.
    (root/'visual_review_notes.md').write_text(notes+'\n',encoding='utf-8')
    write(root/'review_status.json',dict(stage=STAGE,manifest_sha256=result['manifest_sha256'],visual_review_accepted=True,
        reviewed_by=reviewer,reviewed_utc=datetime.now(timezone.utc).isoformat(),review_scope='Sampled full-batch SDR previews and representative paired frames; not calibrated HDR/temporal certification',
        preview_hashes={name:sha256(root/name) for name in ['contact_sheet.jpg','degraded_contact_sheet.jpg']},
        notes_sha256=sha256(root/'visual_review_notes.md'),training_started=False,release_authorized=False))
    ready=dict(stage=STAGE,corpus=str(root.resolve()),rows=3079,by_split=summary['by_split'],
        artifacts={name:sha256(root/name) for name in ARTIFACTS},training_started=False,release_authorized=False)
    write(root/'ready.json',ready);check_ready(root);return ready

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['check','finalize'])
    p.add_argument('--corpus',type=Path,default=DEFAULT_OUT);p.add_argument('--reviewer');p.add_argument('--notes-file',type=Path)
    p.add_argument('--accept-visual-review',action='store_true');a=p.parse_args()
    try:
        if a.action=='finalize':
            require(a.reviewer and a.notes_file,'--reviewer and --notes-file are required')
            ready=finalize(a.corpus,a.reviewer,a.notes_file,a.accept_visual_review)
        else:ready=check_ready(a.corpus)
        print(json.dumps({k:ready[k] for k in ['stage','rows','by_split','release_authorized']},indent=2))
    except (ValueError,OSError,KeyError) as e:
        print('NOT READY: '+str(e),file=sys.stderr);return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
