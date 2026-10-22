"""Finish a source receipt after acquisition/review; does not authorize merging."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.collect_next_hdr_sources import ROOT,dump,sha

def main():
    si=json.loads((ROOT/'sihdr/pixel_review.json').read_text())
    tos=json.loads((ROOT/'tears_of_steel/sequence_review.json').read_text())
    if len(si['references'])!=181 or si['input_count']!=362 or len(tos['frames'])!=166:raise ValueError('Incomplete reviewed media')
    assets=[];excluded=[]
    for group in ('polyhaven','polyhaven_novel'):
        rows=json.loads((ROOT/group/'verified_files.json').read_text())
        overlaps=json.loads((ROOT/group/'overlap_review.json').read_text())['assets']
        if len(rows)!=(15 if group=='polyhaven' else 8):raise ValueError('Incomplete environment pack')
        for row in rows:
            if sha(row['path'])!=row['sha256']:raise ValueError('Environment changed')
        assets.extend(rows);excluded.extend(r['asset_id'] for r in overlaps if r['blocked_reasons'])
    scenes=json.loads((ROOT/'blender/scene_inspection.json').read_text())
    if len(scenes)!=2 or any(s['missing_images'] or s['missing_libraries'] for s in scenes):raise ValueError('Incomplete Blender dependencies')
    visual_path=ROOT/'initial_visual_review.json'
    evidence={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and ('licence_evidence' in p.parts or p.name=='ATTRIBUTION_AND_SOURCE_USE.md')}
    summary=dict(stage='source_pack_downloaded_and_numerically_verified',completed_unix=time.time(),
        photographic_hdr_references=181,simulated_sdr_inputs=362,hdr_reference_resolution=[1888,1280],
        hdr_panoramas_8k=len(assets),panorama_bytes=sum(a['expected_bytes'] for a in assets),
        panorama_training_candidates=len(assets)-len(excluded),historical_overlap_exclusions=excluded,
        verified_tos_frames=166,continuous_exr_frames=144,continuous_sequences=3,sequence_lengths=[48,48,48],
        sparse_tos_stills=22,excluded_tos_frames=48,continuous_exr_resolution=[1920,800],blender_source_assets=2,
        sihdr_publisher_checks=si['publisher_checks'],evidence_sha256=evidence,
        sihdr_reference_notes=json.loads((ROOT/'sihdr/reference_admission_notes.json').read_text()),
        visual_review_file=str(visual_path),visual_review_sha256=sha(visual_path),
        training_pairs_added=0,training_enabled=False,active_training_modified=False,release_authorized=False,
        remaining=['Full-resolution paired/detail and motion review beyond the recorded overview/sample inspections',
        'Colour primaries, exposure and target-objective confirmation',
        'Cross-provider perceptual duplicates and capture/title split freeze',
        'Native-resolution perspective/render preparation and loader validation before a new merge receipt',
        'Permitted real-camera fire/spark/moving-specular footage remains a gap'])
    dump(ROOT/'collection_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('evidence_sha256','sihdr_publisher_checks')},indent=2))

if __name__=='__main__':main()
