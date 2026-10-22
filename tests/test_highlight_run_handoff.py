"""Readiness must fail closed; run-local reviews retain valid source references."""
import json
from pathlib import Path
import pytest
from pipeline import fire_reflection_readiness as gate
from training.run_highlight_commercial import normalized_review,prepare_combined

def test_missing_review_blocks_before_creating_run_data(tmp_path):
    run=tmp_path/'run';run.mkdir()
    with pytest.raises(ValueError,match='Candidate not ready'):
        prepare_combined(run,tmp_path/'unfinished')
    assert not (run/'data').exists()

def test_finalize_requires_actual_acceptance_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError,match='after inspecting'):
        gate.finalize(tmp_path,'reviewer',tmp_path/'missing.md',False)
    assert not (tmp_path/'ready.json').exists()

def ready_fixture(root):
    root.mkdir()
    for name in gate.ARTIFACTS:(root/name).write_text('fixture')
    evidence=root/'evidence.json';evidence.write_text('{}')
    dependency=root/'decoder.py';dependency.write_text('# frozen fixture\n')
    manifest_hash=gate.sha256(root/'combined_candidate_manifest.jsonl')
    review=dict(manifest_sha256=manifest_hash,scope='internal-pilot',release_authorized=False,
        preparation_dependencies={str(dependency):gate.sha256(dependency)},decoder_dependencies={},
        sources={'fixture':{'evidence_path':'evidence.json','evidence_sha256':gate.sha256(evidence)}})
    (root/'commercial_review.json').write_text(json.dumps(review))
    (root/'review_status.json').write_text(json.dumps(dict(manifest_sha256=manifest_hash,visual_review_accepted=True,reviewed_by='test fixture')))
    gate.write(root/'ready.json',dict(stage=gate.STAGE,release_authorized=False,artifacts={n:gate.sha256(root/n) for n in gate.ARTIFACTS}))
    return dependency

def test_stale_manifest_is_rejected(tmp_path):
    root=tmp_path/'candidate';ready_fixture(root);gate.check_ready(root)
    (root/'combined_candidate_manifest.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='artifact changed'):gate.check_ready(root)

def test_changed_dependency_is_rejected_even_with_unchanged_manifest(tmp_path):
    root=tmp_path/'candidate';dep=ready_fixture(root)
    dep.write_text('# silently changed interpretation')
    with pytest.raises(ValueError,match='dependency changed'):gate.check_ready(root)

def test_changed_license_evidence_is_rejected(tmp_path):
    root=tmp_path/'candidate';ready_fixture(root)
    (root/'evidence.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='approval evidence changed'):gate.check_ready(root)

def test_run_local_review_resolves_relative_evidence_and_preserves_original(tmp_path):
    root=tmp_path/'candidate';ready_fixture(root)
    before=(root/'commercial_review.json').read_bytes()
    relocated=normalized_review(root)
    assert Path(relocated['sources']['fixture']['evidence_path'])==root/'evidence.json'
    assert (root/'commercial_review.json').read_bytes()==before

def test_run_data_is_reusable_and_relocated_evidence_remains_readable(tmp_path):
    root=tmp_path/'candidate';ready_fixture(root);run=tmp_path/'run';run.mkdir()
    data=prepare_combined(run,root)
    review=gate.read(data/'review.json')
    assert Path(review['sources']['fixture']['evidence_path']).read_text()=='{}'
    assert prepare_combined(run,root)==data

def test_changed_run_snapshot_is_never_silently_replaced(tmp_path):
    root=tmp_path/'candidate';ready_fixture(root);run=tmp_path/'run';run.mkdir()
    data=prepare_combined(run,root);(data/'manifest.jsonl').write_text('modified run copy')
    with pytest.raises(ValueError,match='Run-local artifact changed'):prepare_combined(run,root)
