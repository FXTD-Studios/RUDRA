"""Protect frozen data, failed readiness, and native panorama geometry."""
import json
from pathlib import Path
import numpy as np
import pytest
from pipeline import next_hdr_readiness as gate
from pipeline.prepare_next_hdr_training import perspective
from training.run_next_hdr_commercial import prepare_combined

def test_missing_candidate_blocks_before_run_data(tmp_path):
    run=tmp_path/'run';run.mkdir()
    with pytest.raises(ValueError,match='Candidate not ready'):prepare_combined(run,tmp_path/'unfinished')
    assert not (run/'data').exists()

def test_visual_acceptance_cannot_be_inferred_from_numeric_success(tmp_path):
    with pytest.raises(ValueError,match='after inspecting'):gate.finalize(tmp_path,'test',tmp_path/'missing.md',False)
    assert not (tmp_path/'ready.json').exists()

def fixture(root):
    root.mkdir()
    for name in gate.base_gate.ARTIFACTS:(root/name).write_text('fixture')
    evidence=root/'evidence.json';evidence.write_text('{}')
    dep=root/'decoder.py';dep.write_text('# test fixture')
    mh=gate.sha256(root/'combined_candidate_manifest.jsonl')
    data={'plan.json':dict(expected_total_pairs=3231,expected_new_pairs=152),
        'preparation_summary.json':dict(total_pairs=3231,new_pairs=152,new_resolution=[1920,1080],by_split={'train':2720,'val':251,'test':260}),
        'loader_smoke.json':dict(count=152),'similarity_review.json':dict(cross_split_flags=[]),
        'commercial_review.json':dict(manifest_sha256=mh,scope='internal-pilot',release_authorized=False,preparation_dependencies={str(dep):gate.sha256(dep)},decoder_dependencies={},sources={'test':dict(evidence_path='evidence.json',evidence_sha256=gate.sha256(evidence))}),
        'review_status.json':dict(manifest_sha256=mh,visual_review_accepted=True,reviewed_by='test fixture')}
    for name,value in data.items():(root/name).write_text(json.dumps(value))
    bind(root);return dep

def bind(root):
    gate.write(root/'ready.json',dict(stage=gate.base_gate.STAGE,release_authorized=False,rows=3231,by_split={'train':2720,'val':251,'test':260},artifacts={n:gate.sha256(root/n) for n in gate.base_gate.ARTIFACTS}))

def test_stale_media_manifest_rejected(tmp_path):
    root=tmp_path/'candidate';fixture(root);gate.check_ready(root)
    (root/'combined_candidate_manifest.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='artifact changed'):gate.check_ready(root)

def test_changed_source_evidence_rejected(tmp_path):
    root=tmp_path/'candidate';fixture(root)
    (root/'evidence.json').write_text('changed')
    with pytest.raises(ValueError,match='approval evidence changed'):gate.check_ready(root)

def test_incomplete_loader_check_rejected_even_if_artifacts_rebound(tmp_path):
    root=tmp_path/'candidate';fixture(root)
    gate.write(root/'loader_smoke.json',dict(count=151));bind(root)
    with pytest.raises(ValueError,match='loader coverage'):gate.check_ready(root)

def test_reviewed_diagnostic_change_invalidates_ready(tmp_path):
    root=tmp_path/'candidate';fixture(root)
    preview=root/'diagnostic.jpg';preview.write_bytes(b'reviewed fixture')
    review=gate.read(root/'review_status.json');review['additional_visual_evidence']={'diagnostic.jpg':gate.sha256(preview)}
    gate.write(root/'review_status.json',review);bind(root);gate.check_ready(root)
    preview.write_bytes(b'replaced after review')
    with pytest.raises(ValueError,match='Visual evidence changed'):gate.check_ready(root)

def test_run_snapshot_preserves_relative_source_approval(tmp_path):
    root=tmp_path/'candidate';fixture(root);run=tmp_path/'run';run.mkdir()
    data=prepare_combined(run,root)
    review=gate.read(data/'review.json')
    assert Path(review['sources']['test']['evidence_path']).read_text()=='{}'
    (data/'manifest.jsonl').write_text('changed run snapshot')
    with pytest.raises(ValueError,match='Run-local artifact changed'):prepare_combined(run,root)

def test_projection_wraps_longitude_without_black_seam():
    pano=np.ones((64,128,3),np.float32)*np.array([.2,1.,3.],np.float32)
    image=perspective(pano,180,0,width=80,height=45)
    assert image.shape==(45,80,3)
    np.testing.assert_allclose(image,np.broadcast_to([.2,1.,3.],image.shape),rtol=1e-6)

def test_projection_full_turn_is_identical():
    rng=np.random.default_rng(4);pano=rng.random((64,128,3),dtype=np.float32)
    np.testing.assert_allclose(perspective(pano,12,-10,width=80,height=45),perspective(pano,372,-10,width=80,height=45),atol=1e-6)

def test_trainer_detection_ignores_own_launcher_and_prepare_only(monkeypatch):
    from types import SimpleNamespace
    import os,psutil
    from training.run_next_hdr_commercial import active_trainers
    parent=os.getpid()+100000;other=parent+1;preparer=parent+2
    monkeypatch.setattr(psutil,'Process',lambda:SimpleNamespace(parents=lambda:[SimpleNamespace(pid=parent)]))
    def proc(pid,script,*extra):return SimpleNamespace(pid=pid,info=dict(name='python.exe',cmdline=['python.exe',script,*extra]))
    monkeypatch.setattr(psutil,'process_iter',lambda attrs:[proc(parent,'run_next_hdr_commercial.py'),proc(other,'run_highlight_commercial.py'),proc(preparer,'run_next_hdr_commercial.py','--prepare-only')])
    assert active_trainers()==[other]
