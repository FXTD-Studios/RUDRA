"""tools/build_real_partial_corpus.py: evaluation scenes never reach training."""
import json

from tools.build_real_partial_corpus import excluded_scenes, pick_frames


def test_every_evaluation_scene_is_refused(tmp_path):
    fresh = tmp_path / "report.json"
    fresh.write_text(json.dumps({"rows": [{"case_id": "a", "scene_id": "S1"}, {"case_id": "b", "scene_id": "S2"}]}))
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps({"rows": [{"id": "x", "scene_id": "S3"}], "selection": {"n": 1}}))
    refused = excluded_scenes([fresh, holdout])
    assert refused == {"S1", "S2", "S3"}
    rows = [{"asset_id": f"{s}_f{i}", "scene_id": s, "frame_index": i} for s in ("S1", "S2", "S3", "T1", "T2") for i in range(10)]
    picked, skipped = pick_frames(rows, 3, refused)
    assert {r["scene_id"] for r in picked} == {"T1", "T2"}
    assert skipped["evaluation scene"] == 30
    assert [r["frame_index"] for r in picked if r["scene_id"] == "T1"] == [0, 4, 9]   # spread through the scene


def test_frame_choice_is_deterministic_and_keeps_short_scenes(tmp_path):
    rows = [{"asset_id": f"T_f{i}", "scene_id": "T", "frame_index": i} for i in (5, 1, 3)]
    a, _ = pick_frames(rows, 6, set())
    b, _ = pick_frames(list(reversed(rows)), 6, set())
    assert [r["asset_id"] for r in a] == [r["asset_id"] for r in b] == ["T_f1", "T_f3", "T_f5"]
