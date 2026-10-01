"""Task 1.2: source-curve labels on a merged manifest, and real-SDR rows in corpus_ev_of."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.build_source_curve_manifest import SOURCE_CURVES, main


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _rendered(n_scenes=4, curve="hable"):
    return [{"asset_id": f"r{s}_{i}", "scene_id": f"rs{s}", "split": "test" if s == 0 else "train",
             "sdr_path": f"G:/v4c/sdr/r{s}_{i}.png", "hdr_path": f"G:/v4c/hdr/r{s}_{i}.png",
             "sdr_curve": curve, "tonemap_ev": 0.0} for s in range(n_scenes) for i in range(2)]


def _real():
    return [{"asset_id": f"nf{i}", "scene_id": "nf_meridian_1" if i < 2 else "nf_nocturne_1",
             "split": "test" if i < 2 else "train", "sdr_path": f"G:/nf/sdr/{i}.png",
             "hdr_path": f"G:/nf/hdr/{i}.png", "sdr_kind": "real", "tonemap_ev": None} for i in range(4)]


def _compare(tmp_path: Path, items: list[dict]) -> Path:
    p = tmp_path / "compare.json"
    p.write_text(json.dumps({"items": items}), encoding="utf-8")
    return p


def test_labels_every_row(tmp_path):
    out = tmp_path / "mix" / "m.jsonl"
    cs = _compare(tmp_path, [{"kind": "still", "asset_id": "r0_0", "scene_id": "rs0",
                              "sdr_path": "x", "hdr_path": "y"}])
    main(["--rendered", str(_jsonl(tmp_path / "r.jsonl", _rendered())),
          "--real", str(_jsonl(tmp_path / "n.jsonl", _real())), "--compare-set", str(cs), "--out", str(out)])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert {r["source_curve"] for r in rows if r["sdr_kind"] == "rendered"} == {"hable"}
    assert {r["source_curve"] for r in rows if r["sdr_kind"] == "real"} == {"unknown"}
    assert all(r["source_curve"] in SOURCE_CURVES for r in rows)
    assert (out.parent / "m_report.json").exists()


def test_refuses_compare_set_in_training(tmp_path):
    cs = _compare(tmp_path, [{"kind": "still", "asset_id": "r1_0", "scene_id": "rs1",
                              "sdr_path": "x", "hdr_path": "y"}])  # rs1 is a train scene
    with pytest.raises(SystemExit, match="comparison-set"):
        main(["--rendered", str(_jsonl(tmp_path / "r.jsonl", _rendered())), "--compare-set", str(cs),
              "--out", str(tmp_path / "o.jsonl")])


def test_refuses_missing_curve_without_assumption(tmp_path):
    rows = [{k: v for k, v in r.items() if k != "sdr_curve"} for r in _rendered()]
    with pytest.raises(SystemExit, match="no sdr_curve"):
        main(["--rendered", str(_jsonl(tmp_path / "r.jsonl", rows)), "--out", str(tmp_path / "o.jsonl")])
    main(["--rendered", str(_jsonl(tmp_path / "r2.jsonl", rows)), "--assume-curve", "aces",
          "--out", str(tmp_path / "o2.jsonl")])


def test_refuses_scene_in_two_splits(tmp_path):
    rows = _rendered()
    rows[-1]["split"] = "val"
    with pytest.raises(SystemExit, match="two splits"):
        main(["--rendered", str(_jsonl(tmp_path / "r.jsonl", rows)), "--out", str(tmp_path / "o.jsonl")])


def test_corpus_ev_ignores_real_rows(tmp_path):
    pytest.importorskip("torch")
    from training.sdr2hdr_dataset import corpus_ev_of

    rows = [{"tonemap_ev": 0.0, "sdr_kind": "rendered"}] * 3 + [{"tonemap_ev": None, "sdr_kind": "real"}] * 2
    assert corpus_ev_of(_jsonl(tmp_path / "m.jsonl", rows)) == 0.0
    # a real row without the flag is still refused: the flag is the contract
    with pytest.raises(ValueError):
        corpus_ev_of(_jsonl(tmp_path / "m2.jsonl", rows[:3] + [{"sdr_path": "x"}]))


def test_drop_frozen_removes_only_the_conflicting_rows(tmp_path):
    cs = _compare(tmp_path, [{"kind": "clip", "clip_id": "c1", "scene_id": "rs1",
                              "sdr_frames": [], "hdr_frames": []}])  # rs1 trains in the rendered corpus
    out = tmp_path / "o.jsonl"
    main(["--rendered", str(_jsonl(tmp_path / "r.jsonl", _rendered())), "--compare-set", str(cs),
          "--drop-frozen", "--out", str(out)])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert not any(r["scene_id"] == "rs1" for r in rows)
    assert {r["scene_id"] for r in rows} == {"rs0", "rs2", "rs3"}
    report = json.loads((tmp_path / "o_report.json").read_text())
    assert report["dropped_for_compare_set"]["records"] == 2
