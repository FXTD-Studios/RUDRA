"""Commercial data guards (17 Oct 2026): quarantine, manifest tiers, organizer."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import build_commercial_manifest as bcm, organize_datasets as org  # noqa: E402
from pipeline.licences import classify  # noqa: E402


def row(aid, scene, split="train", path="G:/datasets/corpora/x/hdr/a.png", **kw):
    return {"asset_id": aid, "scene_id": scene, "split": split, "hdr_path": path, **kw}


def test_quarantine_beats_any_keyword():
    r = row("a", "fxtd_x", path="G:\\datasets\\_noncommercial\\sources\\polyhaven\\a.exr")
    assert classify(r)["source"] == "quarantine" and not classify(r)["commercial_ok"]


def test_strict_keeps_only_polyhaven_and_fxtd():
    rows = [row("a", "fxtd_lab_1"), row("b", "polyhaven_sky", path="corpus_v4c/polyhaven/b.png"),
            row("c", "E:/source_hdr/HdM-HFR-2017_Color-Graded/Bar-Scene"),
            row("d", "nf_sparks_1"), row("e", "fxtd_lab_2", path="G:/datasets/_noncommercial/x.png")]
    kept, dropped = bcm.select(rows, "strict")
    assert [r["asset_id"] for r in kept] == ["a", "b"]
    assert dropped["hdm"] == 1 and dropped["netflix_sparks"] == 1 and dropped["quarantined_path"] == 1
    kept, _ = bcm.select(rows, "netflix_pending")
    assert {r["asset_id"]: r["rights_status"] for r in kept} == {"a": "clear", "b": "clear", "d": "pending_counsel"}


def test_split_straddle_is_refused():
    with pytest.raises(SystemExit):
        bcm.select([row("a", "fxtd_s"), row("b", "fxtd_s", split="val")], "strict")


def test_organizer_moves_and_restores(tmp_path):
    for rel, *_ in org.NON_COMMERCIAL[:2]:
        (tmp_path / rel).mkdir(parents=True)
        (tmp_path / rel / "f.txt").write_text("x")
    assert org.main(["--root", str(tmp_path)]) == 0                       # dry run moves nothing
    assert (tmp_path / org.NON_COMMERCIAL[0][0]).exists()
    assert org.main(["--root", str(tmp_path), "--apply"]) == 0
    assert (tmp_path / "_noncommercial" / org.NON_COMMERCIAL[0][0] / "f.txt").exists()
    assert not (tmp_path / org.NON_COMMERCIAL[0][0]).exists()
    assert "_noncommercial/sources/stuttgart_hdr_2014" in (tmp_path / "DATASETS.md").read_text()
    assert org.main(["--root", str(tmp_path), "--restore"]) == 0
    assert (tmp_path / org.NON_COMMERCIAL[0][0] / "f.txt").exists()
