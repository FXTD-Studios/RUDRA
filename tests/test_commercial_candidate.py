import json
from pathlib import Path

import pytest

from pipeline.prepare_commercial_candidate import POLICY, hash_media, prepare, select


def fixture(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    rows = []
    for i, (title, (sid, slug, _)) in enumerate(POLICY.items()):
        r = dict(asset_id=f"a{i}", scene_id=f"nf_{slug}_1-20", shot="1-20", title=title,
                 sdr_kind="real", licence_source=sid, commercial_ok=True, split="train")
        for kind in ("sdr", "hdr"):
            p = source / f"{kind}{i}.png"; p.write_bytes(f"{kind}{i}".encode())
            r[f"{kind}_path"] = str(p)
            r[f"{kind}_url"] = f"https://download.opencontent.netflix.com.s3.amazonaws.com/aom_test_materials/{kind.upper()}/{slug}/1-20.zip"
        rows.append(r)
    (source / "_ingest_config.json").write_text('{"storage":{"mode":"log2_extended"}}')
    manifest = source / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows, manifest


def test_selection_is_order_independent_and_never_grants_rights(tmp_path):
    rows, _ = fixture(tmp_path)
    a, _ = select(rows); b, _ = select(list(reversed(rows)))
    assert a == b and all(not r["commercial_ok"] for r in a)
    assert {r["split"] for r in a} == {"train", "val", "test"}
    assert all(r["commercial_ok"] for r in rows)


@pytest.mark.parametrize("change,match", [("duplicate", "duplicate"), ("origin", "origin"), ("scene", "scene"), ("missing", "missing")])
def test_bad_provenance_rejected(tmp_path, change, match):
    rows, _ = fixture(tmp_path)
    if change == "duplicate": rows.append(dict(rows[0]))
    if change == "origin": rows[0]["hdr_url"] = "https://other.invalid/data.zip"
    if change == "scene": rows[0]["scene_id"] = "wrong"
    if change == "missing": rows[0]["hdr_path"] += ".missing"
    with pytest.raises(ValueError, match=match): select(rows)


def test_restricted_and_unknown_excluded(tmp_path):
    rows, _ = fixture(tmp_path)
    rows[0]["commercial_ok"] = False
    rows[1]["title"] = "HdM"
    selected, reasons = select(rows)
    assert len(selected) == 3 and sum(reasons.values()) == 2


def test_hashes_find_cross_split_duplicates(tmp_path):
    rows, _ = fixture(tmp_path)
    selected, _ = select(rows)
    Path(selected[-1]["hdr_path"]).write_bytes(Path(selected[0]["hdr_path"]).read_bytes())
    duplicates = hash_media(selected)
    assert any(d["cross_split"] for d in duplicates)
    assert all(len(r["hdr_sha256"]) == 64 for r in selected)


def test_plan_and_build_preserve_source_and_reject_overwrite(tmp_path):
    _, manifest = fixture(tmp_path)
    before = manifest.read_bytes()
    out = tmp_path / "candidate"
    assert prepare(manifest, out, plan=True)["rows"] == 5 and not out.exists()
    result = prepare(manifest, out, hash_files=True)
    assert not result["training_ready"] and result["hash_scope"] == "all_media"
    assert manifest.read_bytes() == before
    with pytest.raises(ValueError, match="already exists"): prepare(manifest, out)
    with pytest.raises(ValueError, match="outside source"): prepare(manifest, manifest.parent / "unsafe")


def test_hash_budget_cache_and_mutation_detection(tmp_path, monkeypatch):
    rows, _ = fixture(tmp_path)
    selected, _ = select(rows)
    journal = tmp_path / "hashes.jsonl"
    hash_media(selected, budget_seconds=0, journal=journal)
    saved = journal.read_text()
    fresh, _ = select(rows)
    monkeypatch.setattr("pipeline.prepare_commercial_candidate.sha256", lambda p: pytest.fail("cache should suffice"))
    hash_media(fresh, budget_seconds=1e-12, journal=journal)
    assert all(r.get("hdr_sha256") for r in fresh) and journal.read_text() == saved
    p = Path(rows[0]["hdr_path"]); p.write_bytes(b"different size")
    fresh, _ = select(rows)
    hash_media(fresh, budget_seconds=1e-12, journal=journal)
    assert "hdr_sha256" not in fresh[0]


def test_real_only_ev_is_explicit_and_cannot_override_rendered_exposure():
    from training.sdr2hdr_dataset import corpus_ev_of
    from rudra.sdr2hdr import LEGACY_CORPUS_EV
    rows = [{"sdr_kind": "real", "tonemap_ev": None}]
    assert corpus_ev_of("unused", rows) == LEGACY_CORPUS_EV
    assert corpus_ev_of("unused", rows, real_only_ev=0.0) == 0.0
    assert rows[0]["tonemap_ev"] is None
    for invalid in ([], [{"tonemap_ev": -1}], rows + [{"tonemap_ev": 0}]):
        with pytest.raises(ValueError, match="all-real"):
            corpus_ev_of("unused", invalid, real_only_ev=0.0)
    with pytest.raises(ValueError, match="finite"):
        corpus_ev_of("unused", rows, real_only_ev=float("nan"))


def test_real_only_picker_recipe_runs_on_tiny_synthetic_fixture(tmp_path, monkeypatch):
    import sys
    import cv2
    import numpy as np
    import torch
    from pipeline.hdr_io import HDRStorage, encode_hdr_u16
    from training.train_sdr2hdr import parse_args, train
    storage = HDRStorage(mode="log2_extended")
    (tmp_path / "_ingest_config.json").write_text(json.dumps({"storage": storage.as_dict()}))
    rng = np.random.default_rng(23)
    rows = []
    for i in range(6):
        rgb = rng.uniform(0.001, 10, (40, 48, 3)).astype(np.float32)
        hdr, _ = encode_hdr_u16(rgb, storage)
        sdr = ((rgb / (1 + rgb)) ** (1 / 2.2) * 65535).astype(np.uint16)
        sp, hp = tmp_path / f"s{i}.png", tmp_path / f"h{i}.png"
        assert cv2.imwrite(str(sp), sdr) and cv2.imwrite(str(hp), hdr)
        rows.append(dict(asset_id=str(i), scene_id=str(i), split="train" if i < 4 else "val",
                         sdr_path=str(sp), hdr_path=str(hp), sdr_kind="real", tonemap_ev=None))
    manifest = tmp_path / "synthetic.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(sys, "argv", ["train_sdr2hdr.py", "--manifest", str(manifest),
        "--output-dir", str(tmp_path / "model"), "--device", "cpu", "--workers", "0",
        "--steps", "2", "--batch-size", "2", "--crop-size", "32", "--base-channels", "8",
        "--eval-every", "2", "--save-every", "2", "--eval-batches", "0",
        "--baseline-source", "--real-corpus-ev", "0", "--exposure-push", "1",
        "--crushed-hold", "1", "--best-metric", "bench_gain", "--bench-val-frames", "1",
        "--bench-val-max-side", "32"])
    checkpoint = train(parse_args())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["config"]["corpus_ev"] == 0
    assert saved["config"]["baseline_source"] and saved["config"]["training_use"] == "research"
    assert all(torch.isfinite(v).all() for v in saved["model"].values() if torch.is_floating_point(v))
    logs = [json.loads(line) for line in (checkpoint.parent / "train.jsonl").read_text().splitlines()]
    training = [r for r in logs if "baseline_curve_counts_since_start" in r]
    assert [sum(r["baseline_curve_counts_since_start"].values()) for r in training] == [2, 4]
    assert all("clean_pu_gain_db" in r["eval"] for r in logs if "eval" in r)
    reports = [json.loads(line) for line in (checkpoint.parent / "bench_val.jsonl").read_text().splitlines()]
    assert [r["step"] for r in reports] == [0, 2]
    assert all(len(r["per_ev"]) == 2 and r["per_scene"] for r in reports)
