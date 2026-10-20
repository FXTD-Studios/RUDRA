import json
from argparse import Namespace

import pytest

from pipeline.commercial_preflight import decoder_dependencies, sha256, training_preflight, validate
from training.cp10_verdicts import main
from training.paired_gate import compare, load


def test_strict_gate_rejects_unrun(tmp_path):
    assert main(["--bench-root", str(tmp_path), "--out", str(tmp_path / "result.json"), "--strict"]) == 1


def test_pairing_rejects_dropped_frames_and_asymmetric_nan():
    a = {"a": {"m": 1.0}, "b": {"m": 2.0}}
    with pytest.raises(ValueError, match="frame IDs differ"):
        compare(a, {"a": a["a"]}, metrics=("m",), require_same_frames=True)
    with pytest.raises(ValueError, match="finite coverage"):
        compare(a, {"a": a["a"], "b": {"m": float("nan")}}, metrics=("m",), require_same_frames=True)


@pytest.mark.parametrize("ids", [["a", "a"], ["a", ""]])
def test_invalid_frame_identity(tmp_path, ids):
    p = tmp_path / "scores.csv"
    p.write_text("frame,m\n" + "".join(f"{key},1\n" for key in ids))
    with pytest.raises(ValueError, match="missing or duplicate"):
        load(p, ("m",))


def test_equal_scene_bootstrap_does_not_overweight_long_shots():
    a = {str(i): {"m": 1.0 if i < 100 else -1.0} for i in range(101)}
    b = {k: {"m": 0.0} for k in a}
    scenes = {k: "long" if int(k) < 100 else "short" for k in a}
    frame = compare(a, b, metrics=("m",))["m"]
    scene = compare(a, b, metrics=("m",), scenes=scenes)["m"]
    assert frame["ci95"][0] > 0
    assert scene["mean"] == 0 and scene["ci95"][0] < 0 and scene["scenes"] == 2
    assert compare(a, b, metrics=("m",), scenes={k: "one" for k in a})["m"]["frames"] == 0


def approved_fixture(tmp_path):
    license = tmp_path / "rights.txt"
    license.write_text("Synthetic test evidence, not a real license grant")
    rows = []
    for i, split in enumerate(("train", "val")):
        row = dict(source_id="fixture", scene_id=f"scene{i}", asset_id=f"a{i}", split=split)
        for kind in ("sdr", "hdr"):
            p = tmp_path / f"{kind}{i}.bin"; p.write_bytes(f"{kind}{i}".encode())
            row[f"{kind}_path"] = str(p.resolve()); row[f"{kind}_sha256"] = sha256(p)
        rows.append(row)
    manifest = tmp_path / "manifest.jsonl"
    evidence = tmp_path / "evidence.json"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "_ingest_config.json").write_text('{"storage":{"mode":"log2_extended"}}')
    receipt = dict(schema_version=2, manifest_sha256=sha256(manifest), reviewed_by="TEST ONLY",
                   reviewed_at="2026-10-13", sources={"fixture": dict(commercial_training=True,
                   weight_distribution=True, evidence_path=license.name, evidence_sha256=sha256(license))})
    receipt["decoder_dependencies"] = {str(p): sha256(p) for p in decoder_dependencies(rows)}
    evidence.write_text(json.dumps(receipt))
    return manifest, evidence, rows, receipt


def test_reviewed_scratch_receipt(tmp_path):
    manifest, evidence, _, _ = approved_fixture(tmp_path)
    assert validate(manifest, evidence)["initialization"] == "scratch"
    args = Namespace(mode="image", manifest=manifest, commercial_evidence=evidence, output_dir=tmp_path / "new")
    assert training_preflight(args)["rows"] == 2
    args.init_checkpoint = "research.pt"
    with pytest.raises(ValueError, match="scratch"):
        training_preflight(args)


@pytest.mark.parametrize("mutation, message", [
    ("source", "approval"), ("restricted", "research-only"), ("scene", "scene crosses"),
    ("duplicate", "content crosses"), ("media", "content hash"), ("manifest", "exact manifest"),
    ("license", "evidence hash"), ("approval", "approval"),
])
def test_commercial_preflight_rejects_invalid_evidence(tmp_path, mutation, message):
    manifest, evidence, rows, receipt = approved_fixture(tmp_path)
    if mutation == "source": rows[0].pop("source_id"); rows[0]["commercial_ok"] = True
    if mutation == "restricted": rows[0]["commercial_ok"] = False
    if mutation == "scene": rows[1]["scene_id"] = rows[0]["scene_id"]
    if mutation == "duplicate":
        for key in ("hdr_path", "hdr_sha256"): rows[1][key] = rows[0][key]
    if mutation == "media": (tmp_path / "hdr0.bin").write_bytes(b"changed")
    if mutation == "license": (tmp_path / "rights.txt").write_text("changed")
    if mutation == "approval": receipt["sources"]["fixture"]["weight_distribution"] = False
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt["manifest_sha256"] = "invalid" if mutation == "manifest" else sha256(manifest)
    evidence.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=message): validate(manifest, evidence)


def test_training_rejects_before_loading_data_or_allocating_gpu(tmp_path, monkeypatch):
    from training.train_sdr2hdr import train
    def forbidden(*args, **kwargs): pytest.fail("must reject before training work")
    monkeypatch.setattr("training.train_sdr2hdr.seed_everything", forbidden)
    with pytest.raises(ValueError, match="scratch"):
        train(Namespace(commercial_evidence="evidence.json", mode="image", init_checkpoint="v10.pt"))


def test_commercial_output_cannot_mix_with_research(tmp_path):
    manifest, evidence, _, _ = approved_fixture(tmp_path)
    with pytest.raises(ValueError, match="empty, separate"):
        training_preflight(Namespace(commercial_evidence=evidence, mode="image", manifest=manifest, output_dir=tmp_path))
    assert training_preflight(Namespace()) is None


def test_strict_scene_gate_requires_matching_scene_metadata(tmp_path):
    from training.cp10_verdicts import _paired
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("frame,scene,m\na,one,1\nb,two,2\n")
    b.write_text("frame,scene,m\na,other,0\nb,two,0\n")
    with pytest.raises(ValueError, match="scene assignments"):
        _paired(a, b, ("m",), scene_bootstrap=True)


@pytest.mark.parametrize("mutation", ["sentinel", "missing", "code", "legacy"])
def test_commercial_decode_convention_is_bound_to_review(tmp_path, mutation):
    manifest, evidence, _, receipt = approved_fixture(tmp_path)
    if mutation == "sentinel":
        (tmp_path / "_ingest_config.json").write_text('{"storage":{"mode":"pq_10000"}}')
    if mutation == "missing": receipt["decoder_dependencies"] = {}
    if mutation == "code":
        key = next(k for k in receipt["decoder_dependencies"] if k.endswith("hdr_io.py"))
        receipt["decoder_dependencies"][key] = "wrong"
    if mutation == "legacy":
        (tmp_path / "_ingest_config.json").write_text('{}')
    evidence.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="dependency|storage metadata"):
        validate(manifest, evidence)


def test_metadata_rejection_happens_before_media_hashing(tmp_path, monkeypatch):
    from pipeline import commercial_preflight as preflight
    manifest, evidence, _, _ = approved_fixture(tmp_path)
    (tmp_path / "_ingest_config.json").write_text('{"storage":{"mode":"pq_10000"}}')
    original = preflight.sha256
    def guarded(path):
        assert path.suffix != ".bin", "invalid decoder evidence should fail before media I/O"
        return original(path)
    monkeypatch.setattr(preflight, "sha256", guarded)
    with pytest.raises(ValueError, match="dependency"):
        validate(manifest, evidence)


@pytest.mark.parametrize("mutation, message", [
    ("restricted", "research-only"), ("source", "approval"),
    ("scene", "scene crosses"), ("duplicate", "content crosses"),
    ("missing_val", "train and val"), ("digest", "SHA256 required"),
    ("conflicting_path", "conflicting content hashes"),
])
def test_late_manifest_errors_fail_before_any_media_read(tmp_path, monkeypatch, mutation, message):
    from pipeline import commercial_preflight as preflight
    manifest, evidence, rows, receipt = approved_fixture(tmp_path)
    last = rows[-1]
    if mutation == "restricted": last["commercial_ok"] = False
    if mutation == "source": last["source_id"] = "unreviewed"
    if mutation == "scene": last["scene_id"] = rows[0]["scene_id"]
    if mutation == "duplicate": last["sdr_sha256"] = rows[0]["sdr_sha256"]
    if mutation == "missing_val": last["split"] = "train"
    if mutation == "digest": last["sdr_sha256"] = "not-a-sha256"
    if mutation == "conflicting_path": last["sdr_path"] = rows[0]["sdr_path"]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt["manifest_sha256"] = sha256(manifest)
    evidence.write_text(json.dumps(receipt))
    original = preflight.sha256
    def guarded(path):
        assert path.suffix != ".bin", "invalid final row must fail before hashing even the first media file"
        return original(path)
    monkeypatch.setattr(preflight, "sha256", guarded)
    with pytest.raises(ValueError, match=message):
        validate(manifest, evidence)


def test_valid_preflight_rehashes_each_unique_media_file(tmp_path, monkeypatch):
    from pipeline import commercial_preflight as preflight
    manifest, evidence, rows, _ = approved_fixture(tmp_path)
    original = preflight.sha256
    reads = []
    def counted(path):
        if path.suffix == ".bin": reads.append(str(path))
        return original(path)
    monkeypatch.setattr(preflight, "sha256", counted)
    assert validate(manifest, evidence)["rows"] == 2
    assert sorted(reads) == sorted(r[f"{k}_path"] for r in rows for k in ("sdr", "hdr"))
