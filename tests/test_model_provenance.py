"""tools/model_provenance.py: the shipped model's record (1.0.3)."""
import hashlib
import json

from tools.model_provenance import main


def _package(tmp_path, ckpt="c" * 64):
    pkg = tmp_path / "RUDRA_ACES_2"
    pkg.mkdir()
    (pkg / "rudra_g1.ts").write_bytes(b"torchscript")
    (pkg / "rudra_g1.onnx").write_bytes(b"onnx")
    sha = lambda b: hashlib.sha256(b).hexdigest()
    (pkg / "manifest.json").write_text(json.dumps({
        "contract": "rudra_native_aces_v2_staged", "contract_major": 2, "checkpoint_sha256": ckpt,
        "g1_file": "rudra_g1.ts", "g1_sha256": sha(b"torchscript"),
        "onnx_file": "rudra_g1.onnx", "onnx_sha256": sha(b"onnx"),
        "parity_pass": True, "release_authorized": True}))
    return pkg


def test_no_report_means_no_evidence(tmp_path):
    pkg = _package(tmp_path)
    out = tmp_path / "rec.json"
    assert main([str(pkg), "--reports", str(tmp_path / "none"), "--out", str(out)]) == 2
    rec = json.loads(out.read_text())
    assert rec["evidence"] == "none" and rec["files_match_manifest"]


def test_a_report_naming_the_checkpoint_is_found(tmp_path):
    pkg = _package(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "gate.json").write_text(json.dumps({"checkpoint_sha256": "c" * 64, "verdict": "PASS",
                                                   "gates": {"grade_0ev": {"value": 0.05, "limit": 0.0604}}}))
    (reports / "other.json").write_text(json.dumps({"checkpoint_sha256": "d" * 64, "verdict": "FAIL"}))
    out = tmp_path / "rec.json"
    assert main([str(pkg), "--reports", str(reports), "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert len(rec["reports_naming_this_checkpoint"]) == 1
    assert rec["reports_naming_this_checkpoint"][0]["verdict_fields"]["verdict"] == "PASS"


def test_a_changed_file_fails(tmp_path):
    pkg = _package(tmp_path)
    (pkg / "rudra_g1.onnx").write_bytes(b"tampered")
    assert main([str(pkg), "--reports", str(tmp_path), "--out", str(tmp_path / "rec.json")]) == 1
