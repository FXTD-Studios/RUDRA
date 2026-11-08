"""tools/promote_aces2_release.py (7 Nov 2026): RUDRA ACES 2 leaves preview only on its evidence.
Made-up files throughout; nothing of the holdout or the model is read."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from promote_aces2_release import main  # noqa: E402

ARM = {"name": "lock_dqs050g", "mode": "lock", "iters": 4, "band_codes": 0.0, "strength": 0.5, "target": "dequant",
       "dq_iters": 200, "dq_max_lap": 1.5}


def put(p: Path, obj) -> Path:
    """Write obj as JSON (or bytes) at p."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(obj if isinstance(obj, bytes) else json.dumps(obj).encode())
    return p


def evidence(tmp: Path) -> list[str]:
    """A complete, passing set of evidence; returns the command-line arguments."""
    pkg = tmp / "pkg"
    g1 = put(pkg / "rudra_g1.ts", b"weights")
    g1_sha = hashlib.sha256(g1.read_bytes()).hexdigest()
    put(pkg / "manifest.json", {"contract_major": 3, "g1_file": "rudra_g1.ts", "g1_sha256": g1_sha, "parity_pass": True,
                                "release_authorized": False, "lock": {k: ARM[k] for k in ARM}})
    reg = put(tmp / "reg.json", {"arms": {"primary": "lock_dqs050g", "partial_tighten": [ARM]}})
    verdict = put(tmp / "verdict.json", {"primary": "w4:lock_dqs050g", "passes": True, "gates": {}})
    nat = tmp / "native"
    put(nat / "parity.json", {"parity_pass": True, "lock": ARM})
    for n in ("check_master.json", "backends_libtorch_directml.json", "backends_libtorch_ortcpu.json"):
        put(nat / n, {"pass": True})
    coreml = put(tmp / "coreml.json", {"pass": True})
    prov = put(tmp / "prov.json", {"files_match_manifest": True, "manifest": {"g1_sha256": g1_sha}, "evidence": "found"})
    review = put(tmp / "review.json", {"schema": "rudra_aces2_visual_review_v1", "approved": True, "reviewer": "A. Talat",
                                       "date": "2026-11-08", "hdr_monitor_review": "Resolve, 1000-nit reference monitor",
                                       "package_manifest_sha256": None,
                                       "frames": [{"frame": "a.png", "verdict": "ok", "notes": ""}]})
    return ["--package", str(pkg), "--holdout-verdict", str(verdict), "--holdout-registration", str(reg),
            "--native-report", str(nat), "--provenance", str(prov), "--review", str(review),
            "--coreml-report", str(coreml), "--stage", str(tmp / "stage" / "RUDRA_ACES_2")]


def test_complete_evidence_stages_a_release_authorized_copy(tmp_path):
    """Everything on file: the staged manifest says release, the source package is untouched."""
    args = evidence(tmp_path)
    put(tmp_path / "stage" / "RUDRA_ACES_2" / "manifest.json", {"old": True})
    assert main(args) == 0
    staged = json.loads((tmp_path / "stage" / "RUDRA_ACES_2" / "manifest.json").read_text())
    assert staged["release_authorized"] is True and staged["release"]["review"]["reviewer"] == "A. Talat"
    assert "within 0.1 code" in staged["release"]["claim"]
    assert json.loads((tmp_path / "pkg" / "manifest.json").read_text())["release_authorized"] is False
    assert list((tmp_path / "stage").glob("RUDRA_ACES_2.previous_*"))     # the old package is kept


@pytest.mark.parametrize("breaks, why", [
    (lambda t: put(t / "verdict.json", {"primary": "w4:lock_dqs050g", "passes": False}), "holdout verdict"),
    (lambda t: put(t / "verdict.json", {"primary": "w4:lock_b040", "passes": True}), "verdict is for"),
    (lambda t: put(t / "native" / "parity.json", {"parity_pass": True, "lock": None}), "without|did not include"),
    (lambda t: put(t / "native" / "backends_libtorch_directml.json", {"pass": False}), "directml"),
    (lambda t: put(t / "coreml.json", {"pass": False}), "Core ML"),
    (lambda t: put(t / "prov.json", {"files_match_manifest": True, "manifest": {"g1_sha256": "x"}, "evidence": "found"}),
     "another package"),
    (lambda t: put(t / "review.json", {"schema": "rudra_aces2_visual_review_v1", "approved": False}), "not approved"),
    (lambda t: (t / "pkg" / "rudra_g1.ts").write_bytes(b"changed"), "does not match"),
])
def test_any_gap_refuses_and_stages_nothing(tmp_path, breaks, why):
    """One missing or failing piece of evidence: refused, nothing staged."""
    args = evidence(tmp_path)
    breaks(tmp_path)
    with pytest.raises(SystemExit, match=f"REFUSED.*({why})"):
        main(args)
    assert not (tmp_path / "stage").exists()


def test_without_coreml_a_reason_is_required(tmp_path):
    """No Core ML report: refused unless a reason is given, which is then recorded."""
    args = evidence(tmp_path)
    i = args.index("--coreml-report")
    args = args[:i] + args[i + 2:]
    with pytest.raises(SystemExit, match="coreml-not-measured"):
        main(args)
    assert main(args + ["--coreml-not-measured", "Mac unavailable; Core ML ships as preview"]) == 0
    staged = json.loads((tmp_path / "stage" / "RUDRA_ACES_2" / "manifest.json").read_text())
    assert staged["release"]["coreml"] == {"not_measured": "Mac unavailable; Core ML ships as preview"}
