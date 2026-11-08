#!/usr/bin/env python3
"""Promote a RUDRA ACES 2 package from preview to release, only on its evidence (7 Nov 2026).

The Crew holdout registration (configs/experiments/holdout_crew_eval_20261107.json,
"if_it_passes") lets w4 + lock_dqs050g leave preview after four separate steps. This
checks every one of them against the files that show it, and refuses on the first gap:

  holdout      the registered verdict passes, for the arm the package's lock names
  lock         the manifest carries that arm (stage 6b), from that registration
  native       C++ == Python stage by stage with the lock (parity.json), the master's
               round trip, and LibTorch against ONNX Runtime DirectML and CPU
  coreml       a Core ML backend report that passes, or --coreml-not-measured REASON
               (recorded in the manifest and to be said in the release notes)
  provenance   tools/model_provenance.py's record: the files match, a report names
               the checkpoint
  review       the visual review signed off (tools/aces2_review_kit.py's review.json):
               approved, a reviewer, a date, the HDR monitor review, a verdict per frame

Then it copies the package to --stage (the folder PUBLISH_HF_MODELS.ps1 uploads; the
old one moves to <stage>.previous_<date>) with release_authorized true and a "release"
block naming all of the above. The source package is not changed.

    python tools/promote_aces2_release.py --package dist/models/RUDRA_ACES_2_w4 ^
        --holdout-verdict G:/rudra_training/holdout_crew_20261107/verdict.json ^
        --holdout-registration configs/experiments/holdout_crew_eval_20261107.json ^
        --native-report reports/aces_v2_<stamp> --provenance docs/models/RUDRA_ACES_2.provenance.json ^
        --review reports/aces2_review/review.json --coreml-report <mac report.json> ^
        --stage dist/hf/rudra-models/models/RUDRA_ACES_2
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256(p: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p: Path, what: str) -> dict:
    """A JSON file, or a refusal naming what is missing."""
    if not p.is_file():
        raise SystemExit(f"REFUSED: no {what} at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def need(ok: bool, msg: str) -> None:
    """Refuse with msg unless ok."""
    if not ok:
        raise SystemExit(f"REFUSED: {msg}")


def check(a) -> dict:
    """Every precondition; returns the release block for the manifest."""
    man = load(a.package / "manifest.json", "package manifest")
    lock = man.get("lock") or {}
    need(man.get("parity_pass") is True, "the export's own parity check did not pass")
    need(bool(lock), "the manifest has no lock: export with --lock-registration/--lock-arm")
    for f, key in ((man.get("g1_file"), "g1_sha256"), (man.get("onnx_file"), "onnx_sha256")):
        if f:
            need(sha256(a.package / f) == man[key], f"{f} does not match the manifest's {key}")

    reg = load(a.holdout_registration, "holdout registration")
    verdict = load(a.holdout_verdict, "holdout verdict")
    primary = reg["arms"]["primary"]
    need(verdict.get("passes") is True, "the holdout verdict does not pass")
    need(verdict.get("primary", "").endswith(":" + primary), f"the verdict is for {verdict.get('primary')}, not {primary}")
    need(lock.get("name") == primary, f"the package's lock is {lock.get('name')!r}; the holdout passed {primary!r}")
    arm = {t["name"]: t for t in reg["arms"]["partial_tighten"]}[primary]
    for k in ("strength", "target", "dq_iters", "dq_max_lap", "iters"):
        need(float(lock[k]) == float(arm[k]) if k != "target" else lock[k] == arm[k],
             f"lock {k} is {lock.get(k)}, the registered arm has {arm.get(k)}")

    nat = a.native_report
    par = load(nat / "parity.json", "native parity report")
    need(par.get("parity_pass") is True, "C++ and Python disagree (parity.json)")
    need((par.get("lock") or {}).get("name") == primary, "the native parity run did not include the lock")
    for name in ("check_master.json", "backends_libtorch_directml.json", "backends_libtorch_ortcpu.json"):
        r = load(nat / name, name)
        need(r.get("pass") is True, f"{name} does not pass")

    if a.coreml_report:
        cm = load(a.coreml_report, "Core ML backend report")
        need(cm.get("pass") is True, "the Core ML backend report does not pass")
        coreml = {"report": str(a.coreml_report), "sha256": sha256(a.coreml_report)}
    else:
        need(bool(a.coreml_not_measured), "give --coreml-report, or --coreml-not-measured REASON")
        coreml = {"not_measured": a.coreml_not_measured}

    prov = load(a.provenance, "provenance record")
    need(prov.get("files_match_manifest") is True, "the provenance record's files do not match")
    need(prov.get("manifest", {}).get("g1_sha256") == man["g1_sha256"], "the provenance record is for another package")
    need(prov.get("evidence") == "found", "no report names the checkpoint (provenance evidence: none)")

    rev = load(a.review, "visual review")
    need(rev.get("schema") == "rudra_aces2_visual_review_v1", "review.json is not a visual review")
    need(rev.get("approved") is True, "the visual review is not approved")
    need(bool(str(rev.get("reviewer", "")).strip()) and bool(str(rev.get("date", "")).strip()), "the review needs a reviewer and a date")
    need(bool(str(rev.get("hdr_monitor_review", "")).strip()), "the review does not say what was checked on the HDR monitor")
    need(rev.get("package_manifest_sha256") in (None, sha256(a.package / "manifest.json")), "the review was of another package")
    frames = rev.get("frames", [])
    need(bool(frames) and all(f.get("verdict") in ("ok", "issue") for f in frames), "every frame needs a verdict, ok or issue")
    need(all(f.get("notes") for f in frames if f["verdict"] == "issue"), "every issue needs a note")

    return {
        "promoted": dt.datetime.now().isoformat(timespec="seconds"),
        "holdout": {"registration": a.holdout_registration.as_posix(), "registration_sha256": sha256(a.holdout_registration),
                    "verdict_sha256": sha256(a.holdout_verdict), "gates": verdict.get("gates")},
        "native": {"report": nat.as_posix(), "parity_sha256": sha256(nat / "parity.json"),
                   "backends": ["libtorch", "onnxruntime_directml", "onnxruntime_cpu"]},
        "coreml": coreml,
        "provenance": {"record": a.provenance.as_posix(), "sha256": sha256(a.provenance)},
        "review": {"reviewer": rev["reviewer"], "date": rev["date"], "hdr_monitor_review": rev["hdr_monitor_review"],
                   "frames": len(frames), "issues": sum(f["verdict"] == "issue" for f in frames), "sha256": sha256(a.review)},
        "claim": "At the delivered exposure, within 0.1 code RMS of the plain ACES 2.0 inverse on partially clipped "
                 "highlights (not better); better at -3 and +3 EV and in highlight colour. Tested on one production "
                 "and one camera (The Crew, ARRI ALEXA LogC3).",
    }


def main(argv: list[str] | None = None) -> int:
    """Check, then stage the released package."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("package", "holdout-verdict", "holdout-registration", "native-report", "provenance", "review", "stage"):
        ap.add_argument("--" + k, type=Path, required=True)
    ap.add_argument("--coreml-report", type=Path)
    ap.add_argument("--coreml-not-measured", default="")
    ap.add_argument("--check-only", action="store_true", help="check the evidence, stage nothing")
    a = ap.parse_args(argv)
    release = check(a)
    print("every precondition holds:")
    print(json.dumps(release, indent=1))
    if a.check_only:
        return 0
    if a.stage.exists():
        old = a.stage.with_name(a.stage.name + ".previous_" + dt.date.today().strftime("%Y%m%d"))
        need(not old.exists(), f"{old} exists; move it away first")
        a.stage.rename(old)
        print(f"previous package kept at {old}")
    shutil.copytree(a.package, a.stage)
    man = json.loads((a.stage / "manifest.json").read_text(encoding="utf-8"))
    man["release_authorized"] = True
    man["release"] = release
    (a.stage / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(f"staged {a.stage} (release_authorized true). Next: scripts\\PUBLISH_HF_MODELS.ps1 -AcesChannel release")
    return 0


if __name__ == "__main__":
    sys.exit(main())
