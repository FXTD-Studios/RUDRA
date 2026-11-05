"""Write the provenance record of a learned ACES 2.0 package (RUDRA_ACES_2).

The 1.0 model went out with release_authorized true in its manifest, but
tools/export_native_v2.py always writes false and no committed report shows
which checkpoint it is or which gates it passed. This closes that gap: it
reads the package, re-hashes every file, then searches report folders for
every JSON that names the package's checkpoint_sha256 (or g1_sha256), and
writes one record you commit beside the model card.

    python tools/model_provenance.py dist/models/RUDRA_ACES_2 ^
        --reports G:/rudra_training reports ^
        --out docs/models/RUDRA_ACES_2.provenance.json

Exit 0: the files match the manifest and at least one report names the
checkpoint. Exit 1: a file does not match. Exit 2: no report names the
checkpoint, so the release has no evidence and must be labelled preview.
It reads only; it never changes the package.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

VERDICT_KEYS = ("verdict", "pass", "passed", "ok", "gates", "gate", "result", "promotion", "release_authorized",
                "registration", "candidate", "checkpoint", "step")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def package_files(root: Path, manifest: dict) -> list[dict]:
    rows = []
    declared = {manifest.get("g1_file"): manifest.get("g1_sha256"), manifest.get("onnx_file"): manifest.get("onnx_sha256")}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        got = sha256(p)
        want = declared.get(rel)
        rows.append({"file": rel, "bytes": p.stat().st_size, "sha256": got,
                     "declared_sha256": want, "matches": None if want is None else got == want})
    return rows


def summarise(obj, depth=0):
    """The verdict-like fields of a report, shallowly, so the record stays readable."""
    if depth > 2:
        return None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if any(key in k.lower() for key in VERDICT_KEYS):
                out[k] = v if not isinstance(v, (dict, list)) or len(json.dumps(v)) < 400 else summarise(v, depth + 1)
        return out or None
    return None


def find_reports(roots: list[Path], needles: list[str], limit_mb: float) -> list[dict]:
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.json"):
            try:
                if p.stat().st_size > limit_mb * 1e6:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            found = [n for n in needles if n and n in text]
            if not found:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
            hits.append({"path": str(p), "sha256": sha256(p), "names": found,
                         "modified": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                         "verdict_fields": summarise(data) if data is not None else None})
    return sorted(hits, key=lambda h: h["modified"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", type=Path)
    ap.add_argument("--reports", nargs="*", type=Path, default=[Path("reports")])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-report-mb", type=float, default=20.0)
    a = ap.parse_args(argv)

    manifest = json.loads((a.package / "manifest.json").read_text(encoding="utf-8"))
    files = package_files(a.package, manifest)
    ckpt, g1 = manifest.get("checkpoint_sha256", ""), manifest.get("g1_sha256", "")
    reports = find_reports(a.reports, [ckpt, g1], a.max_report_mb)
    mismatched = [f["file"] for f in files if f["matches"] is False]

    record = {
        "schema": "rudra_model_provenance_v1",
        "written": dt.datetime.now().isoformat(timespec="seconds"),
        "package": a.package.name,
        "manifest": {k: manifest.get(k) for k in ("contract", "contract_major", "checkpoint_sha256", "g1_file", "g1_sha256",
                                                  "onnx_file", "onnx_sha256", "onnx_tolerance", "parity_pass",
                                                  "release_authorized", "model_contract", "clipped_smoothing",
                                                  "appearance_constraint")},
        "manifest_sha256": sha256(a.package / "manifest.json"),
        "files": files,
        "files_match_manifest": not mismatched,
        "reports_searched": [str(r) for r in a.reports],
        "reports_naming_this_checkpoint": reports,
        "evidence": "found" if reports else "none",
        "note": ("release_authorized is not written by tools/export_native_v2.py; a true value needs a gate "
                 "report below that names this checkpoint. Without one, label the model preview."),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {a.out}")
    print(f"checkpoint {ckpt or '(none in manifest)'}; release_authorized {manifest.get('release_authorized')}")
    print(f"{len(files)} files, {'all match' if not mismatched else 'MISMATCH: ' + ', '.join(mismatched)}")
    print(f"{len(reports)} report(s) name this checkpoint")
    for r in reports[-10:]:
        print(f"  {r['modified']}  {r['path']}")
    if mismatched:
        return 1
    return 0 if reports else 2


if __name__ == "__main__":
    sys.exit(main())
