"""Fail-closed evidence check for a new commercial image-training run.

This verifies a reviewed artifact contract, not legal rights. No path keyword
or legacy commercial_ok flag can grant approval. Research training is unchanged.
Only scratch initialization is supported until ancestor receipts are implemented.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DECODER_CODE = (REPO / "training/sdr2hdr_dataset.py", REPO / "pipeline/hdr_io.py")


def decoder_dependencies(rows: list[dict]) -> set[Path]:
    """Match the loader's nearest ingest-sentinel search; no torch import."""
    required = {p.resolve() for p in DECODER_CODE}
    for row in rows:
        if not row.get("hdr_path") or not Path(row["hdr_path"]).is_absolute():
            raise ValueError("absolute HDR path required for decoder dependency review")
        # The loader (storage_for) searches from the unresolved path; so must
        # this, or a symlink/junction could pin a different sentinel.
        directory = Path(row["hdr_path"]).parent
        sentinel = next((p / "_ingest_config.json" for p in (directory, *list(directory.parents)[:3])
                         if (p / "_ingest_config.json").is_file()), None)
        if sentinel is None or not json.loads(sentinel.read_text(encoding="utf-8")).get("storage"):
            raise ValueError("commercial data requires explicit HDR storage metadata; legacy fallback is not approved")
        required.add(sentinel.resolve())
        # An image row has one sidecar; a clip row one per frame (metadata_paths).
        for meta in [row.get("metadata_path"), *(row.get("metadata_paths") or [])]:
            if meta:
                required.add(Path(meta).resolve())
    return required


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate(manifest: Path, evidence: Path, progress=None) -> dict:
    """Read only: verify approval, source evidence, asset hashes and split isolation.

    Evidence JSON: schema_version=2, manifest_sha256, reviewed_by, reviewed_at,
    sources={source_id: {commercial_training: true, weight_distribution: true,
    evidence_path, evidence_sha256}}. Evidence paths are relative to this JSON.
    Every manifest row needs source_id, scene_id, asset_id, explicit split,
    sdr_path/sdr_sha256 and hdr_path/hdr_sha256. Media paths are absolute to
    avoid the research loader's cwd-relative convention becoming ambiguous.
    """
    receipt = json.loads(evidence.read_text(encoding="utf-8"))
    digest = sha256(manifest)
    if (receipt.get("schema_version") != 2 or receipt.get("manifest_sha256") != digest
            or not receipt.get("reviewed_by") or not receipt.get("reviewed_at")):
        raise ValueError("commercial evidence must identify a reviewer, date and exact manifest hash")
    approvals = receipt.get("sources", {})
    scope = receipt.get("scope", "commercial-candidate")
    if scope not in ("commercial-candidate", "internal-pilot"):
        raise ValueError("unsupported commercial review scope")
    if scope == "internal-pilot" and receipt.get("release_authorized") is not False:
        raise ValueError("internal pilot must explicitly prohibit release")
    approved = set()
    for sid, item in approvals.items():
        if (item.get("commercial_training") is not True
                or (scope != "internal-pilot" and item.get("weight_distribution") is not True)):
            continue
        if not item.get("evidence_path") or not item.get("evidence_sha256"):
            raise ValueError(f"{sid}: missing license evidence")
        if sha256(evidence.parent / item["evidence_path"]) != item["evidence_sha256"]:
            raise ValueError(f"{sid}: license evidence hash mismatch")
        approved.add(sid)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty commercial manifest")
    # Inspect every row before reading media. A late invalid record must not
    # waste a complete corpus rehash. Declared hashes are still verified below.
    scenes, content, ids, expected_files, used, splits = {}, {}, set(), {}, set(), set()
    for row in rows:
        sid, scene, asset, split = (row.get(k) for k in ("source_id", "scene_id", "asset_id", "split"))
        if sid not in approved or not scene or not asset or asset in ids or split not in ("train", "val", "test"):
            raise ValueError(f"{asset}: missing approval/identity/split or duplicate asset ID")
        if row.get("commercial_ok") is False:
            raise ValueError(f"{asset}: research-only record conflicts with commercial approval")
        ids.add(asset); used.add(sid); splits.add(split)
        key = (sid, scene)
        if key in scenes and scenes[key] != split:
            raise ValueError(f"{asset}: scene crosses splits")
        scenes[key] = split
        for kind in ("sdr", "hdr"):
            path = Path(row.get(f"{kind}_path", ""))
            expected = row.get(f"{kind}_sha256")
            if (not path.is_absolute() or not isinstance(expected, str)
                    or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
                raise ValueError(f"{asset}: absolute {kind} path and SHA256 required")
            path = path.resolve()
            if path in expected_files and expected_files[path] != expected:
                raise ValueError(f"{asset}: conflicting content hashes for one media path")
            expected_files[path] = expected
            key = (kind, expected)
            if key in content and content[key] != split:
                raise ValueError(f"{asset}: duplicate {kind} content crosses splits")
            content[key] = split
    if not {"train", "val"} <= splits:
        raise ValueError("commercial manifest requires explicit train and val splits")
    # Check tiny metadata/code dependencies before a potentially multi-hour
    # media rehash. Identical PNGs can mean different nits after a sidecar edit.
    pinned = receipt.get("decoder_dependencies", {})
    for path in decoder_dependencies(rows):
        if not pinned.get(str(path)) or sha256(path) != pinned[str(path)]:
            raise ValueError(f"decoder dependency missing or changed: {path}")
    for index, (path, expected) in enumerate(expected_files.items(), 1):
        if sha256(path) != expected:
            raise ValueError(f"{path}: content hash mismatch")
        if progress is not None and (index % 1000 == 0 or index == len(expected_files)):
            progress(f"[commercial-preflight] verified {index}/{len(expected_files)} media files")
    return {"policy": "reviewed-scratch-v2", "manifest_sha256": digest,
            "scope": scope, "release_authorized": False,
            "evidence_sha256": sha256(evidence), "reviewed_by": receipt["reviewed_by"],
            "reviewed_at": receipt["reviewed_at"], "sources": sorted(used), "rows": len(rows),
            "initialization": "scratch", "decoder_dependencies": pinned,
            "legal_clearance": "external review required"}


def training_preflight(args) -> dict | None:
    evidence = getattr(args, "commercial_evidence", None)
    if not evidence:
        return None
    if args.mode != "image" or any(getattr(args, k, None) for k in ("resume", "init_checkpoint", "image_checkpoint")):
        raise ValueError("commercial training currently requires scratch image initialization; ancestor clearance is unsupported")
    output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("commercial training requires an empty, separate output directory")
    return validate(Path(args.manifest), Path(evidence), progress=lambda message: print(message, flush=True))
