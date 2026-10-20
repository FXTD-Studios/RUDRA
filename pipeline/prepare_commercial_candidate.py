"""Prepare a review-only real-pair manifest; never modify or copy source media.

Explicit title/source policy, frozen title splits, full media SHA256 and exact
cross-split duplicate detection. This creates no commercial approval receipt.
Run with --hash-media only after considering the byte estimate in --plan.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.commercial_preflight import sha256

POLICY = {
    "Nocturne": ("netflix_nocturne", "nocturne", "train"),
    "Sol Levante": ("netflix_sol_levante", "sol_levante", "train"),
    "Cosmos Laundromat": ("netflix_cosmos", "cosmos", "train"),
    "Sparks": ("netflix_sparks", "sparks", "val"),
    "Meridian": ("netflix_meridian", "meridian", "test"),
}


def select(rows):
    selected, excluded = [], Counter()
    ids, scenes, paths = set(), {}, {}
    for source in rows:
        title = source.get("title")
        if title not in POLICY or source.get("sdr_kind") != "real":
            excluded["not_in_explicit_real_title_policy"] += 1
            continue
        sid, slug, split = POLICY[title]
        shot = source.get("shot")
        asset, scene = source.get("asset_id"), source.get("scene_id")
        if source.get("licence_source") != sid or source.get("commercial_ok") is not True:
            excluded["conflicting_legacy_provenance"] += 1
            continue
        if not asset or not scene or not shot or source.get("excluded"):
            raise ValueError("missing identity or excluded input record")
        if asset in ids:
            raise ValueError(f"duplicate asset ID: {asset}")
        ids.add(asset)
        if scene != f"nf_{slug}_{shot}":
            raise ValueError(f"inconsistent scene identity: {asset}")
        if scene in scenes and scenes[scene] != split:
            raise ValueError("scene crosses splits")
        scenes[scene] = split
        row = dict(source)
        for kind in ("sdr", "hdr"):
            row.pop(f"{kind}_sha256", None)  # Only this audit's reads/cache establish coverage.
        for kind in ("sdr", "hdr"):
            url = urlparse(source.get(f"{kind}_url", ""))
            expected = f"/aom_test_materials/{kind.upper()}/{slug}/{shot}.zip"
            if url.scheme not in ("http", "https") or url.netloc != "download.opencontent.netflix.com.s3.amazonaws.com" or url.path != expected:
                raise ValueError(f"unexpected origin for {asset}/{kind}")
            p = Path(source[f"{kind}_path"])
            if not p.is_absolute() or not p.is_file():
                raise ValueError(f"missing absolute media path: {p}")
            p = p.resolve()
            if p in paths and paths[p] != split:
                raise ValueError("media path crosses splits")
            paths[p] = split
            row[f"{kind}_path"] = str(p)
        row.update(source_id=sid, split=split, previous_split=source.get("split"),
                   commercial_ok=False, rights_status="pending_review", initialization_required="scratch")
        selected.append(row)
    selected.sort(key=lambda r: r["asset_id"])
    return selected, dict(excluded)


def hash_media(rows, progress=lambda message: None, budget_seconds=60, journal=None):
    started = time.monotonic()
    hashes, seen, duplicates = {}, {}, []
    cached = {}
    if journal and journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                cached[entry["path"]] = entry
            except (ValueError, KeyError):
                continue  # An interrupted journal is only a performance cache.
    # Round-robin shots so a bounded scan covers diverse scenes first.
    groups = {}
    for row in rows:
        groups.setdefault(row["scene_id"], []).append(row)
    ordered = [g[i] for i in range(max(map(len, groups.values()), default=0)) for g in groups.values() if i < len(g)]
    for i, row in enumerate(ordered):
        for kind in ("sdr", "hdr"):
            p = Path(row[f"{kind}_path"])
            if p not in hashes:
                stat = p.stat()
                entry = cached.get(str(p), {})
                if entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
                    hashes[p] = entry["sha256"]
                else:
                    if budget_seconds and time.monotonic() - started >= budget_seconds:
                        continue
                    hashes[p] = sha256(p)
                    if journal:
                        with journal.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(dict(path=str(p), size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=hashes[p])) + "\n")
            digest = hashes[p]
            row[f"{kind}_sha256"] = digest
            key = (kind, digest)
            if key in seen:
                prior = seen[key]
                duplicates.append({"kind": kind, "first": prior["asset_id"], "second": row["asset_id"],
                                   "cross_split": prior["split"] != row["split"]})
            else:
                seen[key] = row
        if (i + 1) % 200 == 0:
            progress(f"visited {i + 1}/{len(rows)} pairs; {len(hashes)} media hashes")
    return duplicates


def prepare(manifest: Path, out: Path, hash_files=False, plan=False, progress=print, hash_seconds=60, journal=None):
    started = time.monotonic()
    manifest, out = manifest.resolve(), out.resolve()
    rows, excluded = select([json.loads(s) for s in manifest.read_text(encoding="utf-8").splitlines() if s.strip()])
    if not rows:
        raise ValueError("no eligible candidate rows")
    media = {Path(r[f"{k}_path"]) for r in rows for k in ("sdr", "hdr")}
    # Dedicated destination, outside every source media/manifest directory.
    if out.is_relative_to(manifest.parent) or any(out.is_relative_to(p.parent) for p in media):
        raise ValueError("output must be outside source directories")
    summary = {"training_ready": False, "rights_status": "pending_review", "initialization": "scratch",
               "policy": POLICY, "input_manifest": str(manifest), "input_sha256": sha256(manifest),
               "rows": len(rows), "splits": dict(Counter(r["split"] for r in rows)),
               "titles": dict(Counter(r["title"] for r in rows)),
               "scenes": len({r["scene_id"] for r in rows}), "excluded": excluded,
               "media_files": len(media), "media_bytes": sum(p.stat().st_size for p in media),
               "hash_scope": "all_media" if hash_files and not plan else "none",
               "limitations": ["rights review pending", "perceptual duplicates untested",
                               "Meridian is an existing regression test, not an untouched new test",
                               "title splits valid for scratch initialization only",
                               "real-only corpus: exposure/baseline recipe must be explicitly configured"]}
    if plan:
        return summary
    if out.exists():
        raise ValueError("output already exists; use a new versioned directory")
    if hash_seconds < 0:
        raise ValueError("hash time budget must be non-negative; 0 explicitly means unbounded")
    journal = (journal or out.with_suffix(".hashes.jsonl")).resolve()
    if journal.is_relative_to(manifest.parent) or any(journal.is_relative_to(p.parent) for p in media):
        raise ValueError("hash journal must be outside source directories")
    sentinel = manifest.parent / "_ingest_config.json"
    summary["storage_sentinel"] = {"path": str(sentinel), "sha256": sha256(sentinel)}
    summary["builder_sha256"] = sha256(Path(__file__))
    journal.parent.mkdir(parents=True, exist_ok=True)
    duplicates = hash_media(rows, progress, hash_seconds, journal) if hash_files else []
    hashed = {r[f"{k}_path"] for r in rows for k in ("sdr", "hdr") if r.get(f"{k}_sha256")}
    summary["hashed_files"] = len(hashed)
    summary["hashed_bytes"] = sum(Path(p).stat().st_size for p in hashed)
    summary["hash_scope"] = "all_media" if len(hashed) == len(media) else "partial" if hashed else "none"
    summary["hash_journal"] = str(journal) if hash_files else None
    summary["hash_budget_seconds"] = hash_seconds
    summary["hash_cache_policy"] = "size/mtime audit cache only; commercial preflight rehashes every file"
    summary["duplicates"] = duplicates
    summary["cross_split_duplicates"] = sum(d["cross_split"] for d in duplicates)
    summary["elapsed_seconds"] = time.monotonic() - started
    out.mkdir(parents=True)
    target = out / "candidate_manifest.jsonl"
    target.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary["candidate_sha256"] = sha256(target)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "source_ingest_config.json").write_bytes(sentinel.read_bytes())
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--hash-media", action="store_true")
    ap.add_argument("--hash-seconds", type=float, default=60, help="New hash work budget; 0 = unbounded")
    ap.add_argument("--hash-journal", type=Path, help="Reuse an audit hash journal; final preflight still rehashes")
    args = ap.parse_args()
    result = prepare(args.manifest, args.out, args.hash_media, args.plan, lambda s: print(s, flush=True),
                     args.hash_seconds, args.hash_journal)
    print(json.dumps({k: v for k, v in result.items() if k != "duplicates"}, indent=2))
    return 1 if result.get("cross_split_duplicates") else 0


if __name__ == "__main__":
    raise SystemExit(main())
