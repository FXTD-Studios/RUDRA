"""Register, seal and verify a confirmatory holdout (1 Nov 2026).

The fresh FXTD grade v2 set has been looked at (26 Oct) and the full-only arm
was designed knowing its result, so it is development evidence now. A claim
that RUDRA passes its gates needs a set nobody has looked at. This tool makes
one the way v2 was made by hand, and checks it mechanically:

  register  Read candidate rows (JSONL: id, scene_id, path, and any strata
            fields such as project and camera), drop every row whose scene_id,
            file SHA-256 or file name appears in ANY manifest or report under
            the given roots (training corpora, schedules, development and
            fresh sets), then draw N rows stratified by the strata fields with
            a fixed seed. Writes the registration with every file's hash,
            what was excluded and why, and a seal over the whole record.
            Refuses to overwrite.
  verify    Re-hash every file and re-check the seal. Run it right before the
            confirmatory evaluation, and attach its output to the report.

    python tools/holdout_kit.py register --candidates G:/datasets/new_shoot/rows.jsonl ^
        --exclude-roots G:/rudra_training G:/datasets/corpora configs/experiments ^
        --n 48 --strata project camera --seed 20261101 --out configs/experiments/holdout_v1_20261101.json
    python tools/holdout_kit.py verify configs/experiments/holdout_v1_20261101.json

Nothing here looks at a model output. Build the cases from the registered
rows with the same processing as tools/prepare_fresh_fxtd_grade_v2.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
SCENE_KEYS = ("scene_id", "scene", "sceneId")
MANIFEST_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".md"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _walk_values(obj, keys, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, (str, int)):
                out.add(str(v))
            _walk_values(v, keys, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_values(v, keys, out)


def used_identifiers(roots: list[Path], max_mb: float = 200.0, skip: Path | None = None) -> dict:
    """Every scene id, SHA-256 and file name mentioned under the roots."""
    scenes, hashes, names, files = set(), set(), set(), 0
    for root in roots:
        if not root.exists():
            continue
        it = [root] if root.is_file() else root.rglob("*")
        for p in it:
            if not p.is_file() or p.suffix.lower() not in MANIFEST_SUFFIXES or p.stat().st_size > max_mb * 1e6:
                continue
            if skip is not None and p.resolve() == skip.resolve():
                continue
            files += 1
            text = p.read_text(encoding="utf-8", errors="ignore")
            hashes.update(HEX64.findall(text))
            names.update(m.group(0).lower() for m in re.finditer(r"[\w.-]+\.(?:exr|dpx|png|tif|tiff|mov|mp4|mxf|r3d|ari|npy)\b", text, re.I))
            lines = text.splitlines() if p.suffix.lower() == ".jsonl" else [text]
            for line in lines:
                try:
                    _walk_values(json.loads(line), SCENE_KEYS, scenes)
                except (json.JSONDecodeError, ValueError):
                    continue
    return {"scene_ids": scenes, "sha256": hashes, "file_names": names, "manifest_files": files}


def stratified_pick(rows: list[dict], n: int, strata: list[str], seed: int) -> list[dict]:
    """N rows, as even as possible across strata, at most one row per scene
    first (then more per scene only if needed), deterministic for a seed."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in sorted(rows, key=lambda r: str(r["id"])):
        groups[tuple(str(r.get(s, "")) for s in strata)].append(r)
    for g in groups.values():
        rng.shuffle(g)
        # one per scene first
        seen, first, rest = set(), [], []
        for r in g:
            (rest if r["scene_id"] in seen else first).append(r)
            seen.add(r["scene_id"])
        g[:] = first + rest
    keys = sorted(groups)
    picked = []
    while len(picked) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop(0))
    return picked


def seal(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "seal_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def register(a) -> int:
    if a.out.exists():
        print(f"FAILED: {a.out} exists; a registration is never rewritten (make a new one)")
        return 1
    cands = [json.loads(l) for l in a.candidates.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in cands:
        for k in ("id", "scene_id", "path"):
            if k not in r:
                print(f"FAILED: candidate row without {k}: {r}")
                return 1
    used = used_identifiers(a.exclude_roots, skip=a.candidates)
    kept, excluded = [], []
    for r in cands:
        p = Path(r["path"])
        if not p.is_file():
            excluded.append({"id": r["id"], "reason": "file missing"})
            continue
        h = sha256(p)
        why = ("scene_id used" if str(r["scene_id"]) in used["scene_ids"] else
               "file hash used" if h in used["sha256"] else
               "file name used" if p.name.lower() in used["file_names"] else None)
        if why:
            excluded.append({"id": r["id"], "scene_id": r["scene_id"], "reason": why})
        else:
            kept.append({**r, "sha256": h})
    picked = stratified_pick(kept, a.n, a.strata, a.seed)
    if len(picked) < a.n:
        print(f"WARNING: only {len(picked)} eligible rows for n={a.n}")
    record = {
        "schema": "rudra_holdout_registration_v1",
        "registered": dt.datetime.now().isoformat(timespec="seconds"),
        "registered_before_building": True,
        "registered_before_evaluation": True,
        "purpose": a.purpose,
        "candidates_file": str(a.candidates), "candidates_sha256": sha256(a.candidates),
        "exclude_roots": [str(r) for r in a.exclude_roots],
        "exclusion_index": {"manifest_files": used["manifest_files"], "scene_ids": len(used["scene_ids"]),
                            "sha256": len(used["sha256"]), "file_names": len(used["file_names"])},
        "selection": {"n": a.n, "strata": a.strata, "seed": a.seed, "rule": "stratified round robin, one row per scene first"},
        "counts": {"candidates": len(cands), "eligible": len(kept), "excluded": len(excluded), "picked": len(picked),
                   "scenes": len({r["scene_id"] for r in picked})},
        "rows": picked,
        "excluded": excluded,
        "use": "confirmatory only: build the cases, run the registered candidate once, report whatever it shows",
    }
    record["seal_sha256"] = seal(record)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"registered {len(picked)} rows ({record['counts']['scenes']} scenes) of {len(cands)}; "
          f"{len(excluded)} excluded -> {a.out}")
    print(f"seal {record['seal_sha256']}")
    return 0 if len(picked) == a.n else 2


def verify(a) -> int:
    rec = json.loads(a.registration.read_text(encoding="utf-8"))
    ok = True
    if seal(rec) != rec.get("seal_sha256"):
        print("FAILED: the registration was edited after it was sealed")
        ok = False
    for r in rec["rows"]:
        p = Path(r["path"])
        if not p.is_file():
            print(f"FAILED: missing {p}")
            ok = False
        elif sha256(p) != r["sha256"]:
            print(f"FAILED: changed {p}")
            ok = False
    if a.exclude_roots:
        used = used_identifiers(a.exclude_roots, skip=a.registration)
        # Point these at training data only: anything there naming a registered
        # scene or file means the holdout leaked into training.
        leaked = [r["id"] for r in rec["rows"] if str(r["scene_id"]) in used["scene_ids"] or r["sha256"] in used["sha256"]]
        if leaked:
            print(f"FAILED: {len(leaked)} registered scenes now appear under the exclusion roots: {leaked[:5]}")
            ok = False
    print(("PASS" if ok else "FAILED") + f": {len(rec['rows'])} rows, seal {rec.get('seal_sha256', '')[:16]}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--candidates", type=Path, required=True)
    r.add_argument("--exclude-roots", type=Path, nargs="+", required=True)
    r.add_argument("--n", type=int, default=48)
    r.add_argument("--strata", nargs="*", default=["project", "camera"])
    r.add_argument("--seed", type=int, default=20261101)
    r.add_argument("--purpose", default="confirmatory holdout for the learned ACES 2.0 model's release gates")
    r.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("registration", type=Path)
    v.add_argument("--exclude-roots", type=Path, nargs="*", default=[],
                   help="re-check that no registered scene has since entered these roots (point at training data only)")
    a = ap.parse_args(argv)
    return register(a) if a.cmd == "register" else verify(a)


if __name__ == "__main__":
    sys.exit(main())
