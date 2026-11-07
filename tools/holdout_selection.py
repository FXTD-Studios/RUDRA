#!/usr/bin/env python3
"""A sealed holdout registration -> the builder's selection file (7 Nov 2026).

tools/prepare_fresh_fxtd_grade_v2.py builds a grading set from a selection: one row per frame
with hdr_path, asset_id, scene_id, project and plate_primaries. A holdout registration
(tools/holdout_kit.py register) names the same frames as id, scene_id, path, project, camera
and their SHA-256. This maps one to the other and nothing else: every registered row, in
registration order, no frame dropped or added. It refuses a registration whose seal does not
match (edited after sealing). Nothing here reads a plate or a model output.

    python tools/holdout_selection.py configs/experiments/holdout_crew_20261105.json \\
        --plate-primaries rec2020 --out G:/holdout/rudra_holdout_crew_20261105/selection.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout_kit import seal  # noqa: E402


def selection(rec: dict, plate_primaries: str) -> list[dict]:
    """The registration's rows as selection rows, in order; refuses an edited registration."""
    if seal(rec) != rec.get("seal_sha256"):
        raise SystemExit("the holdout registration was edited after it was sealed")
    if plate_primaries not in ("rec2020", "rec709"):
        raise SystemExit("plate primaries must be rec2020 or rec709")
    out = []
    for r in rec["rows"]:
        out.append({"asset_id": r["id"], "scene_id": r["scene_id"], "project": r["project"], "camera": r.get("camera"),
                    "hdr_path": r["path"], "hdr_sha256": r["sha256"], "plate_primaries": plate_primaries})
    if len({s["asset_id"] for s in out}) != len(out):
        raise SystemExit("duplicate asset ids in the registration")
    return out


def main(argv: list[str] | None = None) -> int:
    """Write the selection file; refuse to overwrite one."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("registration", type=Path)
    ap.add_argument("--plate-primaries", required=True, help="the ingest's plate primaries (FXTD camera ingest: rec2020)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists: a selection is written once")
    rows = selection(json.loads(a.registration.read_text(encoding="utf-8")), a.plate_primaries)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} rows from {len({r['scene_id'] for r in rows})} scenes -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
