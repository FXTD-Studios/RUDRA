"""Rows from an ingest (pipeline/ingest_fxtd_footage.py _records.jsonl) for the
0 EV fix (5 Nov 2026), with the fresh set's reference rule: a frame whose
reference is sensor-clipped (any sensor_clipped_channels, or peak_at_ceiling)
cannot serve as truth and is left out.

    --for holdout  candidate JSONL for tools/holdout_kit.py register
                   (id, scene_id, path, project, camera)
    --for corpus   a JSONL for tools/build_real_partial_corpus.py --sources
                   (each record, hdr_path made absolute)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def usable(rec: dict) -> bool:
    return not any(rec.get("sensor_clipped_channels") or []) and not rec.get("peak_at_ceiling")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ingest", type=Path, required=True, help="the ingest's --out folder")
    ap.add_argument("--for", dest="use", choices=["holdout", "corpus"], required=True)
    ap.add_argument("--camera", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    recs = [json.loads(l) for p in sorted(a.ingest.glob("_records*.jsonl")) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = [r for r in recs if usable(r)]
    scenes = {r["scene_id"] for r in keep}
    print(f"{len(recs)} frames ingested; {len(keep)} with an unclipped reference, from {len(scenes)} shots "
          f"({len(recs) - len(keep)} left out: the reference itself is sensor-clipped)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.use == "holdout":
        with a.out.open("w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps({"id": r["asset_id"], "scene_id": r["scene_id"], "path": str(a.ingest / r["hdr_path"]),
                                    "project": r.get("project", ""), "camera": a.camera or r.get("camera", "")}) + "\n")
    else:
        with a.out.open("w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps({**r, "hdr_path": str(a.ingest / r["hdr_path"])}) + "\n")
    print(f"-> {a.out}")
    return 0 if keep else 2


if __name__ == "__main__":
    sys.exit(main())
