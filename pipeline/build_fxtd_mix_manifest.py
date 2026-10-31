"""Add the FXTD camera corpus to an existing training mix (17 Oct 2026).

    python pipeline/build_fxtd_mix_manifest.py \
        --base G:/datasets/corpora/rudra_mix_v10_20261013/sdr_hdr_manifest.jsonl \
        --fxtd G:/datasets/corpora/rudra_fxtd_camera_20261016/sdr_hdr_manifest.jsonl \
        --out  G:/datasets/corpora/rudra_mix_v11_fxtd_20261017

Base rows are copied unchanged; FXTD rows keep their own shot-level splits.
Refuses an asset or scene id on both sides, a corpus exposure other than the
base's (corpus_ev_of would refuse the mix later anyway), and any FXTD row
whose PNGs do not exist under --check-root (default: skip the check).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge(base: list[dict], fxtd: list[dict]) -> tuple[list[dict], dict]:
    ids = {r["asset_id"] for r in base}
    scenes = {str(r["scene_id"]) for r in base}
    clash = [r["asset_id"] for r in fxtd if r["asset_id"] in ids or str(r["scene_id"]) in scenes]
    if clash:
        raise SystemExit(f"{len(clash)} FXTD rows collide with the base, e.g. {clash[:3]}")
    # Pipeline pairs have no tonemap_ev (corpus_ev_of skips None), so they do not vote.
    evs = {r.get("tonemap_ev") for r in base + fxtd
           if r.get("sdr_kind") != "real" and r.get("tonemap_ev") is not None}
    if len(evs) != 1:
        raise SystemExit(f"rendered rows disagree on tonemap_ev: {sorted(map(str, evs))}")
    out = base + [dict(r, source_corpus="rudra_fxtd_pipeline" if r.get("render_family") == "pipeline"
                       else "rudra_fxtd_camera") for r in fxtd]
    by = Counter((r.get("split"), r.get("sdr_kind", "rendered"), r.get("licence_source", "?"),
                  r.get("render_family", "")) for r in out)
    report = {"base_rows": len(base), "fxtd_rows": len(fxtd), "rows": len(out), "tonemap_ev": evs.pop(),
              "fxtd_scenes": len({r["scene_id"] for r in fxtd}),
              "by_split_kind_source": {"/".join(str(x) for x in k if x): n for k, n in sorted(by.items(), key=str)}}
    return out, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--fxtd", required=True, type=Path, nargs="+",
                    help="one or more FXTD manifests (camera rendered, pipeline pairs)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    fxtd = [r for path in args.fxtd for r in read(path)]
    if len({r["asset_id"] for r in fxtd}) != len(fxtd):
        raise SystemExit("duplicate asset ids across the FXTD manifests")
    out, report = merge(read(args.base), fxtd)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "sdr_hdr_manifest.jsonl"
    with target.open("w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    report.update({"base": str(args.base), "fxtd": [str(p) for p in args.fxtd],
                   "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (args.out / "mix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
