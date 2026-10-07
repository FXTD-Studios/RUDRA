"""Split a bench's per-frame scores by the SDR's source curve.

    python training/bench_by_curve.py --bench bench/cp_mix --manifest G:\\datasets\\corpora\\corpus_v4c\\sdr_hdr_manifest.jsonl
    python training/bench_by_curve.py --bench bench/cp_mix --manifest ... --models v7 v7_unknown v4c

Why (7 Oct 2026): on the mix bench v7 scored 22.87 dB with each frame's TRUE
curve given and 23.66 dB blind. A label that is correct and makes things
worse means either the one-hot is wired to the wrong curve for some ids or
the CurveHead learned an ACES-only prior. Per-curve deltas against the
analytic inverse say which. No GPU, no torch: reads results/<model>.csv and
joins on the manifest's asset_id (the frame column is the asset_id with
unsafe characters replaced by "_", as export_bench_pairs writes it).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["frame"]] = {"pu_psnr_db": float(r["pu_psnr_db"]), "cvvdp_jod": float(r["cvvdp_jod"])}
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, type=Path, help="e.g. bench/cp_mix")
    ap.add_argument("--manifest", required=True, help="the manifest the bench was exported from")
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="*", help="result stems to report (default: every CSV but baseline)")
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--out", type=Path, help="write the table as JSON")
    args = ap.parse_args()

    curve_of = {}
    with open(args.manifest, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split", args.split) != args.split:
                continue
            key = UNSAFE.sub("_", str(r["asset_id"]))
            curve_of[key] = str(r.get("source_curve") or r.get("sdr_curve") or "unknown")
    results = args.bench / "results"
    ref = load_scores(results / f"{args.reference}.csv")
    models = args.models or sorted(p.stem for p in results.glob("*.csv") if p.stem != args.reference)
    table = {}
    for model in models:
        path = results / f"{model}.csv"
        if not path.exists():
            print(f"skip {model}: no {path}")
            continue
        got = load_scores(path)
        by_curve: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"pu_psnr_db": [], "cvvdp_jod": []})
        missing = 0
        for frame, s in got.items():
            if frame not in ref:
                continue
            curve = curve_of.get(frame)
            if curve is None:
                missing += 1
                curve = "?"
            for m in ("pu_psnr_db", "cvvdp_jod"):
                by_curve[curve][m].append(s[m] - ref[frame][m])
        print(f"\n{args.bench.name}/{model} vs {args.reference}" + (f"   ({missing} frames not in manifest)" if missing else ""))
        print(f"  {'curve':<12}{'n':>6}{'dPU21 dB':>12}{'wins':>8}{'dJOD':>10}{'wins':>8}")
        table[model] = {}
        for curve in sorted(by_curve, key=lambda c: -len(by_curve[c]["pu_psnr_db"])):
            d = by_curve[curve]
            n = len(d["pu_psnr_db"])
            row = {"frames": n,
                   "pu_psnr_db": statistics.mean(d["pu_psnr_db"]), "pu_wins": sum(x > 0 for x in d["pu_psnr_db"]),
                   "cvvdp_jod": statistics.mean(d["cvvdp_jod"]), "jod_wins": sum(x > 0 for x in d["cvvdp_jod"])}
            table[model][curve] = row
            print(f"  {curve:<12}{n:>6}{row['pu_psnr_db']:>+12.3f}{row['pu_wins']:>8}{row['cvvdp_jod']:>+10.3f}{row['jod_wins']:>8}")
    if args.out:
        args.out.write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
