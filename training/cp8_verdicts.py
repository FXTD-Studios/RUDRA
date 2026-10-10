"""The v8 gate (line G), written 8 Oct 2026, before any v8 checkpoint exists.

    python training/cp8_verdicts.py                         # bench/ -> reports/logs/cp8_results.json
    python training/cp8_verdicts.py --candidate v8_s2 --strict

v8 ships only if it beats the analytic inverse it rides on where it matters
and loses nothing where the inverse is already right. Every row compares the
candidate's tree with the baseline/ tree of the same export (the inverse at
the model's own exposure):

  G8/real          real SDR (Netflix Meridian, 287 held-out frames, blind):
                   PU21 and CVVDP, both 95 % CIs above zero.
  G8/aces|oog|mix  rendered benches: not worse, both CIs reach zero or above.
  G8/clip/ev0|1|2  the Meridian HDR re-rendered with ACES at the model's
                   exposure +0, +1, +2 EV (export_bench_pairs.py --condition
                   exposure): error in stops inside the SDR's clipped pixels
                   (training/score_regions.py), the candidate's lower than the
                   inverse's, CI above zero, at every EV.

Reports, no gate: the crushed-shadow error at each EV and on real SDR, the
real bench's clipped error. A row whose files are missing reads "not run".
No GPU, no torch. Exit 1 only with --strict and a gate that ran and failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.paired_gate import METRICS, compare, load  # noqa: E402

CLIP_EVS = (0, 1, 2)


def _ok_positive(r: dict, metrics) -> bool:
    return all(r[m].get("frames") and r[m]["ci95"][0] > 0 for m in metrics)


def _ok_not_worse(r: dict, metrics) -> bool:
    return all(r[m].get("frames") and r[m]["ci95"][1] >= 0 for m in metrics)


def _whole(root: Path, bench: str, cand: str):
    a = root / f"cp_{bench}" / "results" / f"{cand}.csv"
    b = root / f"cp_{bench}" / "results" / "baseline.csv"
    if not (a.is_file() and b.is_file()):
        return None
    return compare(load(a), load(b))


def _region(root: Path, bench_dir: str, cand: str, metric: str):
    a = root / bench_dir / "results" / f"{cand}.regions.csv"
    b = root / bench_dir / "results" / "baseline.regions.csv"
    if not (a.is_file() and b.is_file()):
        return None
    m = (metric,)
    # error: lower is better, so baseline - candidate > 0 means the candidate wins
    return compare(load(b, m), load(a, m), metrics=m)


def score(root: Path, cand: str) -> dict:
    gates, reports = {}, {}

    def gate(name, question, r, ok_fn, metrics):
        gates[name] = {"question": question, "status": "not run", "passed": None}
        if r is not None:
            ok = ok_fn(r, metrics)
            gates[name].update({"status": "PASS" if ok else "FAIL", "passed": ok, **{m: r[m] for m in metrics}})

    gate("G8/real", f"{cand} beats the inverse on real SDR (Meridian), PU21 and CVVDP CIs above zero",
         _whole(root, "real", cand), _ok_positive, METRICS)
    for bench in ("aces", "oog", "mix"):
        gate(f"G8/{bench}", f"{cand} is not worse than the inverse on the {bench} bench (CIs reach zero)",
             _whole(root, bench, cand), _ok_not_worse, METRICS)
    for ev in CLIP_EVS:
        gate(f"G8/clip/ev{ev}", f"{cand} beats the inverse inside clipped pixels at +{ev} EV (lower error, CI above zero)",
             _region(root, f"cp8_clip_ev{ev}", cand, "clipped_err_stops"), _ok_positive, ("clipped_err_stops",))
        r = _region(root, f"cp8_clip_ev{ev}", cand, "crushed_err_stops")
        if r is not None:
            reports[f"crushed/ev{ev}"] = r
    for metric in ("clipped_err_stops", "crushed_err_stops"):
        r = _region(root, "cp_real", cand, metric)
        if r is not None:
            reports[f"real/{metric}"] = r
    return {"candidate": cand, "gates": gates, "reports": reports,
            "verdict": ("not run" if any(g["passed"] is None for g in gates.values())
                        else "PASS" if all(g["passed"] for g in gates.values()) else "FAIL")}


def fmt(entry: dict) -> str:
    out = []
    for m, unit in (("pu_psnr_db", "dB"), ("cvvdp_jod", "JOD"), ("clipped_err_stops", "st"), ("crushed_err_stops", "st")):
        x = entry.get(m) or {}
        if x.get("frames"):
            out.append(f"{x['mean']:+.3f} {unit} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] {x['wins']}/{x['frames']}")
    return "   ".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bench-root", type=Path, default=REPO / "bench")
    ap.add_argument("--candidate", default="v8")
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "logs" / "cp8_results.json")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    s = score(args.bench_root, args.candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"v8 gate, candidate {args.candidate} (region rows: inverse error minus candidate error, positive = candidate better)")
    for name, g in s["gates"].items():
        print(f"  {g['status']:<8} {name:<14} {g['question']}")
        if g["status"] != "not run":
            print(f"           {fmt(g)}")
    if s["reports"]:
        print("reports")
        for name, r in s["reports"].items():
            print(f"           {name:<26} {fmt(r)}")
    print(f"\nverdict: {s['verdict']}   -> {args.out}")
    return 1 if (args.strict and any(g["passed"] is False for g in s["gates"].values())) else 0


if __name__ == "__main__":
    raise SystemExit(main())
