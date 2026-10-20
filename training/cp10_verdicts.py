"""The v10 gate, written 13 Oct 2026, before any v10 checkpoint exists.

    python training/cp10_verdicts.py --candidate v10            # bench/ -> reports/logs/cp10_results_v10.json

v10 rides on the Source picker's curve, so it is judged against what RUDRA 1.0
ships without a model: the picker's baseline (BT.1886 by default) and, inside
clipped highlights, the clip fill. Every row compares two trees of the same
bench export (training/paired_gate.py, paired bootstrap, 95 % CI):

  G10/real          Meridian real SDR, the model on BT.1886 against the BT.1886
                    tree (baseline_bt1886): PU21 and CVVDP, both CIs above zero.
  G10/aces          the v4b ACES bench, the model told "aces", against the ACES
                    inverse (baseline): not worse, both CIs reach zero.
  G10/oog|mix       out-of-generator and v4c benches, the model on the default
                    BT.1886 against baseline_bt1886: not worse.
  G10/clip/ev0|1|2  inside the SDR's clipped pixels (Meridian re-rendered with
                    ACES at +0, +1, +2 EV), against the clip fill (clipfill):
                    lower error, CI above zero, at every EV.
  G10/crushed/*     inside crushed pixels, real SDR and each clip bench: not
                    worse than the reference tree, CI reaches zero (v8 and v9
                    were 1.6 to 2.4 stops worse, 0 of 287 frames better).

A row whose files are missing reads "not run". With --strict, only a complete
PASS exits zero. --scene-bootstrap is a supplemental equal-scene analysis;
the default preserves the historical frame-bootstrap thresholds. No torch.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.paired_gate import METRICS, compare, load  # noqa: E402

CLIP_EVS = (0, 1, 2)
PICKER = "baseline_bt1886"


def _pos(r: dict, metrics) -> bool:
    return all(r[m].get("frames") and r[m]["ci95"][0] > 0 for m in metrics)


def _not_worse(r: dict, metrics) -> bool:
    return all(r[m].get("frames") and r[m]["ci95"][1] >= 0 for m in metrics)


def _paired(a, b, metrics=METRICS, scene_bootstrap=False):
    scenes = None
    if scene_bootstrap:
        maps = []
        for path in (a, b):
            with path.open(newline="", encoding="utf-8") as handle:
                maps.append({r.get("frame") or r.get("asset_id"): r.get("scene") or r.get("scene_id")
                             for r in csv.DictReader(handle)})
        if maps[0] != maps[1]:
            raise ValueError("candidate/reference scene assignments differ")
        scenes = maps[0]
    return compare(load(a, metrics), load(b, metrics), metrics=metrics,
                   require_same_frames=True, scenes=scenes)


def _whole(root: Path, bench: str, cand: str, against: str, scene_bootstrap=False):
    a = root / f"cp_{bench}" / "results" / f"{cand}.csv"
    b = root / f"cp_{bench}" / "results" / f"{against}.csv"
    if not (a.is_file() and b.is_file()):
        return None
    return _paired(a, b, scene_bootstrap=scene_bootstrap)


def _region(root: Path, bench_dir: str, cand: str, metric: str, against: str, scene_bootstrap=False):
    a = root / bench_dir / "results" / f"{cand}.regions.csv"
    b = root / bench_dir / "results" / f"{against}.regions.csv"
    if not (a.is_file() and b.is_file()):
        return None
    m = (metric,)
    # error: lower is better, so reference - candidate > 0 means the candidate wins
    return _paired(b, a, m, scene_bootstrap)


def score(root: Path, cand: str, scene_bootstrap=False) -> dict:
    gates = {}

    def gate(name, question, r, ok, metrics):
        gates[name] = {"question": question, "status": "not run", "passed": None}
        if r is not None:
            passed = ok(r, metrics)
            gates[name].update({"status": "PASS" if passed else "FAIL", "passed": passed, **{m: r[m] for m in metrics}})

    gate("G10/real", f"{cand} on BT.1886 beats the BT.1886 baseline on real SDR (Meridian), PU21 and CVVDP CIs above zero",
         _whole(root, "real", cand, PICKER, scene_bootstrap), _pos, METRICS)
    gate("G10/aces", f"{cand} told aces is not worse than the ACES inverse on the aces bench",
         _whole(root, "aces", cand, "baseline", scene_bootstrap), _not_worse, METRICS)
    for bench in ("oog", "mix"):
        gate(f"G10/{bench}", f"{cand} on the default BT.1886 is not worse than the BT.1886 baseline on the {bench} bench",
             _whole(root, bench, cand, PICKER, scene_bootstrap), _not_worse, METRICS)
    for ev in CLIP_EVS:
        gate(f"G10/clip/ev{ev}", f"{cand} beats the clip fill inside clipped pixels at +{ev} EV (CI above zero)",
             _region(root, f"cp8_clip_ev{ev}", cand, "clipped_err_stops", "clipfill", scene_bootstrap), _pos, ("clipped_err_stops",))
    gate("G10/crushed/real", f"{cand} is not worse than the BT.1886 baseline inside crushed pixels on real SDR",
         _region(root, "cp_real", cand, "crushed_err_stops", PICKER, scene_bootstrap), _not_worse, ("crushed_err_stops",))
    for ev in CLIP_EVS:
        gate(f"G10/crushed/ev{ev}", f"{cand} is not worse than the ACES inverse inside crushed pixels at +{ev} EV",
             _region(root, f"cp8_clip_ev{ev}", cand, "crushed_err_stops", "baseline", scene_bootstrap), _not_worse, ("crushed_err_stops",))
    verdict = ("FAIL" if any(g["passed"] is False for g in gates.values())
               else "not run" if any(g["passed"] is None for g in gates.values()) else "PASS")
    return {"candidate": cand, "gates": gates, "verdict": verdict,
            "bootstrap_unit": "scene" if scene_bootstrap else "frame"}


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
    ap.add_argument("--candidate", default="v10")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--scene-bootstrap", action="store_true")
    args = ap.parse_args(argv)
    try:
        s = score(args.bench_root, args.candidate, args.scene_bootstrap)
    except ValueError as exc:
        print(f"Invalid benchmark: {exc}", file=sys.stderr)
        return 2
    suffix = "_scene" if args.scene_bootstrap else ""
    out = args.out or REPO / "reports" / "logs" / f"cp10_results_{args.candidate}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"v10 gate, candidate {args.candidate} (positive = candidate better)")
    for name, g in s["gates"].items():
        print(f"  {g['status']:<8} {name:<18} {g['question']}")
        if g["status"] != "not run":
            print(f"           {fmt(g)}")
    print(f"\nverdict: {s['verdict']}   -> {out}")
    return 1 if (args.strict and s["verdict"] != "PASS") else 0


if __name__ == "__main__":
    raise SystemExit(main())
