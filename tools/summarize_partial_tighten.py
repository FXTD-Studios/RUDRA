"""One table for the 0 EV arm's three evaluations (scripts/RUN_PARTIAL_TIGHTEN.ps1).

Reads what the evaluators wrote under --out and prints, per arm, every number
the registered decision rule names, with pass/fail against it. Reads only; the
evaluators' own files stay the evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEV_LIMITS = {"aligned_residual": .010367255277136186, "grade_-3": 18.619855931827, "grade_0": .060381588259977956,
              "grade_+3": .0025028764686429055}


def _find(obj, *names):
    """The first value under any of these keys, anywhere in a nested report."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in names and isinstance(v, (int, float, bool)):
                return v
        for v in obj.values():
            r = _find(v, *names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, *names)
            if r is not None:
                return r
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = []
    fresh = a.out / "fresh_arms.json"
    if fresh.exists():
        r = json.loads(fresh.read_text())
        s = r["summary"]
        an = s["analytic"]
        rms = lambda v: math.sqrt(max(v, 0.0))
        print("Fresh FXTD v2 (rebuilt; development evidence)")
        print(f"  {'arm':22s} {'grade-3':>9s} {'grade0':>8s} {'RMS0 code':>9s} {'grade+3':>8s} {'H':>7s}  tolerant 0 EV rule")
        for name, v in s.items():
            ok = rms(v["grade_0"]) <= rms(an["grade_0"]) + 0.1
            print(f"  {name:22s} {v['grade_-3']:9.3f} {v['grade_0']:8.4f} {rms(v['grade_0']):9.3f} {v['grade_3']:8.4f} {v['H']:7.4f}  "
                  f"{'' if name == 'analytic' else ('pass' if ok else 'FAIL')}")
    for arm in ("bisect", "halving"):
        print(f"\nArm {arm}" + ("  (primary)" if arm == "bisect" else "  (control: 25 Oct method)"))
        dev = a.out / f"dev_{arm}" / "paired_development"
        reports = sorted(dev.glob("*.json")) if dev.exists() else []
        if reports:
            d = {p.name: json.loads(p.read_text()) for p in reports}
            for key, limit in DEV_LIMITS.items():
                names = {"aligned_residual": ("aligned_residual", "aligned"), "grade_-3": ("grade_-3", "-3"),
                         "grade_0": ("grade_0", "0"), "grade_+3": ("grade_+3", "3", "+3")}[key]
                v = None
                for rep in d.values():
                    v = _find(rep, *names)
                    if v is not None:
                        break
                print(f"  {key:18s} {'n/a' if v is None else f'{v:.6g}':>12s}  limit {limit:.6g}  "
                      f"{'' if v is None else ('pass' if v <= limit else 'FAIL')}")
            checks = [_find(rep, "all_pass", "pass", "passed") for rep in d.values()]
            print(f"  evaluator verdicts: {checks}   (files: {', '.join(d)})")
        else:
            print("  development gates: not run")
        vid = a.out / f"video_{arm}.json"
        if vid.exists():
            v = json.loads(vid.read_text())
            flick = _find(v, "flicker_ev", "temporal_ev", "flicker")
            print(f"  Hunyuan48 flicker  {'n/a' if flick is None else f'{flick:.5f} EV'}  limit 0.020  "
                  f"{'' if flick is None else ('pass' if flick <= 0.02 else 'FAIL')}")
        else:
            print("  video: not run")
    print("\nThe registered decision rule (research_partial_tighten_fresh_20261102.json) decides; this table only collects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
