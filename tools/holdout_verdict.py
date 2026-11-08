#!/usr/bin/env python3
"""The Crew holdout's verdict under its registered rule (7 Nov 2026).

tools/evaluate_route_arms.py writes the per-case metrics and its own strict gates (0 EV no worse
than the analytic inverse). The lock arms were registered under the tolerant rule instead
(lockstrength_fresh_20261106, lockguard_fresh_20261107), and the holdout registration
(holdout_crew_eval_20261107.json) keeps it, relative to the holdout's own analytic inverse:

    0 EV RMS (codes) <= analytic 0 EV RMS + 0.1
    -3 EV, +3 EV and H no worse than analytic
    no new negative luminance; no pixel left more than 1 code off (appearance)

    python tools/holdout_verdict.py G:/rudra_training/holdout_crew_20261107/eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

MARGIN_CODES = 0.1


def verdict(result: dict) -> dict:
    """The registered gates for the evaluation's primary arm, each with its numbers."""
    s, p = result["summary"], result["primary"]
    an, lk = s["analytic"], s[p]
    rms = lambda x: math.sqrt(max(x, 0.0))   # noqa: E731
    lim0 = rms(an["grade_0"]) + MARGIN_CODES
    gates = {
        "grade_0_rms": {"value": rms(lk["grade_0"]), "limit": lim0, "pass": rms(lk["grade_0"]) <= lim0},
        "grade_-3": {"value": lk["grade_-3"], "limit": an["grade_-3"], "pass": lk["grade_-3"] <= an["grade_-3"]},
        "grade_+3": {"value": lk["grade_3"], "limit": an["grade_3"], "pass": lk["grade_3"] <= an["grade_3"]},
        "highlight_H": {"value": lk["H"], "limit": an["H"], "pass": lk["H"] <= an["H"]},
        "no_new_negativeY": {"value": lk["new_negative_y"], "limit": 0, "pass": lk["new_negative_y"] == 0},
        "appearance": {"value": lk["remaining_display_errors"], "limit": 0, "pass": lk["remaining_display_errors"] == 0},
    }
    return {"primary": p, "cases": result["cases"], "scenes": result["scenes"], "gates": gates,
            "passes": all(g["pass"] for g in gates.values())}


def main(argv: list[str] | None = None) -> int:
    """Print the gates and the verdict; exit 0 on a pass, 1 on a fail."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("evaluation", type=Path)
    ap.add_argument("--out", type=Path, help="also write the verdict as JSON")
    a = ap.parse_args(argv)
    v = verdict(json.loads(a.evaluation.read_text(encoding="utf-8")))
    print(f"Crew holdout, {v['primary']}: {v['cases']} cases, {v['scenes']} scenes")
    for k, g in v["gates"].items():
        print(f"  {k:17s} {g['value']:.6g}  limit {g['limit']:.6g}  {'pass' if g['pass'] else 'FAIL'}")
    print("PASSES every registered gate" if v["passes"] else "FAILS (see above)")
    if a.out:
        a.out.write_text(json.dumps(v, indent=1), encoding="utf-8")
    return 0 if v["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
