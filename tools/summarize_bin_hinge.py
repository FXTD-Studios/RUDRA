"""Decision table for the bin-hinge arms (scripts/RUN_BIN_HINGE.ps1).

Reads each arm's run folder (the registration's "out") and prints every number
the registered rule names, with pass/fail. Reads only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LIM = {"aligned": .010367255277136186, "-3": 18.619855931827, "0": .060381588259977956, "3": .0025028764686429055}


def g(o, path, default=None):
    for k in path.split("."):
        if not isinstance(o, dict) or k not in o:
            return default
        o = o[k]
    return o


def mark(ok):
    return "pass" if ok else "FAIL"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["w4", "w16"])
    ap.add_argument("--experiments", type=Path, default=Path("configs/experiments"))
    a = ap.parse_args(argv)
    verdicts = {}
    for arm in a.arms:
        reg = json.loads((a.experiments / f"research_bin_hinge_{arm}_20261102.json").read_text())
        out = Path(reg["out"])
        print(f"\n=== {arm}  (bin weight {reg['bin_weight']:g}, tolerance {reg['bin_tol_codes']:g} code)  {out}")
        ok_all = True
        rep = out / "report.json"
        if rep.exists():
            h = json.loads(rep.read_text()).get("history", [])
            sh = [x for x in h if "over_one_code_share" in x]
            if sh:
                first, last = sh[:100], sh[-100:]
                avg = lambda xs, k: sum(x[k] for x in xs) / len(xs)
                print(f"  training: pixels over 1 code {avg(first,'over_one_code_share'):.3f} -> {avg(last,'over_one_code_share'):.3f}; "
                      f"over tolerance {avg(first,'over_tol_share'):.3f} -> {avg(last,'over_tol_share'):.3f}  (first vs last 100 steps)")
        acc = out / "paired_development" / "acceptance.json"
        if acc.exists():
            A = json.loads(acc.read_text())
            al = g(A, "new_aligned_residual_mse.candidate")
            print(f"  aligned residual        {al:.6f}  limit {LIM['aligned']:.6f}  {mark(al is not None and al <= LIM['aligned'])}")
            failed = [k for k, v in (A.get("checks") or {}).items() if v is not True and k != "outside_full_exact"]
            print(f"  paired checks           {'all pass' if not failed else 'FAIL: ' + ', '.join(failed)}  (outside_full_exact is False by design)")
            ok_all &= not failed and al is not None and al <= LIM["aligned"]
        else:
            print("  paired gates: not run"); ok_all = False
        gr = out / "grade.json"
        if gr.exists():
            G = json.loads(gr.read_text())
            for ev in ("-3", "0", "3"):
                v = g(G, f"summary.grades.{ev}.constrained") or g(G, f"grades.{ev}.constrained")
                an = g(G, f"summary.grades.{ev}.analytic") or g(G, f"grades.{ev}.analytic")
                ok = v is not None and v <= LIM[ev]
                print(f"  old24 grade {ev:>2s} EV      {v if v is None else f'{v:.6g}'}  analytic {an if an is None else f'{an:.6g}'}  limit {LIM[ev]:.6g}  {mark(ok)}")
                ok_all &= ok
        else:
            print("  grades: not run"); ok_all = False
        vd = out / "video.json"
        if vd.exists():
            V = json.loads(vd.read_text())
            failed = [k for k, v in (V.get("checks") or {}).items() if v is not True]
            mt = V.get("max_transition_mean_ev")
            print(f"  Hunyuan48               max transition {mt}  checks {'all pass' if not failed else 'FAIL: ' + ', '.join(failed)}")
            ok_all &= not failed
        else:
            print("  video: not run"); ok_all = False
        fr = out / "fresh.json"
        if fr.exists():
            S = json.loads(fr.read_text())["summary"]
            an = S["analytic"]
            c = next(v for k, v in S.items() if k.endswith(":learned"))
            rms = lambda x: math.sqrt(max(x, 0.0))
            ok0 = rms(c["grade_0"]) <= rms(an["grade_0"]) + 0.1
            okm = c["grade_-3"] <= an["grade_-3"] and c["grade_3"] <= an["grade_3"] and c["H"] <= an["H"]
            print(f"  fresh 0 EV RMS          {rms(c['grade_0']):.3f}  limit {rms(an['grade_0']) + 0.1:.3f}  {mark(ok0)}")
            print(f"  fresh -3 / +3 / H       {c['grade_-3']:.2f} / {c['grade_3']:.4f} / {c['H']:.4f}  vs analytic "
                  f"{an['grade_-3']:.2f} / {an['grade_3']:.4f} / {an['H']:.4f}  {mark(okm)}")
            ok_all &= ok0 and okm
        else:
            print("  fresh: not run"); ok_all = False
        verdicts[arm] = ok_all
        print(f"  VERDICT {arm}: {'passes every registered gate (development evidence; the holdout decides)' if ok_all else 'does not pass'}")
    print("\n" + ", ".join(f"{k}: {'pass' if v else 'fail'}" for k, v in verdicts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
