"""Decision table for the per-channel lock (scripts/RUN_LOCK.ps1). Reads only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LIM = {"aligned": .010367255277136186, "-3": 18.619855931827, "0": .060381588259977956, "3": .0025028764686429055}


def g(o, path):
    for k in path.split("."):
        if not isinstance(o, dict) or k not in o:
            return None
        o = o[k]
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["w4", "d6400"])
    ap.add_argument("--bands", nargs="+", help="band-lock arms on w4 (RUN_LOCK_BAND.ps1): b015 b025 b035")
    a = ap.parse_args(argv)
    if a.bands:
        return bands(a)
    verdicts = {}
    for arm in a.arms:
        print(f"\n=== {arm} + lock")
        ok = True
        acc = a.out / f"dev_{arm}" / "paired_development" / "acceptance.json"
        if acc.exists():
            A = json.loads(acc.read_text())
            al = g(A, "new_aligned_residual_mse.candidate")
            fails = [k for k, v in (A.get("checks") or {}).items() if v is not True and k != "outside_full_exact"]
            print(f"  aligned residual   {al if al is None else f'{al:.6f}'}  limit {LIM['aligned']:.6f}  {'pass' if al is not None and al <= LIM['aligned'] else 'FAIL'}")
            print(f"  paired checks      {'all pass' if not fails else 'FAIL: ' + ', '.join(fails)}")
            ok &= not fails and al is not None and al <= LIM["aligned"]
        else:
            print("  development gates: not run"); ok = False
        gr = a.out / f"grade_{arm}.json"
        if gr.exists():
            G = json.loads(gr.read_text())
            for ev in ("-3", "0", "3"):
                v = g(G, f"summary.grades.{ev}.constrained") or g(G, f"grades.{ev}.constrained")
                good = v is not None and v <= LIM[ev]
                print(f"  old24 {ev:>2s} EV        {v if v is None else f'{v:.6g}'}  limit {LIM[ev]:.6g}  {'pass' if good else 'FAIL'}")
                ok &= good
        else:
            print("  grades: not run"); ok = False
        vd = a.out / f"video_{arm}.json"
        if vd.exists():
            V = json.loads(vd.read_text())
            fails = [k for k, v in (V.get("checks") or {}).items() if v is not True]
            print(f"  Hunyuan48          max {V.get('max_transition_mean_ev')}  {'all pass' if not fails else 'FAIL: ' + ', '.join(fails)}")
            ok &= not fails
        else:
            print("  video: not run"); ok = False
        fr = a.out / f"fresh_{arm}.json"
        if fr.exists():
            S = json.loads(fr.read_text())["summary"]
            an = S["analytic"]
            rms = lambda x: math.sqrt(max(x, 0.0))
            for name, val in S.items():
                if name == "analytic":
                    continue
                print(f"  fresh {name:16s} 0 EV RMS {rms(val['grade_0']):.3f} (limit {rms(an['grade_0']) + .1:.3f}) | -3 {val['grade_-3']:.1f} | +3 {val['grade_3']:.3f} | H {val['H']:.4f}")
            lk = next((v for k, v in S.items() if k.endswith(":lock")), None)
            if lk:
                f_ok = rms(lk["grade_0"]) <= rms(an["grade_0"]) + .1 and lk["grade_-3"] <= an["grade_-3"] and lk["grade_3"] <= an["grade_3"] and lk["H"] <= an["H"]
                print(f"  fresh verdict      {'pass' if f_ok else 'FAIL'}  (analytic: 0 EV RMS {rms(an['grade_0']):.3f}, -3 {an['grade_-3']:.1f}, +3 {an['grade_3']:.3f}, H {an['H']:.4f})")
                ok &= f_ok
        else:
            print("  fresh: not run"); ok = False
        verdicts[arm] = ok
        print(f"  VERDICT {arm} + lock: {'passes every registered gate: next, the Crew holdout, once' if ok else 'does not pass'}")
    print("\n" + ", ".join(f"{k}: {'pass' if v else 'fail'}" for k, v in verdicts.items()))
    return 0


def bands(a) -> int:
    """The band arms: each band's gates, then the registered choice (lowest fresh 0 EV among those passing)."""
    fr = a.out / "fresh_w4.json"
    S = json.loads(fr.read_text())["summary"] if fr.exists() else {}
    an = S.get("analytic")
    rms = lambda x: math.sqrt(max(x, 0.0))
    passing = {}
    for b in a.bands:
        print(f"\n=== w4 + lock band {b}")
        ok = True
        acc = a.out / f"dev_{b}" / "paired_development" / "acceptance.json"
        if acc.exists():
            A = json.loads(acc.read_text())
            al = g(A, "new_aligned_residual_mse.candidate")
            fails = [k for k, v in (A.get("checks") or {}).items() if v is not True and k != "outside_full_exact"]
            print(f"  aligned residual   {al if al is None else f'{al:.6f}'}  limit {LIM['aligned']:.6f}  {'pass' if al is not None and al <= LIM['aligned'] else 'FAIL'}")
            print(f"  paired checks      {'all pass' if not fails else 'FAIL: ' + ', '.join(fails)}")
            ok &= not fails and al is not None and al <= LIM["aligned"]
        else:
            print("  development gates: not run"); ok = False
        gr = a.out / f"grade_{b}.json"
        if gr.exists():
            G = json.loads(gr.read_text())
            for ev in ("-3", "0", "3"):
                v = g(G, f"summary.grades.{ev}.constrained") or g(G, f"grades.{ev}.constrained")
                good = v is not None and v <= LIM[ev]
                print(f"  old24 {ev:>2s} EV        {v if v is None else f'{v:.6g}'}  limit {LIM[ev]:.6g}  {'pass' if good else 'FAIL'}")
                ok &= good
        else:
            print("  grades: not run"); ok = False
        vd = a.out / f"video_{b}.json"
        if vd.exists():
            V = json.loads(vd.read_text())
            fails = [k for k, v in (V.get("checks") or {}).items() if v is not True]
            print(f"  Hunyuan48          max {V.get('max_transition_mean_ev')}  {'all pass' if not fails else 'FAIL: ' + ', '.join(fails)}")
            ok &= not fails
        else:
            print("  video: not run"); ok = False
        lk = S.get(f"w4:lock_{b}")
        if lk and an:
            f_ok = rms(lk["grade_0"]) <= rms(an["grade_0"]) + .1 and lk["grade_-3"] <= an["grade_-3"] and lk["grade_3"] <= an["grade_3"] and lk["H"] <= an["H"]
            print(f"  fresh 0 EV RMS     {rms(lk['grade_0']):.3f} (limit {rms(an['grade_0']) + .1:.3f}) | -3 {lk['grade_-3']:.1f} | +3 {lk['grade_3']:.3f} | H {lk['H']:.4f}  {'pass' if f_ok else 'FAIL'}")
            ok &= f_ok
            if ok:
                passing[b] = rms(lk["grade_0"])
        else:
            print("  fresh: not run"); ok = False
        print(f"  {'PASSES every registered gate' if ok else 'does not pass'}")
    print()
    if passing:
        best = min(passing, key=passing.get)
        print(f"CHOSEN (registered rule: lowest fresh 0 EV among those passing): band {best} (fresh 0 EV RMS {passing[best]:.3f}). Next: the Crew holdout, once.")
    else:
        print("No band passes every gate: no candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
