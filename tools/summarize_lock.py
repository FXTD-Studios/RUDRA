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
    ap.add_argument("--margin", nargs="+", help="arms judged by the 6 Nov margin rule (RUN_LOCK_STRENGTH.ps1)")
    ap.add_argument("--also", help="one more arm for the margin rule: <folder>=<arm>, e.g. G:/rudra_training/lockdq_20261105=b040")
    a = ap.parse_args(argv)
    if a.margin:
        return by_margin(a)
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


def arm_result(out, b):
    """One lock arm's gates (dev, grades, video, fresh) -> (passes, aligned, fresh 0 EV RMS, lines)."""
    lines, ok, al, f0 = [], True, None, None
    acc = out / f"dev_{b}" / "paired_development" / "acceptance.json"
    if acc.exists():
        A = json.loads(acc.read_text())
        al = g(A, "new_aligned_residual_mse.candidate")
        fails = [k for k, v in (A.get("checks") or {}).items() if v is not True and k != "outside_full_exact"]
        ok &= not fails and al is not None and al <= LIM["aligned"]
        lines.append(f"  aligned residual   {al if al is None else f'{al:.6f}'}  limit {LIM['aligned']:.6f}  {'pass' if al is not None and al <= LIM['aligned'] else 'FAIL'}"
                     + ("" if not fails else "  (" + ", ".join(fails) + ")"))
    else:
        ok = False; lines.append("  development gates: not run")
    gr = out / f"grade_{b}.json"
    if gr.exists():
        G = json.loads(gr.read_text())
        for ev in ("-3", "0", "3"):
            v = g(G, f"summary.grades.{ev}.constrained") or g(G, f"grades.{ev}.constrained")
            good = v is not None and v <= LIM[ev]; ok &= good
            lines.append(f"  old24 {ev:>2s} EV        {v if v is None else f'{v:.6g}'}  limit {LIM[ev]:.6g}  {'pass' if good else 'FAIL'}")
    else:
        ok = False; lines.append("  grades: not run")
    vd = out / f"video_{b}.json"
    if vd.exists():
        V = json.loads(vd.read_text())
        fails = [k for k, v in (V.get("checks") or {}).items() if v is not True]
        ok &= not fails
        lines.append(f"  Hunyuan48          max {V.get('max_transition_mean_ev')}  {'all pass' if not fails else 'FAIL: ' + ', '.join(fails)}")
    else:
        ok = False; lines.append("  video: not run")
    fr = out / "fresh_w4.json"
    S = json.loads(fr.read_text())["summary"] if fr.exists() else {}
    an, lk = S.get("analytic"), S.get(f"w4:lock_{b}")
    if lk and an:
        rms = lambda x: math.sqrt(max(x, 0.0))
        f0 = rms(lk["grade_0"]); lim0 = rms(an["grade_0"]) + .1
        f_ok = f0 <= lim0 and lk["grade_-3"] <= an["grade_-3"] and lk["grade_3"] <= an["grade_3"] and lk["H"] <= an["H"]
        ok &= f_ok
        lines.append(f"  fresh 0 EV RMS     {f0:.3f} (limit {lim0:.3f}) | -3 {lk['grade_-3']:.1f} | +3 {lk['grade_3']:.3f} | H {lk['H']:.4f}  {'pass' if f_ok else 'FAIL'}")
    else:
        ok = False; lines.append("  fresh: not run")
    return ok, al, f0, lines


def by_margin(a) -> int:
    """The 6 Nov rule: among passing arms (these and the earlier lock_b040), the largest smallest relative margin."""
    arms = [(a.out, b) for b in a.margin]
    if a.also:
        d, b = a.also.split("=", 1)
        arms.append((Path(d), b))
    best, cand = None, {}
    for out, b in arms:
        ok, al, f0, lines = arm_result(out, b)
        print(f"\n=== w4 + lock {b}   ({out.name})")
        print("\n".join(lines))
        if ok:
            m = min((LIM["aligned"] - al) / LIM["aligned"], (.338 - f0) / .338)
            cand[b] = m
            print(f"  PASSES every gate; smallest relative margin {m * 100:.1f} %")
        else:
            print("  does not pass")
    print()
    if cand:
        best = max(cand, key=cand.get)
        print(f"CHOSEN for the Crew holdout (largest smallest margin): {best} ({cand[best] * 100:.1f} %). "
              + ", ".join(f"{k} {v * 100:.1f} %" for k, v in sorted(cand.items(), key=lambda kv: -kv[1])))
    else:
        print("No arm passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
