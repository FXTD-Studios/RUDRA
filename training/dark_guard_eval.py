"""Select dark-guard settings on the VALIDATION exports, with the rule frozen in this file.

Reads a finished production_eval_kit evaluation directory (ref/, test/ and
baseline_bt1886/ EXRs for clean and hard) -- no new inference -- applies each
candidate guard (rudra.dark_guard) to the model frames and scores it with the
bench's own loader and PU21-PSNR. Unguarded scores must reproduce the
directory's test.json exactly, proving identical frames and metric.

Frozen selection rule (written before any guarded score existed, 22 Oct 2026):
  1. validation PU gate on the full kit: clean lower CI >= -0.1 dB AND hard lower CI > 0;
  2. dark slice = validation frames with >= 10% of reference pixels below 0.1 nits
     (counts reported). Its mean gain vs baseline must be >= -0.1 dB, clean and hard.
     If the slice spans < 3 scenes the criterion is reported as NOT TESTABLE;
  3. among settings meeting 1 and 2 (or 1 when 2 is not testable), the highest
     full-kit hard mean gain; within 0.02 dB prefer the setting with larger `hi`
     (more protection).
The reserved Mataha test set is NOT read: it is spent and must not tune this.

    python training/dark_guard_eval.py --run <eval dir, e.g. shortlist_run02_20261021\\run02_best_step16200> \
        --kit <validation_kit> --out <new summary.json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRID = [dict(lo=lo, hi=hi, feather_px=f) for lo, hi in ((0.02, 0.1), (0.05, 0.3), (0.1, 0.5), (0.2, 1.0))
        for f in (0, 8)]
DARK_NITS, DARK_FRACTION, MIN_DARK_SCENES = 0.1, 0.10, 3
NITS_SCALE = 203.0


def label(s: dict) -> str:
    return f"lo{s['lo']}_hi{s['hi']}_f{s['feather_px']}"


def score_condition(cond_dir: Path, settings: list[dict]) -> list[dict]:
    from rudra.delivery.bench import load_frame, pu_psnr
    from rudra.dark_guard import apply_dark_guard
    recorded = {(r["clip"], r["frame"]): r["pu_psnr_db"]
                for r in json.loads((cond_dir / "test.json").read_text())["results"]}
    base_rec = {(r["clip"], r["frame"]): r["pu_psnr_db"]
                for r in json.loads((cond_dir / "baseline_bt1886.json").read_text())["results"]}
    rows = []
    for (clip, frame), value in sorted(recorded.items()):
        rel = Path(clip) / f"{frame}.exr"
        ref = load_frame(cond_dir / "ref" / rel, NITS_SCALE)
        model = load_frame(cond_dir / "test" / rel, NITS_SCALE)
        base = load_frame(cond_dir / "baseline_bt1886" / rel, NITS_SCALE)
        unguarded = pu_psnr(model, ref)
        if not math.isclose(unguarded, value, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"{clip}/{frame}: recomputed {unguarded} != recorded {value}; not the same frames/metric")
        y_ref = ref @ np.array([0.2627, 0.6780, 0.0593])
        row = {"clip": clip, "frame": frame, "baseline": base_rec[(clip, frame)], "model": unguarded,
               "dark_fraction": float((y_ref < DARK_NITS).mean()), "guarded": {}}
        for s in settings:
            # The bench clamps negatives to 0 in load_frame; guarded frames get the same treatment.
            row["guarded"][label(s)] = pu_psnr(np.maximum(apply_dark_guard(model, base, **s), 0.0), ref)
        rows.append(row)
    return rows


def ci(rows: list[dict], key) -> dict:
    from training.paired_gate import compare
    a = {(r["clip"], r["frame"]): {"m": key(r)} for r in rows}
    b = {(r["clip"], r["frame"]): {"m": r["baseline"]} for r in rows}
    return compare(a, b, metrics=("m",), require_same_frames=True, scenes={k: k[0] for k in a})["m"]


def summarize(rows_by_cond: dict, settings: list[dict]) -> dict:
    names = ["unguarded"] + [label(s) for s in settings]
    get = {n: (lambda r: r["model"]) if n == "unguarded" else (lambda r, n=n: r["guarded"][n]) for n in names}
    out = {}
    dark_scenes = {c: {r["clip"] for r in rows if r["dark_fraction"] >= DARK_FRACTION} for c, rows in rows_by_cond.items()}
    for n in names:
        entry = {}
        for cond, rows in rows_by_cond.items():
            full = ci(rows, get[n])
            dark = [r for r in rows if r["dark_fraction"] >= DARK_FRACTION]
            dark_mean = float(np.mean([get[n](r) - r["baseline"] for r in dark])) if dark else None
            worst = sorted(rows, key=lambda r: get[n](r) - r["baseline"])[:5]
            entry[cond] = {"full": full, "dark_frames": len(dark), "dark_scenes": len(dark_scenes[cond]),
                           "dark_mean_gain_db": dark_mean,
                           "worst": [dict(clip=r["clip"], frame=r["frame"], gain_db=get[n](r) - r["baseline"]) for r in worst]}
        gate = entry["clean"]["full"]["ci95"][0] >= -0.1 and entry["hard"]["full"]["ci95"][0] > 0
        testable = all(entry[c]["dark_scenes"] >= MIN_DARK_SCENES for c in entry)
        dark_ok = all(entry[c]["dark_mean_gain_db"] is not None and entry[c]["dark_mean_gain_db"] >= -0.1 for c in entry)
        entry.update(validation_gate=gate, dark_testable=testable, dark_ok=dark_ok if testable else None)
        out[n] = entry
    eligible = [n for n in names[1:] if out[n]["validation_gate"] and (out[n]["dark_ok"] or not out[n]["dark_testable"])]
    eligible.sort(key=lambda n: out[n]["hard"]["full"]["mean"], reverse=True)
    choice = None
    if eligible:
        top = out[eligible[0]]["hard"]["full"]["mean"]
        near = [n for n in eligible if top - out[n]["hard"]["full"]["mean"] <= 0.02]
        hi_of = {label(s): s["hi"] for s in settings}
        choice = max(near, key=lambda n: hi_of[n])
    return {"settings": out, "eligible": eligible, "selected": choice,
            "dark_slice_definition": f">= {DARK_FRACTION:.0%} of reference pixels below {DARK_NITS} nits",
            "meaning": "Validation-stage guard selection only. Needs confirmation on fresh held-out dark footage; "
                       "the reserved Mataha test is spent.", "release_authorized": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit("output exists; use a new path")
    from training.production_eval_kit import digest
    report = json.loads((a.run / "comparison.json").read_text())
    if report.get("contract_sha256") != digest(a.kit / "contract.json"):
        raise SystemExit("evaluation directory was not produced from this validation kit")
    if "validation_pu_gate_passed" not in report:
        raise SystemExit("refusing: not a validation-kit evaluation (the reserved test must not tune the guard)")
    rows = {}
    for cond in ("clean", "hard"):
        rows[cond] = score_condition(a.run / cond, GRID)
        print(f"[dark-guard] {cond}: {len(rows[cond])} frames scored", flush=True)
    summary = summarize(rows, GRID)
    summary.update(run=str(a.run), checkpoint_sha256=report["checkpoint_sha256"], grid=GRID)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (a.out.with_name(a.out.stem + "_per_frame.json")).write_text(json.dumps(rows), encoding="utf-8")
    brief = {n: {c: (round(v[c]["full"]["mean"], 3), [round(x, 3) for x in v[c]["full"]["ci95"]],
                     None if v[c]["dark_mean_gain_db"] is None else round(v[c]["dark_mean_gain_db"], 3))
                 for c in ("clean", "hard")} for n, v in summary["settings"].items()}
    print(json.dumps({"full_mean_ci_and_dark_mean": brief, "selected": summary["selected"],
                      "dark_frames": {c: summary["settings"]["unguarded"][c]["dark_frames"] for c in ("clean", "hard")},
                      "dark_scenes": {c: summary["settings"]["unguarded"][c]["dark_scenes"] for c in ("clean", "hard")}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
