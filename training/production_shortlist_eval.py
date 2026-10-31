"""Run a frozen checkpoint shortlist through the production validation kit.

One command, many candidates, no hand-copying between steps:

1. read a shortlist JSON frozen BEFORE any full-resolution score exists
   (candidates + references, each with path and SHA256);
2. verify every checkpoint's bytes, then score each one with the unchanged
   ``production_eval_kit.evaluate`` into ``<root>/<label>/`` -- a directory that
   already holds a finished ``comparison.json`` for the same checkpoint bytes and
   kit is reused as evidence, anything else that exists is refused;
3. compare every candidate against every reference, and the candidates against
   each other, with ``production_candidate_compare.compare_runs``;
4. apply the selection rule written into the shortlist (or the default below)
   and write ``summary.json``.

The rule only *proposes* a candidate for the next stage (reserved test set,
temporal, independent and display review). It never copies, renames or promotes
weights, and ``release_authorized`` is always false.

    python training/production_shortlist_eval.py --shortlist shortlist.json \
        --kit <validation_kit> --root <new_or_existing_eval_root> [--dry-run]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.production_eval_kit import digest  # noqa: E402

DEFAULT_RULE = {
    "gate": "validation PU gate: clean lower CI >= -0.1 dB AND hard lower CI > 0 dB (vs bt1886 baseline)",
    "not_worse_than_references": "no candidate whose candidate-minus-reference CI is entirely below 0 on either condition",
    "rank": "highest hard mean gain vs baseline; within 0.05 dB prefer higher clean mean gain",
}


def load_shortlist(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = [*data.get("candidates", []), *data.get("references", [])]
    if not data.get("candidates") or not data.get("references"):
        raise ValueError("shortlist needs at least one candidate and one reference")
    labels = [e.get("label") for e in entries]
    if any(not isinstance(x, str) or not x or not x.replace("_", "").isalnum() for x in labels):
        raise ValueError("every entry needs a simple alphanumeric/underscore label")
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate labels")
    for e in entries:
        h = e.get("sha256", "")
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError(f"{e['label']}: sha256 missing or malformed")
    if len({e["sha256"] for e in entries}) != len(entries):
        raise ValueError("two shortlist entries are the same checkpoint bytes")
    return data


def verify_checkpoints(entries: list[dict]) -> None:
    for e in entries:
        path = Path(e["path"])
        if not path.is_file():
            raise FileNotFoundError(f"{e['label']}: {path}")
        actual = digest(path)
        if actual != e["sha256"]:
            raise ValueError(f"{e['label']}: checkpoint bytes changed since the shortlist was frozen")


def finished_run(out: Path, entry: dict, kit: Path) -> bool:
    """True when out already holds this exact evaluation; raise on any other content."""
    if not out.exists():
        return False
    report_path = out / "comparison.json"
    if not report_path.is_file():
        raise ValueError(f"{out} exists without comparison.json (incomplete evaluation); "
                         "inspect it and choose a new --root -- it is not overwritten")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("checkpoint_sha256") != entry["sha256"]:
        raise ValueError(f"{out} holds a different checkpoint's evaluation")
    if report.get("contract_sha256") != digest(kit / "contract.json"):
        raise ValueError(f"{out} came from a different validation kit")
    return True


def _gain(report: dict, condition: str) -> dict:
    m = report["conditions"][condition]["metrics"]["pu_psnr_db"]
    return {"mean": m["mean"], "ci95": m["ci95"], "scenes": m.get("scenes"), "frames": m["frames"],
            "passed": bool(report["conditions"][condition]["passed"])}


def worst_cases(run: Path, condition: str, n: int = 5) -> dict:
    """Per-scene and per-frame model-minus-baseline, worst first, for visual follow-up."""
    test = {(r["clip"], r["frame"]): r["pu_psnr_db"]
            for r in json.loads((run / condition / "test.json").read_text())["results"]}
    base = {(r["clip"], r["frame"]): r["pu_psnr_db"]
            for r in json.loads((run / condition / "baseline_bt1886.json").read_text())["results"]}
    diffs = {k: test[k] - base[k] for k in test}
    scenes: dict[str, list[float]] = {}
    for (clip, _), d in diffs.items():
        scenes.setdefault(clip, []).append(d)
    per_scene = sorted(((c, sum(v) / len(v), len(v)) for c, v in scenes.items()), key=lambda x: x[1])
    frames = sorted(diffs.items(), key=lambda kv: kv[1])[:n]
    return {"scenes_below_baseline": sum(1 for _, m, _ in per_scene if m < 0),
            "worst_scenes": [dict(scene=c, mean_gain_db=m, frames=k) for c, m, k in per_scene[:n]],
            "worst_frames": [dict(scene=c, frame=f, gain_db=d) for (c, f), d in frames]}


def select(summary: dict) -> dict:
    """Apply the frozen rule; returns the proposal and why every other candidate lost."""
    eligible, reasons = [], {}
    for label, c in summary["candidates"].items():
        if not c["validation_pu_gate_passed"]:
            reasons[label] = "failed validation PU gate"
            continue
        worse = [f"{ref}/{cond}" for ref, conds in c["vs_references"].items()
                 for cond, v in conds.items() if v["ci_verdict"] == "b_better"]
        if worse:
            reasons[label] = "CI-worse than reference on " + ", ".join(worse)
            continue
        eligible.append(label)
    eligible.sort(key=lambda l: summary["candidates"][l]["hard"]["mean"], reverse=True)
    if len(eligible) > 1:
        top = summary["candidates"][eligible[0]]["hard"]["mean"]
        near = [l for l in eligible if top - summary["candidates"][l]["hard"]["mean"] <= 0.05]
        near.sort(key=lambda l: summary["candidates"][l]["clean"]["mean"], reverse=True)
        eligible = near + [l for l in eligible if l not in near]
    for l in eligible[1:]:
        reasons[l] = f"ranked below {eligible[0]} by the frozen rule"
    return {"proposed_for_next_stage": eligible[0] if eligible else None,
            "ranking": eligible, "not_selected": reasons,
            "meaning": ("Validation-stage proposal only. Next: reserved test set, temporal, independent "
                        "and calibrated-display review. Not production approval.")}


def run(shortlist_path: Path, kit: Path, root: Path, dry_run: bool = False, evaluate=None, compare=None) -> dict:
    from training.production_candidate_compare import compare_runs
    from training.production_eval_kit import evaluate as kit_evaluate
    evaluate = evaluate or kit_evaluate
    compare = compare or compare_runs
    shortlist = load_shortlist(shortlist_path)
    candidates, references = shortlist["candidates"], shortlist["references"]
    entries = [*candidates, *references]
    verify_checkpoints(entries)
    plan = {e["label"]: dict(checkpoint=e["path"], out=str(root / e["label"]),
                             reuse=finished_run(root / e["label"], e, kit)) for e in entries}
    if dry_run:
        return {"dry_run": True, "plan": plan, "release_authorized": False}
    root.mkdir(parents=True, exist_ok=True)
    reports = {}
    for e in entries:
        out = root / e["label"]
        if not plan[e["label"]]["reuse"]:
            evaluate(kit, Path(e["path"]), out)
        reports[e["label"]] = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    verify_checkpoints(entries)  # unchanged across the whole batch, not only per evaluation
    pairs_dir = root / "_pairs"
    pairs_dir.mkdir(exist_ok=True)

    def pair(a: str, b: str) -> dict:
        target = pairs_dir / f"{a}__vs__{b}.json"
        if target.exists():
            result = json.loads(target.read_text(encoding="utf-8"))
        else:
            result = compare(kit, root / a, root / b)
            target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        return {c: dict(mean=v["a_minus_b"]["mean"], ci95=v["a_minus_b"]["ci95"],
                        ci_verdict=v["ci_verdict"], scenes_a_worse=v["scenes_a_worse"])
                for c, v in result["conditions"].items()}

    summary = {"shortlist": str(shortlist_path), "shortlist_sha256": digest(shortlist_path),
               "kit_contract_sha256": digest(kit / "contract.json"),
               "rule": shortlist.get("selection_rule", DEFAULT_RULE),
               "candidates": {}, "references": {}, "candidate_pairs": {},
               "release_authorized": False, "promotion": False,
               "not_measured": ["Temporal stability", "Independent real SDR", "CVVDP / perceptual video",
                                ">1000-nit highlights (validation targets rarely exceed 1000 nits)",
                                "Calibrated HDR display review", "Reserved test set"]}
    for group, items in (("candidates", candidates), ("references", references)):
        for e in items:
            rep = reports[e["label"]]
            row = {"checkpoint": e["path"], "sha256": e["sha256"], "step": e.get("step"),
                   "clean": _gain(rep, "clean"), "hard": _gain(rep, "hard"),
                   "validation_pu_gate_passed": bool(rep["validation_pu_gate_passed"]),
                   "worst": {c: worst_cases(root / e["label"], c) for c in ("clean", "hard")}}
            for v in (row["clean"]["mean"], row["hard"]["mean"]):
                if not math.isfinite(v):
                    raise ValueError(f"{e['label']}: nonfinite summary metric")
            summary[group][e["label"]] = row
    for c in candidates:
        summary["candidates"][c["label"]]["vs_references"] = {
            r["label"]: pair(c["label"], r["label"]) for r in references}
    for a, b in itertools.combinations([c["label"] for c in candidates], 2):
        summary["candidate_pairs"][f"{a}__vs__{b}"] = pair(a, b)
    summary["selection"] = select(summary)
    out = root / "summary.json"
    if out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        if previous.get("shortlist_sha256") != summary["shortlist_sha256"]:
            raise ValueError("summary.json exists for a different shortlist; use a new --root")
    out.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortlist", type=Path, required=True)
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="verify hashes and print the plan only")
    args = ap.parse_args()
    result = run(args.shortlist, args.kit, args.root, args.dry_run)
    if args.dry_run:
        print(json.dumps(result, indent=2))
    else:
        brief = {k: dict(clean=v["clean"]["ci95"], hard=v["hard"]["ci95"], gate=v["validation_pu_gate_passed"])
                 for k, v in {**result["candidates"], **result["references"]}.items()}
        print(json.dumps(dict(gains_vs_baseline_ci95=brief, selection=result["selection"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
