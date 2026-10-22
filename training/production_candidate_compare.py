"""Paired candidate-vs-candidate comparison of two production validation runs.

`evaluate_hdr_production.ps1` scores ONE checkpoint against the bt1886
baseline. Selecting between saved checkpoints (e.g. a new native1080 step vs
old highlight step 7000) needs the two candidates compared directly on the
same frames. This reads two finished evaluation directories, read-only:

* both must come from the same frozen kit (identical contract hash, which must
  match the kit on disk) and from different checkpoint files;
* every condition must cover exactly the contract's frames and scenes, with
  finite PU-PSNR (strict_scores) and no duplicates;
* the bt1886 baseline scores of both runs must agree per frame, proving the
  two runs saw identical inputs through the identical scoring path;
* the difference A - B gets an equal-scene bootstrap CI (paired_gate.compare)
  and a per-scene table sorted worst-first for visual follow-up.

It never selects, copies or promotes a checkpoint and always writes
release_authorized=false. "unresolved" is the honest answer when the CI spans 0.

    python training/production_candidate_compare.py --kit <kit> --a <runA> --b <runB> --out cmp.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.paired_gate import compare                    # noqa: E402
from training.production_eval_kit import digest, strict_scores  # noqa: E402

BASELINE = "baseline_bt1886"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scores(run: Path, condition: str, name: str, contract: dict) -> dict:
    results = _load(run / condition / f"{name}.json")["results"]
    if len(results) != contract["frames"]:
        raise ValueError(f"{run.name}/{condition}/{name}: {len(results)} frames, contract {contract['frames']}")
    expected = {(r["clip"], r["frame"]): r["clip"] for r in results}
    scores = strict_scores(results, expected)
    if len({k[0] for k in scores}) != contract["scenes"]:
        raise ValueError(f"{run.name}/{condition}/{name}: scene count differs from contract")
    return scores


def compare_runs(kit: Path, run_a: Path, run_b: Path, baseline_tol_db: float = 1e-6) -> dict:
    contract = _load(kit / "contract.json")
    kit_hash = digest(kit / "contract.json")
    if digest(kit / "validation.jsonl") != contract["validation_sha256"]:
        raise ValueError("Frozen validation manifest changed")
    reports = {}
    for label, run in (("a", run_a), ("b", run_b)):
        rep = _load(run / "comparison.json")
        if rep.get("contract_sha256") != kit_hash:
            raise ValueError(f"run {label} was not produced from this kit")
        if rep.get("release_authorized") is not False:
            raise ValueError(f"run {label} lacks release_authorized=false")
        reports[label] = rep
    if reports["a"]["checkpoint_sha256"] == reports["b"]["checkpoint_sha256"]:
        raise ValueError("both runs scored the same checkpoint bytes")
    out = dict(kit=str(kit), contract_sha256=kit_hash,
               a=dict(run=str(run_a), checkpoint=reports["a"]["checkpoint"],
                      checkpoint_sha256=reports["a"]["checkpoint_sha256"]),
               b=dict(run=str(run_b), checkpoint=reports["b"]["checkpoint"],
                      checkpoint_sha256=reports["b"]["checkpoint_sha256"]),
               conditions={}, release_authorized=False, promotion=False,
               not_measured=["Temporal stability", "Independent real SDR", ">1000-nit highlights "
                             "(validation targets rarely exceed 1000 nits)", "HDR display review"])
    for condition in contract["conditions"]:
        a = _scores(run_a, condition, "test", contract)
        b = _scores(run_b, condition, "test", contract)
        base_a = _scores(run_a, condition, BASELINE, contract)
        base_b = _scores(run_b, condition, BASELINE, contract)
        if a.keys() != b.keys() or base_a.keys() != a.keys() or base_b.keys() != a.keys():
            raise ValueError(f"{condition}: frame keys differ between runs")
        drift = max(abs(base_a[k]["pu_psnr_db"] - base_b[k]["pu_psnr_db"]) for k in a)
        if drift > baseline_tol_db:
            raise ValueError(f"{condition}: baseline scores differ by {drift:.3g} dB; inputs/path not identical")
        scenes = {k: k[0] for k in a}
        stats = compare(a, b, metrics=("pu_psnr_db",), require_same_frames=True, scenes=scenes)["pu_psnr_db"]
        lo, hi = stats["ci95"]
        per_scene = {}
        for k in a:
            s = per_scene.setdefault(k[0], dict(frames=0, a_minus_b=0.0, a_vs_base=0.0, b_vs_base=0.0))
            s["frames"] += 1
            s["a_minus_b"] += a[k]["pu_psnr_db"] - b[k]["pu_psnr_db"]
            s["a_vs_base"] += a[k]["pu_psnr_db"] - base_a[k]["pu_psnr_db"]
            s["b_vs_base"] += b[k]["pu_psnr_db"] - base_b[k]["pu_psnr_db"]
        for s in per_scene.values():
            for f in ("a_minus_b", "a_vs_base", "b_vs_base"):
                s[f] = s[f] / s["frames"]
        ranked = sorted(per_scene.items(), key=lambda kv: kv[1]["a_minus_b"])
        out["conditions"][condition] = dict(
            a_minus_b=stats,
            ci_verdict="a_better" if lo > 0 else "b_better" if hi < 0 else "unresolved",
            scenes_a_worse=sum(1 for _, s in ranked if s["a_minus_b"] < 0),
            per_scene_worst_first=[dict(scene=k, **v) for k, v in ranked],
            baseline_max_abs_drift_db=drift)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("output exists; use a new path")
    result = compare_runs(args.kit, args.a, args.b)
    text = json.dumps(result, indent=2, allow_nan=False)
    args.out.write_text(text, encoding="utf-8")
    print(json.dumps({c: dict(ci95=v["a_minus_b"]["ci95"], verdict=v["ci_verdict"])
                      for c, v in result["conditions"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
