"""Reserved final-test evaluation of ONE frozen candidate (Mataha, test split).

The validation kit (production_eval_kit.py) selects; this measures the single
candidate that selection froze (freeze_candidate.py). Rules:
  * the kit is prepared from the frozen run manifest's TEST rows only, hashes
    checked, with no scene overlap with train/val;
  * evaluation requires a freeze record that still verifies, and the kit
    accepts one candidate: a second evaluation is refused, because comparing
    several candidates on the test set turns it into a validation set;
  * same exporter, settings, bt1886 baseline, units and strict coverage as the
    validation kit; same gate thresholds, reported, never tuned.
release_authorized is always false.

    python training/production_test_kit.py prepare --manifest <run data/manifest.jsonl> --out <new kit>
    python training/production_test_kit.py evaluate --kit <kit> --freeze <freeze.json> --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.production_eval_kit import digest, strict_scores  # noqa: E402


def select_test(rows):
    if len({r["asset_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate asset IDs")
    test = [r for r in rows if r.get("split") == "test"]
    if not test or any(not r.get("scene_id") for r in test):
        raise ValueError("Missing test scenes")
    scenes = {r["scene_id"] for r in test}
    if scenes & {r.get("scene_id") for r in rows if r.get("split") != "test"}:
        raise ValueError("Test scene overlaps another split")
    if len(scenes) < 2:
        raise ValueError("Need multiple independent scene groups")
    return test


def prepare(manifest: Path, out: Path) -> dict:
    rows = select_test([json.loads(l) for l in manifest.read_text().splitlines() if l.strip()])
    if out.exists():
        raise ValueError("Use a new kit directory; frozen kits are immutable")
    for r in rows:
        for kind in ("sdr", "hdr"):
            if digest(r[kind + "_path"]) != r[kind + "_sha256"]:
                raise ValueError(f"Media hash mismatch: {r['asset_id']} {kind}")
    out.mkdir(parents=True)
    frozen = out / "test.jsonl"
    frozen.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    contract = dict(source_manifest=str(manifest), source_manifest_sha256=digest(manifest),
                    test_sha256=digest(frozen), frames=len(rows), scenes=len({r["scene_id"] for r in rows}),
                    source_counts=dict(Counter(r.get("source_id", "unknown") for r in rows)),
                    sdr_kind_counts=dict(Counter(r.get("sdr_kind", "unknown") for r in rows)),
                    conditions=["clean", "hard"], nits_scale=203, tile_size=512, baseline="bt1886",
                    precision="fp32", gate=dict(clean_lower_ci_min=-0.1, hard_lower_ci_min=0),
                    one_candidate_only=True, release_authorized=False,
                    purpose="Reserved final test of one frozen candidate; never used for selection")
    (out / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


def evaluate(kit: Path, freeze_path: Path, out: Path, run=subprocess.run) -> dict:
    from training.freeze_candidate import verify
    from training.paired_gate import compare
    from training.export_bench_pairs import slug, UNSAFE
    check = verify(freeze_path)
    if not check["ok"]:
        raise ValueError(f"freeze record no longer verifies: {check['problems']}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    contract = json.loads((kit / "contract.json").read_text())
    manifest = kit / "test.jsonl"
    if digest(manifest) != contract["test_sha256"]:
        raise ValueError("Frozen test manifest changed")
    used = kit / "used_by.json"
    if used.exists():
        prior = json.loads(used.read_text())
        raise ValueError(f"test kit already used for {prior['checkpoint_sha256'][:12]} ({prior['label']}); "
                         "a second candidate on the reserved test set needs an explicit owner decision and a new kit")
    if out.exists():
        raise ValueError("Use an empty output path")
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    for r in rows:
        for kind in ("sdr", "hdr"):
            if digest(r[kind + "_path"]) != r[kind + "_sha256"]:
                raise ValueError("Test input changed")
    used.write_text(json.dumps({"label": freeze["label"], "checkpoint_sha256": freeze["checkpoint_sha256"],
                                "freeze": str(freeze_path)}), encoding="utf-8")
    expected = {(slug(r["scene_id"]), UNSAFE.sub("_", r["asset_id"])): r["scene_id"] for r in rows}
    out.mkdir(parents=True)
    ckpt = freeze["checkpoint"]
    report = dict(checkpoint=ckpt, checkpoint_sha256=freeze["checkpoint_sha256"], freeze=str(freeze_path),
                  contract_sha256=digest(kit / "contract.json"), conditions={}, release_authorized=False)
    for condition in contract["conditions"]:
        dest = out / condition
        run([sys.executable, str(ROOT / "training/export_bench_pairs.py"), "--checkpoint", ckpt,
             "--manifest", str(manifest), "--split", "test", "--out", str(dest), "--condition", condition,
             "--max-side", "0", "--tile-size", "512", "--tile-overlap", "64", "--precision", "fp32",
             "--baseline-curve", "bt1886", "--picker-baseline", "bt1886", "--no-baseline"], cwd=ROOT, check=True)
        export = json.loads((dest / "export.json").read_text())
        if export["skipped"] or export["frames"] != len(rows):
            raise ValueError("Exporter skipped test frames")
        for name in ("test", "baseline_bt1886"):
            run([sys.executable, "-m", "rudra.delivery.cli", "bench", str(dest), "--test-dir", name,
                 "--nits-scale", "203", "--output", str(dest / f"{name}.json")], cwd=ROOT, check=True)
        a, b = [strict_scores(json.loads((dest / f"{n}.json").read_text())["results"], expected)
                for n in ("test", "baseline_bt1886")]
        metrics = compare(a, b, metrics=("pu_psnr_db",), require_same_frames=True, scenes=expected)
        lower = metrics["pu_psnr_db"].get("ci95", [float("-inf")])[0]
        report["conditions"][condition] = dict(
            metrics=metrics, passed=lower >= -0.1 if condition == "clean" else lower > 0)
    if verify(freeze_path)["ok"] is not True:
        raise ValueError("checkpoint or inference code changed during the test")
    report["test_pu_gate_passed"] = all(c["passed"] for c in report["conditions"].values())
    report["not_measured"] = ["Temporal stability", "Independent real SDR", "CVVDP", "HDR display review",
                              "Commercial release clearance"]
    (out / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--kit", type=Path, required=True)
    ev.add_argument("--freeze", type=Path, required=True)
    ev.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = prepare(a.manifest, a.out) if a.action == "prepare" else evaluate(a.kit, a.freeze, a.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
