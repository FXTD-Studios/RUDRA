"""Group saved benchmark frames by their reference HDR brightness (read-only).

Reuses frame-level ``bench_val.jsonl`` evidence: no inference, no media reads.
Each validation frame's reference mean luminance comes from the manifest row
(``hdr_mean_nits``, recorded when the candidate set was prepared). Gains are
model minus picker baseline; positive favours the model.

    python tools/frame_luminance_breakdown.py --manifest M.jsonl \
        --bench det=reports/x/control/bench_val.jsonl --out out.json

``--baseline-mismatch`` adds the analytic table behind the 14 Oct 2026
diagnosis: exposure-push renders read through the BT.1886 and ACES picker
baselines (needs torch; CPU only).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BINS = ((0.0, 10.0), (10.0, 30.0), (30.0, 100.0), (100.0, math.inf))
DARK_NITS = 10.0


def _ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = np.asarray(values)[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = _ranks(np.asarray(x, float)), _ranks(np.asarray(y, float))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _summary(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    return {"frames": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None}


def breakdown(report, manifest_rows):
    """One bench report (a bench_val.jsonl line) -> luminance-grouped gains."""
    nits = {}
    for row in manifest_rows:
        if row.get("hdr_mean_nits") is not None:
            nits[row["asset_id"]] = float(row["hdr_mean_nits"])
    frames = report["frames"]
    missing = sorted({f["asset_id"] for f in frames if f["asset_id"] not in nits})
    if missing:
        raise ValueError(f"manifest lacks hdr_mean_nits for {len(missing)} frames, e.g. {missing[0]}")
    ids = [(f["asset_id"], f["condition"], f["ev"]) for f in frames]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate benchmark observations")
    out = {"step": report["step"], "comparison": report.get("comparison"), "conditions": []}
    conditions = sorted({(f["condition"], f["ev"]) for f in frames}, key=lambda c: (c[0] != "real", c[1] or 0))
    for condition, ev in conditions:
        rows = [{**f, "hdr_mean_nits": nits[f["asset_id"]]} for f in frames
                if f["condition"] == condition and f["ev"] == ev]
        scored = [r for r in rows if r.get("pu_gain_db") is not None]
        entry = {"condition": condition, "ev": ev, "all": _summary(rows, "pu_gain_db"),
                 "crushed_all": _summary(rows, "crushed_gain_stops"),
                 "bins": [{"min_nits": lo, "max_nits": None if math.isinf(hi) else hi,
                           **_summary([r for r in rows if lo <= r["hdr_mean_nits"] < hi], "pu_gain_db"),
                           "crushed": _summary([r for r in rows if lo <= r["hdr_mean_nits"] < hi],
                                               "crushed_gain_stops")}
                          for lo, hi in BINS],
                 "excluding_dark_frames": _summary([r for r in rows if r["hdr_mean_nits"] >= DARK_NITS],
                                                   "pu_gain_db"),
                 "spearman_gain_vs_log_mean_nits": spearman(
                     [math.log(max(r["hdr_mean_nits"], 1e-6)) for r in scored],
                     [r["pu_gain_db"] for r in scored]),
                 "worst": sorted(({"asset_id": r["asset_id"], "hdr_mean_nits": r["hdr_mean_nits"],
                                   "pu_gain_db": r["pu_gain_db"]} for r in scored),
                                 key=lambda r: r["pu_gain_db"])[:5]}
        out["conditions"].append(entry)
    return out


def baseline_mismatch(nits=(0.05, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000), evs=(0.0, 1.0, 2.0)):
    """Error, in stops, of each picker baseline on a training exposure-push render."""
    import torch
    from rudra.sdr2hdr import SDR2HDRNet
    from rudra.source_baseline import baseline_curve_index
    from training.sdr2hdr_dataset import exposure_push
    net = SDR2HDRNet(baseline_source=True, corpus_ev=0.0).eval()
    level = torch.tensor(nits, dtype=torch.float32) / 10000.0
    hdr = level.view(1, 1, -1).expand(3, 1, -1).contiguous()
    rows = []
    for ev in evs:
        sdr, target, _ = exposure_push(hdr, ev)
        errors = {}
        for curve in ("bt1886", "aces"):
            base = net.picker_baseline(sdr[None], baseline_curve_index(curve))[0]
            errors[curve] = [round(float(v), 3) for v in torch.log2(base.clamp_min(1e-9) / target)[1, 0]]
        rows.append({"ev": ev, "target_nits": [round(float(v) * 10000.0, 4) for v in target[1, 0]],
                     "code": [int(round(float(v) * 255)) for v in sdr[1, 0]],
                     "error_stops": errors})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--bench", action="append", default=[],
                        help="label=path/to/bench_val.jsonl[@step]; default step is the last line")
    parser.add_argument("--baseline-mismatch", action="store_true")
    parser.add_argument("--out", type=Path)
    a = parser.parse_args(argv)
    manifest = [json.loads(line) for line in a.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {"tool": "frame_luminance_breakdown", "manifest": str(a.manifest),
              "dark_nits_threshold": DARK_NITS, "runs": {}}
    for spec in a.bench:
        label, _, rest = spec.partition("=")
        path, _, step = rest.partition("@")
        reports = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        report = next(r for r in reports if r["step"] == int(step)) if step else reports[-1]
        result["runs"][label] = {"bench": path, **breakdown(report, manifest)}
    if a.baseline_mismatch:
        result["baseline_mismatch"] = baseline_mismatch()
    text = json.dumps(result, indent=2, allow_nan=False)
    if a.out:
        if a.out.exists():
            raise SystemExit(f"{a.out} exists; evidence is never overwritten")
        a.out.write_text(text + "\n", encoding="utf-8")
    print(text if not a.out else json.dumps({k: {c["condition"] + str(c["ev"] or ""): c["all"]
                                                 for c in v["conditions"]} for k, v in result["runs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
