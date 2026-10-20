"""Inference-only matched-frame diagnosis of an existing pilot checkpoint.

Preserves trainer, weights, manifests and source data. Writes a new report.
Not a promotion benchmark; all comparisons reuse a previously selected val set.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline.commercial_preflight import sha256
from rudra.sdr2hdr import SDR2HDRNet
from rudra.source_baseline import BASELINE_CURVES, baseline_curve_index
from training.bench_val import BenchVal, fit, pu_psnr
from training.sdr2hdr_dataset import (load_rgb, _crop_params, _resize_crop,
                                    effective_baseline_curve, read_jsonl)
from tools.diagnose_bench_gap import frame_stats


def routing_probe():
    return [{"kind": kind, "pushed": pushed, "degraded": degraded,
             "actual": BASELINE_CURVES[effective_baseline_curve(
                 {"sdr_kind": kind, "source_curve": "aces"}, degraded, pushed, True, 0.0)],
             "documented_expected": "bt1886" if degraded or (kind == "real" and not pushed) else "aces"}
            for kind in ("real", "rendered") for pushed in (False, True) for degraded in (False, True)]


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        for view, values in row["views"].items():
            groups[view].append(values)
    return {name: {"frames": len(v), **{
        key: {"mean": float(np.mean([x[key] for x in v])),
              "median": float(np.median([x[key] for x in v])),
              "positive_frames": int(sum(x[key] > 0 for x in v))}
        for key in v[0] if key.endswith("gain_db")}}
        for name, v in groups.items()}


@torch.inference_mode()
def diagnose(manifest, checkpoint, out, seconds=300):
    if out.exists():
        raise ValueError("use a new diagnostic output directory")
    if not out.resolve().is_relative_to((REPO / "reports").resolve()):
        raise ValueError("diagnostic output must be under repository reports")
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SDR2HDRNet.from_config(saved["config"]).eval().to(device)
    model.load_state_dict(saved["model"], strict=True)
    bv = BenchVal.from_manifest(str(manifest), real_frames=48, rendered_frames=0)
    rows = []
    started = time.monotonic()
    bt = baseline_curve_index("bt1886")
    cpu_array = lambda x: x.float().cpu().numpy() * 10000.0

    def infer(sdr):
        o = model(sdr[None].to(device), baseline_curve=bt)
        return o.hdr[0], o.analytic_baseline[0]

    def score(pred, base, ref, bands=False):
        p, b, r = (cpu_array(x) for x in (pred, base, ref))
        ps, bs = frame_stats(p, r), frame_stats(b, r)
        result = {"pu_gain_db": ps["pu_psnr"] - bs["pu_psnr"],
                  "log_gain_db": ps["train_psnr"] - bs["train_psnr"]}
        if bands:
            result["bands_model"], result["bands_baseline"] = ps["bands"], bs["bands"]
        return result

    with (out / "frames.jsonl").open("w", encoding="utf-8") as journal:
        for i, record in enumerate(bv.records):
            if time.monotonic() - started > seconds:
                raise TimeoutError("diagnostic time cap; partial frame journal retained")
            s = torch.from_numpy(load_rgb(record["sdr_path"], False)).permute(2, 0, 1)
            h = torch.from_numpy(load_rgb(record["hdr_path"], True)).permute(2, 0, 1)
            params = _crop_params(*s.shape[-2:], 256, False)
            sc, hc = _resize_crop(s, params, 256), _resize_crop(h, params, 256)
            # Whole image at exactly the scale/interpolator used before crop.
            full_params = (params[0], params[1], 0, 0)
            sf = _resize_crop(s, full_params, max(params[:2]))
            hf = _resize_crop(h, full_params, max(params[:2]))
            sb, hb = fit(s, 960), fit(h, 960)
            views = {}
            for name, x, ref in (("train_center_crop", sc, hc),
                                  ("full_at_train_scale", sf, hf),
                                  ("bench960_fp32", sb, hb),
                                  ("bench960_half_cache", sb.half().float(), hb.half().float())):
                p, b = infer(x)
                views[name] = score(p, b, ref, bands=name == "bench960_half_cache")
                if name == "bench960_half_cache":
                    # Same pixels/resolution, crop origin aligned to both strides.
                    y, xx = ((d - 256) // 8 * 4 for d in x.shape[-2:])
                    cropped = x[:, y:y+256, xx:xx+256]
                    pc, bc = infer(cropped)
                    refc = ref[:, y:y+256, xx:xx+256]
                    inner = (slice(None), slice(64, 192), slice(64, 192))
                    full_center = p[:, y:y+256, xx:xx+256]
                    views["same_pixels_full_context"] = score(full_center[inner], b[:, y:y+256, xx:xx+256][inner], refc[inner])
                    views["same_pixels_crop_context"] = score(pc[inner], bc[inner], refc[inner])
                    views["context_difference"] = {
                        "mean_absolute_nits": float((pc[inner] - full_center[inner]).abs().mean()) * 10000,
                        "baseline_max_absolute_nits": float((bc - b[:, y:y+256, xx:xx+256]).abs().max()) * 10000}
                    if i == 0:
                        oracle = pu_psnr(p * 10000, ref.to(device) * 10000) - pu_psnr(b * 10000, ref.to(device) * 10000)
                        assert abs(oracle - views[name]["pu_gain_db"]) < 1e-4
            row = {"asset_id": record["asset_id"], "scene_id": record["scene_id"], "views": views}
            rows.append(row); journal.write(json.dumps(row) + "\n"); journal.flush()
            if (i + 1) % 8 == 0:
                print(f"diagnosed {i+1}/{len(bv.records)} matched frames", flush=True)
    all_rows = read_jsonl(manifest)
    train = [r for r in all_rows if r["split"] == "train"]
    scenes_by_title = defaultdict(set)
    for row in train:
        scenes_by_title[row["title"]].add(row["scene_id"])
    summary = {"checkpoint_sha256": sha256(checkpoint), "checkpoint_step": saved["step"],
               "manifest_sha256": sha256(manifest), "algorithm_sha256": sha256(Path(__file__)),
               "elapsed_seconds": time.monotonic() - started, "device": str(device),
               "frames": len(rows), "scene_frames": dict(Counter(r["scene_id"] for r in rows)),
               "aggregates": aggregate(rows), "routing_probe": routing_probe(),
               "train_scenes_by_title": {k: len(v) for k, v in scenes_by_title.items()},
               "train_rows_by_title": dict(Counter(r["title"] for r in train)),
               "train_ceiling_metadata_rows": sum(bool(r.get("ceiling_nits")) for r in train),
               "limitations": ["existing 48-frame Sparks sample only", "diagnosis, not promotion or a held-out experiment", "context differences do not isolate GroupNorm from every other spatial effect", "no weights or training recipe changed"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "routing_probe"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=300)
    a = parser.parse_args()
    diagnose(a.manifest, a.checkpoint, a.out, a.seconds)
