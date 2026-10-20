"""Sample first/middle/last frame of each shot; read-only HDR decode/alignment QA."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.commercial_preflight import sha256
from training.sdr2hdr_dataset import load_rgb, storage_for


def sample_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_id"]].append(row)
    selected = []
    for _, group in sorted(groups.items()):
        group.sort(key=lambda r: (r["frame_index"], r["asset_id"]))
        selected.extend(group[i] for i in sorted({0, len(group) // 2, len(group) - 1}))
    return selected


def alignment_proxy(sdr, luminance):
    def edges(x):
        x = cv2.resize(x, (256, 144), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        return cv2.magnitude(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))
    a, b = edges(sdr.mean(2)), edges(np.log2(np.maximum(luminance, 0.005)))
    shift, response = cv2.phaseCorrelate(a, b, cv2.createHanningWindow((256, 144), cv2.CV_32F))
    return {"shift_at_256x144": list(shift), "response": response,
            "review_flag": bool(response < 0.1 or max(map(abs, shift)) > 2)}


def inspect(row):
    entry = {k: row[k] for k in ("asset_id", "scene_id", "split")}
    try:
        s = load_rgb(row["sdr_path"], hdr=False)
        h = load_rgb(row["hdr_path"], hdr=True, ceiling=1000.0)
        lum = (h * np.array([.2627, .6780, .0593], np.float32)).sum(2) * 10000
        entry.update(sdr_shape=list(s.shape), hdr_shape=list(h.shape), shape_matches=s.shape == h.shape,
                     finite=bool(np.isfinite(h).all() and np.isfinite(s).all()),
                     hdr_peak_nits=float(h.max() * 10000), luma_p99_nits=float(np.percentile(lum, 99)),
                     sdr_clip_fraction=float((s.max(2) >= 254 / 255).mean()),
                     sdr_crushed_fraction=float((s.max(2) <= 1 / 255).mean()),
                     recorded_peak_nits=row.get("peak_nits"), storage=storage_for(row["hdr_path"]).as_dict())
        if s.shape == h.shape:
            entry["alignment_proxy"] = alignment_proxy(s, lum)
    except (ValueError, RuntimeError, OSError, cv2.error, AttributeError) as exc:
        entry["error"] = str(exc)
    return entry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(l) for l in a.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = a.out.resolve()
    if out.is_relative_to(a.manifest.resolve().parent) or any(out.is_relative_to(Path(r[k]).resolve().parent) for r in rows for k in ("sdr_path", "hdr_path")):
        raise ValueError("output must be outside input directories")
    if out.exists():
        raise ValueError("use a new output path")
    cv2.setNumThreads(2)
    samples = []
    for row in sample_rows(rows):
        samples.append(inspect(row))
        print(row["asset_id"], flush=True)
    report = {"manifest_sha256": sha256(a.manifest), "algorithm_sha256": sha256(Path(__file__)),
              "samples": samples, "limitations": ["sampled first/middle/last only", "edge phase correlation is not registration certification"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return 1 if any("error" in r or not r["shape_matches"] or r["alignment_proxy"]["review_flag"] for r in samples) else 0


if __name__ == "__main__":
    raise SystemExit(main())
