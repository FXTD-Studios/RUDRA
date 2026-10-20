"""Read-only: do real training pairs ask for a dark lift over the BT.1886 baseline?

For sampled manifest rows, compares each SDR channel's BT.1886 picker baseline
(nits, corpus_ev 0, as rudra/source_baseline.py + sdr_to_baseline_hdr) with
the HDR target channel (pipeline/hdr_io decode, as training reads it), binned
by 8-bit SDR code. Reports the median log2(target / baseline) per bin, per
title and split. Numpy/OpenCV only; never writes next to the media.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline.hdr_io import HDRStorage, decode_hdr_u16  # noqa: E402

_STORAGE = {}


def storage_for(hdr_path):
    """Same sentinel search as training/sdr2hdr_dataset.storage_for, without torch."""
    directory = Path(hdr_path).parent
    if directory not in _STORAGE:
        found = None
        for candidate in (directory, *list(directory.parents)[:3]):
            sentinel = candidate / "_ingest_config.json"
            if sentinel.exists():
                block = json.loads(sentinel.read_text(encoding="utf-8")).get("storage")
                found = HDRStorage.from_dict(block) if block else None
                break
        if found is None:
            raise ValueError(f"no storage sentinel for {hdr_path}")
        _STORAGE[directory] = found
    return _STORAGE[directory]

BINS = ((1, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 256))


def srgb_eotf(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def bt709_oetf(x):
    return np.where(x < 0.018, 4.5 * x, 1.099 * np.maximum(x, 0.0) ** 0.45 - 0.099)


def bt1886_baseline_nits(code01):
    """The picker's BT.1886 entry at corpus_ev 0 (its knots are exact at 8-bit codes)."""
    display = np.clip(srgb_eotf(code01), 0.0, 0.995)
    return np.maximum(bt709_oetf(display), 0.0) ** 2.4 * 203.0


def read(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"unreadable {path}")
    return cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)


def frame_stats(row, root_map, step):
    sdr_p = Path(root_map(row["sdr_path"]))
    hdr_p = Path(root_map(row["hdr_path"]))
    sdr = read(sdr_p)[::step, ::step]
    hdr = read(hdr_p)[::step, ::step]
    sdr01 = sdr.astype(np.float64) / float(np.iinfo(sdr.dtype).max)
    nits = decode_hdr_u16(hdr, storage_for(hdr_p)).astype(np.float64) * storage_for(hdr_p).diffuse_white_nits
    code = np.rint(sdr01 * 255).astype(int)                # the training network's 8-bit view
    base = bt1886_baseline_nits(code / 255.0)
    out = {"zero_code_target_nits": [float(v) for v in np.percentile(nits[code == 0], [50, 90])] if (code == 0).any() else None,
           "zero_code_fraction": float((code == 0).mean()), "bins": []}
    for lo, hi in BINS:
        m = (code >= lo) & (code < hi)
        if m.sum() < 50:
            out["bins"].append(None)
            continue
        r = np.log2(np.maximum(nits[m], 1e-4) / base[m])
        out["bins"].append({"n": int(m.sum()), "median_log2": float(np.median(r)),
                            "p25": float(np.percentile(r, 25)), "p75": float(np.percentile(r, 75))})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--per-title", type=int, default=40)
    ap.add_argument("--step", type=int, default=4, help="pixel subsample stride")
    ap.add_argument("--map", nargs=2, metavar=("FROM", "TO"), help="path prefix remap for a mounted copy")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; evidence is never overwritten")
    rows = [json.loads(l) for l in a.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    def root_map(p):
        if a.map and p.startswith(a.map[0]):
            p = a.map[1] + p[len(a.map[0]):].replace("\\", "/")
        return p
    by_title = {}
    for r in rows:
        by_title.setdefault((r["title"], r["split"]), []).append(r)
    started, result = time.time(), {"tool": "tools/dark_tone_targets.py", "bins": BINS, "step": a.step, "groups": {}}
    for (title, split), group in sorted(by_title.items()):
        idx = np.linspace(0, len(group) - 1, min(a.per_title, len(group))).round().astype(int)
        frames = [dict(asset_id=group[i]["asset_id"], hdr_mean_nits=group[i].get("hdr_mean_nits"),
                       **frame_stats(group[i], root_map, a.step)) for i in sorted(set(idx))]
        summary = []
        for b in range(len(BINS)):
            vals = [f["bins"][b] for f in frames if f["bins"][b]]
            summary.append(None if not vals else {
                "frames": len(vals), "pixels": sum(v["n"] for v in vals),
                "median_of_frame_medians_log2": float(np.median([v["median_log2"] for v in vals])),
                "pixel_weighted_mean_of_medians_log2": float(np.average([v["median_log2"] for v in vals],
                                                                          weights=[v["n"] for v in vals]))})
        result["groups"][f"{title}|{split}"] = {"rows": len(group), "sampled": len(frames), "summary": summary,
                                                "frames": frames}
        print(title, split, len(frames), [None if s is None else round(s["median_of_frame_medians_log2"], 2) for s in summary],
              f"{time.time() - started:.0f}s", flush=True)
    result["seconds"] = time.time() - started
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as h:
        json.dump(result, h, indent=1)


if __name__ == "__main__":
    main()
