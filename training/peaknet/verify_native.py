"""Score PeakNet through the shipped clip fill itself (17 Oct 2026).

train.py scores PeakNet on E(a), the error curve build_dataset.py measured.
This re-renders the same val/test frames (deterministic per frame), hands the
predicted peaks to native/tools/clip_fill_probe as ClipOverrides -- the exact
path the app takes -- and measures the error inside each region from what the
real fill returns. Frames are weighted as in train.py (each frame once).

    python training/peaknet/verify_native.py --predictions predictions.json --strict <manifest> \
        --moves G:/rudra_v02/pairs_moves --probe build/clip_fill_probe --split val

predictions.json: {"<frame>#<exposure>#<region>": a, ...} as train.py --dump-predictions writes.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.peaknet.build_dataset import (EXPOSURES_PER_FRAME, frame_regions, frame_variant,  # noqa: E402
                                            load_linear, pano_view, plan_frames, shard_frames)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--strict", required=True, type=Path)
    ap.add_argument("--moves", type=Path, default=None)
    ap.add_argument("--panos", type=Path, default=None)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--map", action="append", default=[], metavar="WIN=LOCAL")
    ap.add_argument("--out", type=Path, required=True, help="jsonl of per-frame results (appended, resumable)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--time-budget", type=float, default=0.0)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args(argv)
    if args.summarize:
        rows = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
        d = np.array([r["default"] for r in rows]); p = np.array([r["native"] for r in rows])
        e = np.array([r["ecurve"] for r in rows])
        rng = np.random.default_rng(0)
        boots = [(d - p)[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        out = {"frames": len(rows), "default": round(float(d.mean()), 4), "peaknet_native": round(float(p.mean()), 4),
               "peaknet_ecurve": round(float(e.mean()), 4), "gain_stops": round(float((d - p).mean()), 4),
               "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
               "frames_won": int(((d - p) > 0).sum()),
               "max_abs_native_vs_ecurve": round(float(np.abs(p - e).max()), 4)}
        print(json.dumps(out, indent=2))
        return 0
    pred = json.loads(args.predictions.read_text())
    by_frame: dict[str, dict[int, float]] = defaultdict(dict)
    for key, a in pred.items():
        frame, k, region = key.rsplit("#", 2)
        by_frame[f"{frame}#{k}"][int(region)] = float(a)
    maps = dict(m.split("=", 1) for m in args.map)
    frames = [f for f in plan_frames(args.strict, args.moves, maps, args.panos) if f["split"] == args.split]
    i, n = map(int, args.shard.split("/"))
    frames = shard_frames(frames, i, n)
    done = set()
    if args.out.exists():
        done = {json.loads(line)["key"] for line in args.out.read_text().splitlines() if line.strip()}
    tmp, t0 = Path(tempfile.gettempdir()), time.time()
    with open(args.out, "a") as log:
        for f in frames:
            if args.time_budget and time.time() - t0 > args.time_budget:
                break
            keys = [f"{f['id']}#{k}" for k in range(EXPOSURES_PER_FRAME)]
            if all(k not in by_frame or k in done for k in keys):
                continue
            lin = pano_view(f) if f["kind"] == "polyhaven_pano" else load_linear(Path(f["hdr"]))
            for k, key in enumerate(keys):
                if key not in by_frame or key in done:
                    continue
                img, recipe = frame_variant(f, lin, k)
                samples = frame_regions(args.probe, img, recipe, f["ceiling_nits"], tmp,
                                        {"frame": f["id"], "exposure": k}, predicted=by_frame[key])
                samples = [s for s in samples if s["e_native"] is not None]
                if not samples:
                    continue
                w = np.array([s["pixels"] for s in samples], float)
                w /= w.sum()
                grid = np.arange(0.0, 12.0001, 0.125)
                e_curve = [float(np.interp(by_frame[key][s["meta"]["region"]], grid, s["ecurve"])) for s in samples]
                log.write(json.dumps({"key": key, "ev": recipe["render_ev"], "kind": f["kind"],
                                      "default": float(w @ [s["e_default"] for s in samples]),
                                      "native": float(w @ [s["e_native"] for s in samples]),
                                      "ecurve": float(w @ e_curve)}) + "\n")
                log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
