"""Write a bench tree of the clip fill with PeakNet's peaks (18 Oct 2026).

The twin of `rudra-native bench-clip-fill`: for every <bench>/ref/<scene>/<asset>.exr it
reads <bench>/sdr/<scene>/<asset>.png, runs the shipped fill through clip_fill_probe
with the overrides PeakNet predicts (training/peaknet/infer.py), and writes
<bench>/<tree>/<scene>/<asset>.exr: scene-linear, diffuse white 1.0, half -- the units
of the export's clipfill/ and baseline/ trees. Then:

    python training/score_regions.py <bench> --test-dir <tree> clipfill baseline

``--no-model`` writes the plain fill instead, which must reproduce the clipfill/ tree:
the check that this path and the native CLI agree before PeakNet is scored on it.

    python training/peaknet/bench_tree.py bench/cp8_clip_ev1 --model checkpoints/peaknet_v2/peaknet.onnx \
        --probe build/clip_fill_probe --tree peaknet_v2
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.peaknet.build_dataset import run_probe  # noqa: E402
from training.peaknet.infer import load_peaknet, predict_overrides  # noqa: E402

TO_SCENE = 10000.0 / 203.0


def read_sdr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / float(np.iinfo(img.dtype).max)).astype(np.float32)


def write_exr_half(path: Path, rgb: np.ndarray) -> None:
    """Uncompressed half scanline EXR (rudra.delivery.exr), which score_regions.py reads."""
    from rudra.delivery.exr import write_exr
    path.parent.mkdir(parents=True, exist_ok=True)
    write_exr(path, rgb, half=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench", type=Path)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--time-budget", type=float, default=0.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if not args.no_model and args.model is None:
        raise SystemExit("--model or --no-model")
    model = None if args.no_model else load_peaknet(args.model)
    refs = sorted((args.bench / "ref").rglob("*.exr"))
    i, n = map(int, args.shard.split("/"))
    refs = refs[i::n]
    tmp, t0, wrote = Path(tempfile.gettempdir()), time.time(), 0
    for ref in refs:
        if args.time_budget and time.time() - t0 > args.time_budget:
            break
        scene, asset = ref.parent.name, ref.stem
        out = args.bench / args.tree / scene / f"{asset}.exr"
        if out.exists() and not args.force:
            continue
        codes = read_sdr(args.bench / "sdr" / scene / f"{asset}.png")
        overrides = [] if model is None else predict_overrides(model, codes, args.probe, tmp)
        plain = args.bench / "clipfill" / scene / f"{asset}.exr"
        if model is not None and not overrides and plain.exists():
            # Nothing fully clipped: PeakNet has nothing to set, so the frame IS the plain fill.
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(plain, out)
            wrote += 1
            continue
        filled, _ = run_probe(args.probe, codes, overrides=overrides or None, tmp=tmp)
        write_exr_half(out, filled * np.float32(TO_SCENE))
        wrote += 1
    print(f"{args.bench} {args.tree}: wrote {wrote} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
