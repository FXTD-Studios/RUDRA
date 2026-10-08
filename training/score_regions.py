"""Score a bench export inside the regions the SDR destroyed (roadmap 4.2).

Whole-frame PU21 and CVVDP are dominated by pixels the SDR never lost, so
they cannot say whether a model reconstructs what clipping and crushing took
away (training/measure_clipping.py found, on 10 Sep 2026, that the analytic
inverse and the model agreed to the digit on the one clipped frame it could
judge). This scores each tree of a bench export only where the SDR the
network saw was

    clipped  max(R, G, B) code >= 254/255
    crushed  max(R, G, B) code <=   1/255

as the mean absolute error in stops of Rec.2020 luminance against the
reference (luminance floored at 0.005 nits, PU21's bottom). A frame whose
region has fewer than --min-pixels pixels gets no number for it.

The SDR is the export's sdr/ tree (export_bench_pairs.py --write-sdr, which
every exposure export writes), else the manifest's sdr_path for the frame
(clean exports of real SDR, where the network saw that file).

    python training/score_regions.py bench/cp8_clip_ev1
    python training/score_regions.py bench/cp_real --test-dir baseline v8

Writes results/<tree>.regions.csv (frame, scene, clipped_pct, crushed_pct,
clipped_err_stops, crushed_err_stops) and results/<tree>.regions.json (means).
No torch; numpy, OpenCV and rudra.delivery.exr.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.delivery.exr import read_exr  # noqa: E402

DIFFUSE_WHITE_NITS = 203.0
LUMA_2020 = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)
FLOOR_NITS = 0.005
CLIPPED = 254.0 / 255.0
CRUSHED = 1.0 / 255.0
UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")   # export_bench_pairs.UNSAFE
REGION_METRICS = ("clipped_err_stops", "crushed_err_stops")


def read_sdr(path: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"unreadable SDR: {path}")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    return img.astype(np.float64) / float(np.iinfo(img.dtype).max)


def log_luma(rgb_scene_linear: np.ndarray) -> np.ndarray:
    y = np.maximum(rgb_scene_linear[..., :3].astype(np.float64), 0.0) @ LUMA_2020 * DIFFUSE_WHITE_NITS
    return np.log2(np.maximum(y, FLOOR_NITS))


def region_masks(sdr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top = sdr.max(axis=-1)
    return top >= CLIPPED - 1e-9, top <= CRUSHED + 1e-9


def region_error(ref_log: np.ndarray, test_log: np.ndarray, mask: np.ndarray, min_pixels: int) -> float:
    n = int(mask.sum())
    if n < min_pixels:
        return math.nan
    return float(np.abs(test_log[mask] - ref_log[mask]).mean())


def sdr_index(root: Path) -> dict[str, Path]:
    """asset (as the export names its files) -> the manifest's sdr_path."""
    out: dict[str, Path] = {}
    for meta in sorted(root.glob("export*.json")):
        manifest = Path(json.loads(meta.read_text(encoding="utf-8")).get("manifest", ""))
        if not manifest.is_file():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("sdr_path"):
                p = Path(r["sdr_path"])
                out[UNSAFE.sub("_", str(r["asset_id"]))] = p if p.is_absolute() else manifest.parent / p
        if out:
            break
    return out


def score_tree(root: Path, tree: str, min_pixels: int, by_manifest: dict[str, Path]) -> list[dict]:
    rows = []
    for ref_path in sorted((root / "ref").rglob("*.exr")):
        scene, asset = ref_path.parent.name, ref_path.stem
        test_path = root / tree / scene / f"{asset}.exr"
        if not test_path.is_file():
            continue
        sdr_path = root / "sdr" / scene / f"{asset}.png"
        if not sdr_path.is_file():
            sdr_path = by_manifest.get(asset)
        if sdr_path is None or not Path(sdr_path).is_file():
            raise SystemExit(f"no SDR for {scene}/{asset}: export with --write-sdr, or keep the manifest it names")
        ref, _ = read_exr(ref_path)
        test, _ = read_exr(test_path)
        sdr = read_sdr(Path(sdr_path))
        if sdr.shape[:2] != ref.shape[:2]:
            import cv2
            sdr = cv2.resize(sdr, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
        clipped, crushed = region_masks(sdr)
        rl, tl = log_luma(ref), log_luma(test)
        rows.append({"frame": asset, "scene": scene,
                     "clipped_pct": 100.0 * float(clipped.mean()), "crushed_pct": 100.0 * float(crushed.mean()),
                     "clipped_err_stops": region_error(rl, tl, clipped, min_pixels),
                     "crushed_err_stops": region_error(rl, tl, crushed, min_pixels)})
    return rows


def summarise(rows: list[dict]) -> dict:
    out = {"frames": len(rows)}
    for k in ("clipped_pct", "crushed_pct") + REGION_METRICS:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        out[k] = {"frames": int(v.size), "mean": float(v.mean()) if v.size else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench", type=Path)
    ap.add_argument("--test-dir", nargs="*", default=None, help="trees to score (default: every tree but ref and sdr)")
    ap.add_argument("--min-pixels", type=int, default=256)
    args = ap.parse_args(argv)
    root = args.bench
    if not (root / "ref").is_dir():
        raise SystemExit(f"{root} has no ref/ tree")
    trees = args.test_dir or sorted(p.name for p in root.iterdir()
                                    if p.is_dir() and p.name not in ("ref", "sdr", "results"))
    by_manifest = sdr_index(root)
    (root / "results").mkdir(exist_ok=True)
    for tree in trees:
        rows = score_tree(root, tree, args.min_pixels, by_manifest)
        if not rows:
            print(f"{tree}: no frames")
            continue
        with open(root / "results" / f"{tree}.regions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        s = summarise(rows)
        (root / "results" / f"{tree}.regions.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        def m(k):
            x = s[k]
            return f"{x['mean']:.3f} st on {x['frames']}" if x["mean"] is not None else "n/a"
        print(f"{tree:14s} {s['frames']} frames  clipped {s['clipped_pct']['mean']:.2f} % of pixels, error {m('clipped_err_stops')}"
              f"  |  crushed {s['crushed_pct']['mean']:.2f} %, error {m('crushed_err_stops')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
