"""Pixel-level near-duplicate screen: evaluation images vs a training manifest.

Outstanding item from independent_eval_prep_20261019.json: SI-HDR references
were only string-screened against training. This compares log-tonemapped,
mean-normalised 32x32 luminance thumbnails (plus mirrored variants) of every
evaluation image against every TRAIN row's HDR target; pairs above the cosine
threshold are written for visual review. A screen, not proof: crops, other
viewing directions of a panorama and heavy reframing can escape it. Read-only.

    python pipeline/screen_eval_overlap.py --eval-dir <SI-HDR reference dir> \
        --manifest <run data/manifest.jsonl> --out overlap.json [--threshold 0.97]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def thumb(linear: np.ndarray) -> np.ndarray:
    import cv2
    y = np.asarray(linear, np.float64)[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    y = np.log2(np.maximum(np.nan_to_num(y), 0) + 1e-6)
    t = cv2.resize(y.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA).ravel()
    t = t - t.mean()
    return t / max(float(np.linalg.norm(t)), 1e-12)


def variants(linear: np.ndarray) -> list[np.ndarray]:
    return [thumb(linear), thumb(linear[:, ::-1])]


def screen(eval_items: dict, train_items: dict, threshold: float = 0.97) -> dict:
    """eval/train: name -> thumbnail list. Returns flags (name pairs) and best match per eval item."""
    names = list(train_items)
    bank = np.stack([train_items[n][0] for n in names]) if names else np.zeros((0, 1024))
    flags, best = [], {}
    for e, ts in eval_items.items():
        sims = np.max(np.stack([bank @ t for t in ts]), axis=0) if names else np.zeros(0)
        i = int(np.argmax(sims)) if sims.size else -1
        best[e] = {"train": names[i] if i >= 0 else None, "cosine": float(sims[i]) if i >= 0 else None}
        flags += [{"eval": e, "train": names[j], "cosine": float(sims[j])} for j in np.where(sims >= threshold)[0]]
    return {"threshold": threshold, "eval_items": len(eval_items), "train_items": len(names),
            "flags": sorted(flags, key=lambda f: -f["cosine"]), "best_match": best,
            "limits": "thumbnail screen; crops/other panorama views may escape; flags need visual review"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.97)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit("output exists")
    from training.exr_any import read_rgb
    from training.sdr2hdr_dataset import load_rgb
    evals = {p.name: variants(read_rgb(p)) for p in sorted(a.eval_dir.rglob("*.exr"))}
    rows = [json.loads(l) for l in a.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    train = {}
    for n, r in enumerate(r for r in rows if r.get("split") == "train"):
        train[r["asset_id"]] = [thumb(load_rgb(r["hdr_path"], hdr=True))]
        if n % 250 == 0:
            print(f"[overlap] {n} train targets", flush=True)
    result = screen(evals, train, a.threshold)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"flags": len(result["flags"]), "eval": len(evals), "train": len(train)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
