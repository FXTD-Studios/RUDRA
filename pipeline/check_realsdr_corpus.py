"""Checks a real-SDR corpus the way the 1 Oct 2026 Netflix pull was checked, as
one command instead of a notebook:

  1  every record's two PNGs exist and are complete (signature and IEND)
  2  a random sample decodes: SDR and HDR share a geometry, the HDR's
     recorded peak is reproduced by hdr_io within 0.1 %
  3  the pair is the same frame: log-luminance correlation between the SDR
     and the HDR at offset 0 beats the same at +/-1 frame of the same shot
  4  splits do not share a scene; per split and title counts are printed

Exit code 1 on any failure, so it can gate the manifest build.

    python pipeline/check_realsdr_corpus.py --manifest G:/.../manifest.jsonl --sample 40
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.hdr_io import HDRStorage, decode_hdr_u16, linear_to_nits  # noqa: E402

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def png_complete(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            f.seek(-12, 2)
            tail = f.read(12)
    except OSError:
        return False
    return head == PNG_SIG and tail[4:8] == b"IEND"


def read_u16(path: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"unreadable: {path}")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)


def log_luma(a: np.ndarray) -> np.ndarray:
    y = 0.2627 * a[..., 0] + 0.6780 * a[..., 1] + 0.0593 * a[..., 2]
    return np.log2(np.maximum(y, 1e-4))


def sdr_linear(code: np.ndarray) -> np.ndarray:
    c = code.astype(np.float32) / 65535.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=40, help="pairs to decode and align")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--report", type=Path, default=None, help="JSON report (default: beside the manifest)")
    args = ap.parse_args(argv)
    root = args.manifest.parent
    recs = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    storage = HDRStorage(mode="log2_extended")
    ok = True
    report: dict = {"manifest": str(args.manifest), "records": len(recs)}

    def path_of(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else root / q

    # 1 completeness
    bad = [r["asset_id"] for r in recs if not (png_complete(path_of(r["sdr_path"])) and png_complete(path_of(r["hdr_path"])))]
    report["incomplete_png"] = bad
    print(f"1 PNGs complete: {len(recs) - len(bad)} of {len(recs)}" + (f"  FAIL {bad[:5]}" if bad else ""))
    ok &= not bad

    # 4 splits and scenes
    by = Counter((r["split"], r.get("title", "?")) for r in recs)
    scenes = defaultdict(set)
    for r in recs:
        scenes[r["scene_id"]].add(r["split"])
    straddle = sorted(s for s, v in scenes.items() if len(v) > 1)
    report["by_split_title"] = {f"{s}/{t}": n for (s, t), n in sorted(by.items())}
    report["scenes"] = len(scenes)
    report["scenes_straddling_splits"] = straddle
    for (s, t), n in sorted(by.items()):
        print(f"   {s:5s} {t:20s} {n}")
    print(f"4 scenes {len(scenes)}, straddling splits: {straddle or 'none'}")
    ok &= not straddle

    # 2 + 3 on a sample
    rng = random.Random(args.seed)
    by_shot = defaultdict(list)
    for r in recs:
        by_shot[r["scene_id"]].append(r)
    for v in by_shot.values():
        v.sort(key=lambda r: r["frame_index"])
    sample = rng.sample(recs, min(args.sample, len(recs)))
    peak_err, corr0, corr1, geom_bad = [], [], [], []
    for r in sample:
        sdr = read_u16(path_of(r["sdr_path"]))
        hdr_code = read_u16(path_of(r["hdr_path"]))
        if sdr.shape != hdr_code.shape:
            geom_bad.append(r["asset_id"])
            continue
        nits = linear_to_nits(decode_hdr_u16(hdr_code, storage), storage)
        peak = float(nits.max())
        peak_err.append(abs(peak - r["peak_nits"]) / max(r["peak_nits"], 1e-6))
        ls = log_luma(sdr_linear(sdr))
        lh = log_luma(nits)
        corr0.append(float(np.corrcoef(ls.ravel(), lh.ravel())[0, 1]))
        shot = by_shot[r["scene_id"]]
        i = next(k for k, q in enumerate(shot) if q["asset_id"] == r["asset_id"])
        for j in (i - 1, i + 1):
            if 0 <= j < len(shot) and abs(shot[j]["frame_index"] - r["frame_index"]) <= 2:
                other = read_u16(path_of(shot[j]["hdr_path"]))
                lo = log_luma(linear_to_nits(decode_hdr_u16(other, storage), storage))
                corr1.append(float(np.corrcoef(ls.ravel(), lo.ravel())[0, 1]))
                break
    report["sample"] = len(sample)
    report["geometry_mismatch"] = geom_bad
    report["peak_rel_err_max"] = max(peak_err) if peak_err else None
    report["luma_corr_offset0_mean"] = float(np.mean(corr0)) if corr0 else None
    report["luma_corr_neighbour_mean"] = float(np.mean(corr1)) if corr1 else None
    print(f"2 sample {len(sample)}: geometry mismatches {len(geom_bad)}, peak reproduced within "
          f"{100 * (max(peak_err) if peak_err else 0):.3f} %")
    print(f"3 alignment: log-luma correlation at offset 0 {np.mean(corr0):.3f}"
          + (f", at the neighbouring frame {np.mean(corr1):.3f}" if corr1 else " (no neighbours in the sample)"))
    ok &= not geom_bad and (not peak_err or max(peak_err) < 1e-3) and np.mean(corr0) > 0.9
    if corr1:
        ok &= np.mean(corr0) > np.mean(corr1)
    report["pass"] = bool(ok)
    out = args.report or args.manifest.with_name("_check.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(("PASS" if ok else "FAIL") + f"  -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
