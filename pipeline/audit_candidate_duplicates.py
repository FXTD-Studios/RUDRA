"""CPU perceptual screening across candidate splits; never certifies uniqueness.

Reads SDR only. Flags similar 32x32 structures using a 64-bit DCT hash and
normalized correlation. Cropped/reframed matches may be missed. No deletion.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.commercial_preflight import sha256


def fingerprint(image):
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("expected finite RGB/BGR image")
    scale = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
    grey = cv2.cvtColor(image.astype(np.float32) / scale, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(thumb)[:8, :8].reshape(-1)
    bits = dct > np.median(dct[1:]); bits[0] = False
    code = int.from_bytes(np.packbits(bits).tobytes(), "big")
    return {"phash": f"{code:016x}", "std": float(thumb.std()),
            "aspect": image.shape[1] / image.shape[0],
            "thumb": base64.b64encode(np.round(thumb.clip(0, 1) * 255).astype(np.uint8).tobytes()).decode()}


def correlation(a, b):
    x = np.frombuffer(base64.b64decode(a["thumb"]), np.uint8).astype(float)
    y = np.frombuffer(base64.b64decode(b["thumb"]), np.uint8).astype(float)
    x -= x.mean(); y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom else 0.0


def screen(records, max_hamming=6, min_correlation=0.98):
    """Predeclared conservative screening thresholds; flags require review."""
    matches, low_information = [], []
    eligible = []
    for r in records:
        if r["std"] < 0.01:
            low_information.append(r["asset_id"])
        else:
            eligible.append(r)
    codes = [int(r["phash"], 16) for r in eligible]
    for i, a in enumerate(eligible):
        for j in range(i + 1, len(eligible)):
            b = eligible[j]
            if a["split"] == b["split"] or abs(a["aspect"] / b["aspect"] - 1) > 0.03:
                continue
            distance = (codes[i] ^ codes[j]).bit_count()
            if distance <= max_hamming:
                similarity = correlation(a, b)
                if similarity >= min_correlation:
                    matches.append({"a": a["asset_id"], "b": b["asset_id"], "hamming": distance,
                                    "correlation": similarity, "splits": [a["split"], b["split"]]})
    return matches, low_information


def audit(manifest, out):
    manifest, out = Path(manifest).resolve(), Path(out).resolve()
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if out.is_relative_to(manifest.parent) or any(out.is_relative_to(Path(r["sdr_path"]).resolve().parent) for r in rows):
        raise ValueError("output must be outside input directories")
    if out.exists():
        raise ValueError("use a new output directory")
    out.mkdir(parents=True)
    cv2.setNumThreads(2)
    started = time.monotonic(); records = []; errors = []
    with (out / "fingerprints.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            try:
                image = cv2.imread(row["sdr_path"], cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError("decode failed")
                rec = {"asset_id": row["asset_id"], "split": row["split"], "scene_id": row["scene_id"],
                       **fingerprint(image)}
                records.append(rec); f.write(json.dumps(rec) + "\n"); f.flush()
            except (ValueError, cv2.error) as e:
                errors.append({"asset_id": row["asset_id"], "error": str(e)})
            if (i + 1) % 200 == 0:
                print(f"fingerprinted {i + 1}/{len(rows)} SDR frames", flush=True)
    matches, low = screen(records)
    summary = {"input_manifest": str(manifest), "manifest_sha256": sha256(manifest),
               "algorithm_sha256": sha256(Path(__file__)), "records": len(rows), "decoded": len(records),
               "errors": errors, "cross_split_flags": matches, "low_information": low,
               "thresholds": {"phash_hamming_max": 6, "correlation_min": 0.98, "std_min": 0.01, "aspect_relative_max": 0.03},
               "elapsed_seconds": time.monotonic() - started,
               "limitations": ["SDR structural screening only, not proof of uniqueness", "crops/reframing and HDR-only duplicates may be missed", "low-information frames require separate review"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("cross_split_flags", "low_information")}, indent=2))
    print(f"flags={len(matches)}, low-information={len(low)}")
    return 1 if errors or matches else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); raise SystemExit(audit(a.manifest, a.out))
