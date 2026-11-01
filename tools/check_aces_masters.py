"""Check `rudra-native aces-v2` masters against the source and against each other.

Two checks that need only the published model package, not the training
checkpoint, so CI can run them on what users actually download:

  R  round trip: the master's ACES 2.0 SDR 100-nit Rec.709 view (OCIO, the
     model's own view) gives back the source's display codes. The package's
     appearance constraint promises max 1 code, fully clipped pixels included.
  B  backends: two masters of the same frames (LibTorch vs ONNX Runtime, CPU
     vs Core ML, ...) agree in that view within the manifest's
     onnx_tolerance.limit_display_codes.

    python tools/check_aces_masters.py --source frames/ --master out_libtorch/ \\
        [--other out_onnxruntime/] [--package models/RUDRA_ACES_2] --report r.json

Sources: 8-bit stills (png, tif, jpg) read as display RGB, matched to
<master>/<stem>.exr. Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STILLS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def read_source(path: Path, round_codes: bool = False) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read {path}")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3][..., ::-1]
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    codes = img.astype(np.float64) / scale * 255.0       # display codes, 8-bit units
    if round_codes:   # an 8-bit video: the engine rounds its 16-bit decode to whole codes (aces_v2_inputs.cpp)
        codes = np.rint(codes)
    return codes.astype(np.float32)


def read_master(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    import OpenEXR
    with OpenEXR.File(str(path)) as f:
        ch = f.channels()
        if "RGB" in ch:
            rgb = np.asarray(ch["RGB"].pixels, np.float32)
        else:
            rgb = np.stack([np.asarray(ch[c].pixels, np.float32) for c in ("R", "G", "B")], axis=-1)
        mask = ch.get("rudra.reconstructed")
        return rgb, None if mask is None else np.asarray(mask.pixels, np.float32)


def to_codes(view, master: np.ndarray) -> np.ndarray:
    return np.asarray(view.forward(master), np.float32) * 255.0


def stats(err: np.ndarray, limit: float) -> dict:
    e = err.max(axis=-1) if err.ndim == 3 else err
    return {"max": float(e.max()), "rms": float(np.sqrt(np.mean(e.astype(np.float64) ** 2))),
            "p999": float(np.quantile(e, 0.999)), "within_limit": float(np.mean(e <= limit)), "pixels": int(e.size)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--other", type=Path, help="a second master folder of the same frames")
    ap.add_argument("--package", type=Path, help="model package: reads onnx_tolerance from its manifest")
    ap.add_argument("--r-limit", type=float, default=1.0, help="round-trip limit, display codes (contract: 1)")
    ap.add_argument("--r-fraction", type=float, default=1.0, help="share of pixels that must be within --r-limit")
    ap.add_argument("--b-limit", type=float, help="backend limit, display codes (default: the manifest's)")
    ap.add_argument("--slack", type=float, default=0.02, help="float slack added to both limits (OCIO build differences)")
    ap.add_argument("--round-codes", action="store_true",
                    help="round 16-bit source frames to whole 8-bit codes, as the engine does for an 8-bit video")
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args(argv)

    b_limit = a.b_limit
    if b_limit is None and a.package:
        m = json.loads((a.package / "manifest.json").read_text(encoding="utf-8"))
        b_limit = float(m.get("onnx_tolerance", {}).get("limit_display_codes", 1.0))
    if b_limit is None:
        b_limit = 1.0

    from rudra.aces_view import ACES2View
    view = ACES2View(threads=4)
    sources = sorted(p for p in (a.source.iterdir() if a.source.is_dir() else [a.source]) if p.suffix.lower() in STILLS)
    if not sources:
        print(f"FAILED: no stills in {a.source}")
        return 1

    rows, ok = [], True
    for src in sources:
        row = {"frame": src.name}
        exr = a.master / f"{src.stem}.exr"
        if not exr.is_file():
            row["error"] = f"missing {exr}"
            rows.append(row); ok = False
            continue
        codes = read_source(src, a.round_codes)
        master, mask = read_master(exr)
        if master.shape[:2] != codes.shape[:2]:
            row["error"] = f"size {master.shape[:2]} vs source {codes.shape[:2]}"
            rows.append(row); ok = False
            continue
        if not np.isfinite(master).all():
            row["error"] = "non-finite values in the master"
            rows.append(row); ok = False
            continue
        view_a = to_codes(view, master)
        r = stats(np.abs(view_a - codes), a.r_limit + a.slack)
        r["pass"] = r["within_limit"] >= a.r_fraction
        if mask is not None:
            r["reconstructed_share"] = float(np.mean(mask > 0))
        row["R_round_trip"] = r
        ok &= r["pass"]
        if a.other:
            exr_b = a.other / f"{src.stem}.exr"
            if not exr_b.is_file():
                row["error"] = f"missing {exr_b}"
                ok = False
            else:
                master_b, _ = read_master(exr_b)
                b = stats(np.abs(to_codes(view, master_b) - view_a), b_limit + a.slack)
                b["pass"] = b["max"] <= b_limit + a.slack
                lin = np.abs(master_b - master) / np.maximum(np.abs(master), 1e-3)
                b["scene_linear_rel_max"] = float(lin.max())
                row["B_backends"] = b
                ok &= b["pass"]
        rows.append(row)

    report = {"schema": "rudra_aces_master_check_v1", "view": view.identity, "source": str(a.source),
              "master": str(a.master), "other": str(a.other) if a.other else None,
              "r_limit_codes": a.r_limit, "r_fraction": a.r_fraction, "b_limit_codes": b_limit,
              "slack": a.slack, "frames": rows, "pass": bool(ok)}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        r, b = row.get("R_round_trip"), row.get("B_backends")
        line = f"{row['frame']}: "
        line += row.get("error", "") or ""
        if r:
            line += f"R max {r['max']:.3f} codes ({'ok' if r['pass'] else 'FAIL'})"
        if b:
            line += f"; B max {b['max']:.3f} codes ({'ok' if b['pass'] else 'FAIL'})"
        print(line)
    print("PASS" if ok else "FAILED", f"-> {a.report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
