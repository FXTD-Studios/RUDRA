"""Motion-compensated temporal stability of an HDR reconstruction vs its reference.

Per consecutive frame pair (t-1, t) inside one shot:

    err(X) = mean_mask | PQ(X_t) - warp(PQ(X_{t-1}), flow_{t->t-1}) |

computed for the model output and for the reference with the SAME flow and
mask. ``excess = err(output) - err(reference)``: the reference absorbs genuine
lighting change and flow error, so a positive excess means the output changes
more along motion trajectories than the ground truth does (flicker, popping
highlights). It is reported over all valid pixels and over reference
highlights (> ``highlight_nits``).

Safeguards
* Pairs across a cut are never scored as flicker; cuts are supplied or detected
  from guide histograms and reported separately.
* The mask keeps only forward/backward-consistent flow in textured regions.
  DIS drifts in flat areas (2.61 px on open sky, see training/run_raft_gate.ps1),
  so flat pixels are excluded rather than trusted. Use RAFT flow (``--flow``
  arrays) for final evidence when the GPU is free.
* Pairs with too few valid pixels are skipped and counted, never filled.
* PQ (SMPTE ST 2084) luminance is the error domain; this is not CVVDP and not
  a perceptual video-quality pass. No thresholds are applied here: acceptance
  limits must be fixed before final test evaluation.

    python training/temporal_stability.py --output out_dir --reference ref_dir \
        --nits-scale 203 --report temporal.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402

BT2100_LUMA = (0.2627, 0.6780, 0.0593)
_M1, _M2 = 2610 / 16384, 2523 / 4096 * 128
_C1, _C2, _C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def pq_encode(nits: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(nits, np.float64) / 10000.0, 0.0, 1.0)
    p = np.power(y, _M1)
    return np.power((_C1 + _C2 * p) / (1.0 + _C3 * p), _M2).astype(np.float32)


def luminance(rgb: np.ndarray, weights=BT2100_LUMA) -> np.ndarray:
    rgb = np.asarray(rgb, np.float32)
    if rgb.ndim == 2:
        return rgb
    return rgb[..., 0] * weights[0] + rgb[..., 1] * weights[1] + rgb[..., 2] * weights[2]


def _check_frames(frames, name):
    if len(frames) < 2:
        raise ValueError(f"{name}: need at least two frames")
    shape = frames[0].shape
    for i, f in enumerate(frames):
        if f.shape != shape:
            raise ValueError(f"{name}: frame {i} shape {f.shape} != {shape}")
        if not np.isfinite(f).all():
            raise ValueError(f"{name}: frame {i} has NaN/Inf")
        if (f < 0).any():
            raise ValueError(f"{name}: frame {i} has negative luminance")


def _to_u8(pq: np.ndarray) -> np.ndarray:
    return np.clip(pq * 255.0 + 0.5, 0, 255).astype(np.uint8)


def dis_flow(from_pq: np.ndarray, to_pq: np.ndarray) -> np.ndarray:
    """Flow F such that from[y, x] ~ to[y + F_y, x + F_x]."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(_to_u8(from_pq), _to_u8(to_pq), None)


def warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img.astype(np.float32), gx + flow[..., 0], gy + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def valid_mask(guide_cur: np.ndarray, fwd: np.ndarray, bwd: np.ndarray,
               fb_max_px: float = 0.5, grad_min: float = 0.01) -> np.ndarray:
    """fwd: cur->prev, bwd: prev->cur. Consistent, in-bounds, textured pixels."""
    back = np.dstack([warp(bwd[..., 0], fwd), warp(bwd[..., 1], fwd)])
    fb = np.linalg.norm(fwd + back, axis=2)
    gx = cv2.Sobel(guide_cur.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(guide_cur.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8.0
    textured = np.hypot(gx, gy) >= grad_min
    return np.isfinite(fb) & (fb <= fb_max_px) & textured


def detect_cuts(guide_pq: list[np.ndarray], threshold: float = 0.5, bins: int = 32) -> list[int]:
    """Indices t where a cut lies between t-1 and t (histogram L1 distance)."""
    hists = []
    for g in guide_pq:
        h, _ = np.histogram(g, bins=bins, range=(0.0, 1.0))
        hists.append(h / max(h.sum(), 1))
    return [t for t in range(1, len(hists)) if np.abs(hists[t] - hists[t - 1]).sum() > threshold]


def _masked_err(cur, prev_warped, mask):
    m = mask & np.isfinite(prev_warped)
    if not m.any():
        return float("nan"), 0
    return float(np.abs(cur[m] - prev_warped[m]).mean()), int(m.sum())


def evaluate_sequence(outputs, references, guide=None, cuts=None, flows=None,
                      highlight_nits: float = 203.0, min_valid_fraction: float = 0.02,
                      fb_max_px: float = 0.5, grad_min: float = 0.01) -> dict:
    """outputs/references: lists of luminance (H,W) or RGB (H,W,3) arrays in nits.

    guide: optional list of arrays in nits used only for flow/cuts (default
    reference); for an SDR input pass display-linear nits (e.g. BT.1886 x 100).
    flows: optional list of (fwd, bwd) per pair index t (cur->prev, prev->cur),
    e.g. RAFT, replacing DIS. cuts: list of t, or None to detect from guide.
    """
    out_y = [luminance(f) for f in outputs]
    ref_y = [luminance(f) for f in references]
    _check_frames(out_y, "output")
    _check_frames(ref_y, "reference")
    if len(out_y) != len(ref_y) or out_y[0].shape != ref_y[0].shape:
        raise ValueError("output/reference sequences differ in length or shape")
    out_pq = [pq_encode(y) for y in out_y]
    ref_pq = [pq_encode(y) for y in ref_y]
    if guide is None:
        guide_pq = ref_pq
    else:
        guide_y = [luminance(g) for g in guide]
        _check_frames(guide_y, "guide")
        if len(guide_y) != len(ref_y) or guide_y[0].shape != ref_y[0].shape:
            raise ValueError("guide sequence differs in length or shape")
        guide_pq = [pq_encode(y) for y in guide_y]
    cut_set = set(detect_cuts(guide_pq) if cuts is None else cuts)
    if any(not 1 <= t < len(ref_pq) for t in cut_set):
        raise ValueError("cut index out of range")
    rows, n = [], ref_pq[0].size
    for t in range(1, len(ref_pq)):
        row = {"t": t, "cut": t in cut_set}
        if row["cut"]:
            rows.append(row)
            continue
        if flows is not None:
            fwd, bwd = flows[t]
        else:
            fwd = dis_flow(guide_pq[t], guide_pq[t - 1])
            bwd = dis_flow(guide_pq[t - 1], guide_pq[t])
        mask = valid_mask(guide_pq[t], fwd, bwd, fb_max_px, grad_min)
        row["valid_fraction"] = float(mask.mean())
        if row["valid_fraction"] < min_valid_fraction:
            row["skipped"] = "too_few_valid_pixels"
            rows.append(row)
            continue
        e_ref, k = _masked_err(ref_pq[t], warp(ref_pq[t - 1], fwd), mask)
        e_out, _ = _masked_err(out_pq[t], warp(out_pq[t - 1], fwd), mask)
        row.update(ref_err=e_ref, out_err=e_out, excess=e_out - e_ref, pixels=k)
        hl = mask & (ref_y[t] > highlight_nits)
        if hl.sum() >= max(16, 1e-4 * n):
            h_ref, hk = _masked_err(ref_pq[t], warp(ref_pq[t - 1], fwd), hl)
            h_out, _ = _masked_err(out_pq[t], warp(out_pq[t - 1], fwd), hl)
            row.update(highlight_ref_err=h_ref, highlight_out_err=h_out,
                       highlight_excess=h_out - h_ref, highlight_pixels=hk)
        rows.append(row)
    scored = [r for r in rows if "excess" in r]
    summary = {"frames": len(ref_pq), "pairs": len(rows), "cuts": sorted(cut_set),
               "pairs_scored": len(scored),
               "pairs_skipped": sum(1 for r in rows if r.get("skipped")),
               "flow": "supplied" if flows is not None else "DIS medium (flat regions masked)",
               "error_domain": "PQ luminance", "thresholds_applied": False}
    for key in ("excess", "highlight_excess"):
        v = np.array([r[key] for r in scored if key in r])
        summary[key] = ({"pairs": int(v.size), "mean": float(v.mean()), "p95": float(np.quantile(v, .95)),
                         "max": float(v.max())} if v.size else {"pairs": 0})
    return {"summary": summary, "pairs": rows}


def reset_check(isolated, in_sequence, atol_pq: float = 1e-3) -> dict:
    """Frame after a cut processed alone vs inside the sequence (stateful paths)."""
    a, b = pq_encode(luminance(isolated)), pq_encode(luminance(in_sequence))
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    d = float(np.abs(a - b).max())
    return {"max_abs_pq_diff": d, "atol_pq": atol_pq, "reset_ok": d <= atol_pq}


def _finite_or_none(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite_or_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite_or_none(v) for v in x]
    return x


def _load_dir(path: Path, nits_scale: float) -> list[np.ndarray]:
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".exr", ".npy"))
    if not files:
        raise ValueError(f"no .exr/.npy frames in {path}")
    frames = []
    for p in files:
        a = np.load(p) if p.suffix.lower() == ".npy" else cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError(f"cannot read {p}")
        if a.ndim == 3 and p.suffix.lower() == ".exr":
            a = a[..., 2::-1]
        frames.append(np.asarray(a, np.float32) * nits_scale)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True, help="dir of model frames (sorted names)")
    ap.add_argument("--reference", type=Path, required=True, help="dir of reference frames")
    ap.add_argument("--guide", type=Path, default=None, help="optional dir for flow/cuts")
    ap.add_argument("--nits-scale", type=float, default=203.0, help="nits per stored 1.0")
    ap.add_argument("--cuts", type=str, default=None, help="comma list of t, or omit to detect")
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    if a.report.exists():
        raise SystemExit("report exists; use a new path")
    out, ref = _load_dir(a.output, a.nits_scale), _load_dir(a.reference, a.nits_scale)
    guide = _load_dir(a.guide, a.nits_scale) if a.guide else None
    cuts = [int(x) for x in a.cuts.split(",") if x] if a.cuts is not None else None
    res = evaluate_sequence(out, ref, guide=guide, cuts=cuts)
    res["inputs"] = {"output": str(a.output), "reference": str(a.reference), "nits_scale": a.nits_scale}
    a.report.write_text(json.dumps(_finite_or_none(res), indent=2, allow_nan=False))
    print(json.dumps(res["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
