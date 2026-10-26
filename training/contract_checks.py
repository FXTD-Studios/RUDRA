#!/usr/bin/env python3
"""Product-contract checks for the "virtual RAW" master (22 Oct 2026).

docs/production/PRODUCT_TARGET.md defines "works correctly". This module measures
it, for the analytic baseline and any checkpoint, on run05-style pairs
(pipeline/run05_corpus.py: input SDR, target, reconstruction mask):

  R  round trip   forward(declared view, output) vs the input SDR, in 8-bit
                  SOURCE code values, on pixels the mask says were not
                  reconstructed. Pass: >= 99.9 % within 1 code.
                  (Needs no ground truth: also usable on generated clips.)
  G  headroom     push the master -3..+3 EV, view it, and measure banding in
                  gentle gradients (share of "flat step" pixels) against the
                  truth's own value; and new clipping at the model ceiling.
  H  highlights   inside the reconstruction mask: |log2 Y_out / Y_target|,
                  against the baseline's.
  S  shadows      inside the crushed mask: share of pixels lifted > 1 stop
                  above the truth AND above 0.5 nit.
  F  fidelity     log-L1 to the target on the whole frame (dequantisation
                  shows up here; the baseline cannot dequantise).

    python training/contract_checks.py --manifest <run05 eval>/manifest.jsonl \
        --checkpoint run05=G:/.../best.pt [--checkpoint run03=...] --out <new dir> [--limit 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LUMA2020 = np.array([0.2627, 0.6780, 0.0593], np.float32)
PUSHES = (-3, -2, -1, 0, 1, 2, 3)
R_TOLERANCE_CODES, R_PASS_FRACTION = 1.0, 0.999
VISIBLE_NITS = 0.5


# --------------------------------------------------------------- view maths
@lru_cache(maxsize=8)
def _view_table(view: str, n: int = 8193):
    from pipeline.run05_corpus import declared_inverse
    codes = np.linspace(0.0, 1.0, n, dtype=np.float32)
    vals = declared_inverse(np.repeat(codes[None, :, None], 3, axis=2), view)[0, :, 0].astype(np.float64)
    vals = np.maximum.accumulate(vals) + np.arange(n) * 1e-15          # strictly increasing for interp
    return codes.astype(np.float64), vals


def view_forward(net: np.ndarray, view: str = "bt1886") -> np.ndarray:
    """Network units -> canonical BT.2020 sRGB code under the declared view (clips at 1)."""
    codes, vals = _view_table(view)
    return np.interp(np.asarray(net, np.float64), vals, codes).astype(np.float32)


def canonical_to_source(canon: np.ndarray) -> np.ndarray:
    """Inverse of pipeline/run05_corpus.canonical: canonical code -> Rec.709 source code."""
    from rudra.delivery.colorspace import rgb_to_rgb_matrix

    def to_lin(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def to_code(l):
        l = np.clip(l, 0.0, 1.0)
        return np.where(l <= 0.0031308, l * 12.92, 1.055 * np.power(l, 1 / 2.4) - 0.055)
    m = rgb_to_rgb_matrix("rec2020", "rec709").astype(np.float64)
    return to_code(to_lin(np.asarray(canon, np.float64)) @ m.T).astype(np.float32)


def reconstruction_mask(canon_in: np.ndarray, policy='legacy_nearwhite') -> np.ndarray:
    """From the INPUT alone, the same bands run05_corpus uses: where range was lost."""
    from pipeline.run05_corpus import _smooth
    from rudra.reconstruction_policy import reconstruction_bands
    hi_band, lo_band = reconstruction_bands(policy)
    cmax = canonical_to_source(canon_in).max(axis=2)
    return np.maximum(_smooth(cmax, *hi_band), 1.0 - _smooth(cmax, *lo_band)).astype(np.float32)


def enforce_contract(output: np.ndarray, canon_in: np.ndarray, view: str = "bt1886", policy='legacy_nearwhite') -> np.ndarray:
    """Guarantee R for ANY model: outside the input's reconstruction mask, keep the output inside
    the 8-bit quantisation interval of the source code (dequantisation allowed, regrading not)."""
    from pipeline.run05_corpus import canonical, declared_inverse
    src_out = canonical_to_source(view_forward(output, view))
    q = np.rint(canonical_to_source(canon_in) * 255.0)
    src_cl = np.clip(src_out, (q - 0.49) / 255.0, (q + 0.49) / 255.0)
    clamped = declared_inverse(canonical(np.clip(src_cl, 0, 1).astype(np.float32)), view)
    m = np.clip((reconstruction_mask(canon_in, policy) - 0.5) * 2.0, 0.0, 1.0)[..., None]   # 0 wherever R applies
    return (clamped * (1 - m) + output * m).astype(np.float32)


# --------------------------------------------------------------- checks
def roundtrip(output: np.ndarray, canon_in: np.ndarray, mask: np.ndarray | None = None,
              view: str = "bt1886", policy='legacy_nearwhite') -> dict:
    src_out = canonical_to_source(view_forward(output, view))
    src_in = canonical_to_source(canon_in)
    err = np.abs(src_out - src_in).max(axis=2) * 255.0
    # The contract's mask is decided from the INPUT alone (also on generated clips).
    keep = (reconstruction_mask(canon_in, policy) if mask is None else mask) < 0.5
    e = err[keep]
    if e.size == 0:
        return {"r_pixels": 0}
    frac = float((e <= R_TOLERANCE_CODES).mean())
    return {"r_pixels": int(e.size), "r_within_1_code": frac, "r_p999_codes": float(np.percentile(e, 99.9)),
            "r_max_codes": float(e.max()), "r_pass": frac >= R_PASS_FRACTION}


def banding_index(output: np.ndarray, push: float, view: str = "bt1886") -> dict:
    """Share of gentle-gradient pixels that sit on a flat step after an EV push (10-bit view)."""
    import cv2
    disp = view_forward(output * np.float32(2.0 ** push), view) @ LUMA2020 * 1023.0
    smooth = cv2.GaussianBlur(disp.astype(np.float32), (0, 0), 4.0)
    sy, sx = np.gradient(smooth)
    slope = np.hypot(sx, sy)
    gx = np.abs(np.diff(disp, axis=1, append=disp[:, -1:]))
    gy = np.abs(np.diff(disp, axis=0, append=disp[-1:, :]))
    g = np.maximum(gx, gy)
    gentle = (slope > 0.03) & (slope < 1.5) & (disp > 2) & (disp < 1020)
    if gentle.sum() < 500:
        return {"gentle_pixels": int(gentle.sum()), "banding": None}
    return {"gentle_pixels": int(gentle.sum()), "banding": float((g[gentle] < 0.01).mean())}


def headroom(output: np.ndarray, target: np.ndarray, view: str = "bt1886", max_hdr: float = 4.0) -> dict:
    out = {"clip_at_ceiling": float((output.max(axis=2) >= 0.999 * max_hdr).mean()),
           "nonfinite_or_negative": float((~np.isfinite(output) | (output < 0)).mean())}
    excesses = []
    for p in PUSHES:
        bo, bt = banding_index(output, p, view)["banding"], banding_index(target, p, view)["banding"]
        out[f"banding_{p:+d}"] = bo
        out[f"banding_truth_{p:+d}"] = bt
        if bo is not None and bt is not None:
            excesses.append(bo - bt)
    out["banding_measured_pushes"] = len(excesses)
    out["banding_excess_worst"] = max(0.0, max(excesses)) if excesses else None
    return out


def highlights_shadows(output: np.ndarray, target: np.ndarray, mask: np.ndarray, canon_in: np.ndarray) -> dict:
    yo, yt = output @ LUMA2020, target @ LUMA2020
    res = {}
    hi = (mask >= 0.5) & (canon_in.max(axis=2) > 0.5)
    lo = (mask >= 0.5) & (canon_in.max(axis=2) <= 0.5)
    if hi.sum() >= 50:
        r = np.log2(np.maximum(yo[hi], 1e-9) / np.maximum(yt[hi], 1e-9))
        res.update(h_pixels=int(hi.sum()), h_abs_log2=float(np.abs(r).mean()), h_median_log2=float(np.median(r)))
    if lo.sum() >= 50:
        lifted = (yo[lo] > 2.0 * yt[lo]) & (yo[lo] * 1e4 > VISIBLE_NITS)
        res.update(s_pixels=int(lo.sum()), s_visible_lift=float(lifted.mean()))
    a, b = np.log2(np.maximum(yo, 1e-7)), np.log2(np.maximum(yt, 1e-7))
    res["f_log2_l1"] = float(np.abs(a - b).mean())
    return res


def evaluate_pair(outputs: dict, canon_in, target, mask, view="bt1886", policy='legacy_nearwhite') -> dict:
    return {label: {**roundtrip(o, canon_in, None, view, policy), **headroom(o, target, view),
                    **highlights_shadows(o, target, mask, canon_in)} for label, o in outputs.items()}


# --------------------------------------------------------------- runner
class Runner:
    def __init__(self, checkpoints: dict[str, Path], device: str | None, view: str, enforce: bool = False, policy='legacy_nearwhite'):
        self.enforce = enforce
        self.policy = policy
        import torch
        from training.export_bench_pairs import load_model
        self.torch, self.view = torch, view
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = {k: load_model(p, self.device) for k, p in checkpoints.items()}

    def run(self, canon: np.ndarray) -> dict:
        from pipeline.run05_corpus import declared_inverse
        from rudra.source_baseline import baseline_curve_index
        from training.infer_sdr2hdr import predict_image
        torch = self.torch
        out = {"baseline_" + self.view: declared_inverse(canon, self.view)}
        x = torch.from_numpy(np.ascontiguousarray(canon)).permute(2, 0, 1)[None].to(self.device)
        with torch.inference_mode():
            for label, model in self.models.items():
                curve = baseline_curve_index(self.view) if getattr(model, "baseline_source", False) else None
                y = predict_image(model, x, preserve_outside=True, tile_size=512, overlap=64,
                                  bf16=False, baseline_curve=curve)
                out[label] = np.maximum(y[0].permute(1, 2, 0).float().cpu().numpy(), 0)
                if self.enforce:
                    out[label + "+contract"] = enforce_contract(out[label], canon, self.view, self.policy)
        return out


def _summ(vals):
    vals = [v for v in vals if v is not None]
    return None if not vals else float(np.mean(vals))


def load_reference(path):
    """Keep the reference range independent of the model's output ceiling."""
    from training.sdr2hdr_dataset import load_rgb
    return load_rgb(path, hdr=True, ceiling=1e6)


def run(manifest: Path, checkpoints: dict[str, Path], out: Path, device=None, limit=None,
        enforce: bool = False, max_hdr: float = 4.0) -> dict:
    import cv2
    from training.sdr2hdr_dataset import load_rgb
    if out.exists():
        raise ValueError("use a new output directory")
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:limit] if limit else rows
    view = rows[0].get("declared_view", "bt1886")
    policy = rows[0].get('reconstruction_policy', 'legacy_nearwhite')
    if any(r.get('declared_view', 'bt1886') != view or r.get('reconstruction_policy', 'legacy_nearwhite') != policy for r in rows):
        raise ValueError('evaluation requires a single declared view and reconstruction policy')
    runner = Runner(checkpoints, device, view, enforce, policy)
    per = []
    for i, r in enumerate(rows, 1):
        canon = load_rgb(r["sdr_path"], hdr=False)
        target = load_reference(r["hdr_path"])
        mask = cv2.imread(r["mask_path"], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        res = evaluate_pair(runner.run(canon), canon, target, mask, view, policy)
        per.append({"asset_id": r["asset_id"], "scene_id": r["scene_id"], "family": r.get("sdr_curve"),
                    "codec": r.get("sdr_codec"), "methods": res})
        if i % 20 == 0:
            print(f"[contract] {i}/{len(rows)}", flush=True)
    labels = list(per[0]["methods"]) if per else []
    keys = ("r_within_1_code", "banding_excess_worst", "banding_+2", "clip_at_ceiling", "h_abs_log2",
            "s_visible_lift", "f_log2_l1")
    summary = {}
    for lab in labels:
        ms = [p["methods"][lab] for p in per]
        summary[lab] = {k: _summ([m.get(k) for m in ms]) for k in keys}
        summary[lab]["metric_pair_counts"] = {k: sum(m.get(k) is not None for m in ms) for k in keys}
        summary[lab]["r_pass_rate"] = _summ([float(m["r_pass"]) for m in ms if "r_pass" in m])
        summary[lab]["by_family_f_log2_l1"] = {
            f: _summ([p["methods"][lab]["f_log2_l1"] for p in per if p["family"] == f])
            for f in sorted({p["family"] for p in per})}
    report = {"manifest": str(manifest), "view": view, "reconstruction_policy": policy, "checkpoints": {k: str(v) for k, v in checkpoints.items()},
              "pairs": len(per), "summary": summary, "per_pair": per,
              "contract": {"R": f">= {R_PASS_FRACTION:.1%} of unmasked pixels within {R_TOLERANCE_CODES} code",
                           "G": "banding_excess_worst <= 0.05 and clip_at_ceiling ~ 0",
                           "H": "h_abs_log2 below the baseline's", "S": "s_visible_lift ~ 0",
                           "F": "f_log2_l1 below the baseline's"},
              "release_authorized": False}
    out.mkdir(parents=True)
    (out / "contract_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"pairs": len(per), "summary": summary, "out": str(out)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--checkpoint", action="append", default=[], help="label=path")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--enforce", action="store_true", help="also score each model with enforce_contract")
    a = ap.parse_args()
    cps = dict((s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in a.checkpoint)
    print(json.dumps(run(a.manifest, cps, a.out, a.device, a.limit, a.enforce), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
