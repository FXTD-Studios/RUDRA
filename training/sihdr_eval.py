"""SI-HDR independent-capture evaluation with a protocol frozen before any score.

SI-HDR (Hanji et al. 2022, CC BY 4.0) supplies 181 photographic HDR captures
(1888x1280 half-float EXR, relative scale, normalised so the maximum is 65504)
and SIMULATED 8-bit SDR inputs at two clipping levels (clip_95, clip_97) made
with random camera response curves and sensor noise. It is independent of every
RUDRA training source, but its SDR is simulated, not graded: results here are
"independent HDR captures, simulated SDR", never "real SDR".

Protocol v1.1 (RUDRA's adaptation; NOT a reproduction of the publisher's tables)
  * input: the PNG as 8-bit sRGB-coded Rec.709 -> canonical BT.2020 sRGB code,
    exactly as rudra/video.py feeds the model; model and analytic baseline use
    the app default bt1886 picker curve; fp32, 512 px tiles, overlap 64,
    preserve_outside on (the production validation-kit settings).
  * display mapping: the reference is scaled so its per-image, per-clip-level
    input saturation anchor T (independent_prep/sihdr/
    sihdr_exposure_anchor_measurements.json, measured before any model output
    existed) lands at 100 cd/m2. Each prediction (model and baseline alike)
    gets ONE scalar exposure, the median log-luminance ratio to the mapped
    reference over input-unsaturated pixels only, so highlight expansion is never
    fitted. Both are clamped to [0.005, 1000] cd/m2.
  * metric: PU21-PSNR on luminance (Rec.709 weights for the reference,
    Rec.2020 for the BT.2020 predictions; luminance avoids the provisional colour
    interpretation), with the bench's PU21 peak convention. Regions: full frame;
    input-saturated (any channel >= 250, dilated 2 px); unsaturated (outside a
    4 px dilation). A region with < 100 valid pixels is NaN for both methods.
  * exclusions: reference half-float ceiling pixels in 008/080/194 are masked
    everywhere; 195 (3.9% ceiling) is excluded from highlight metrics.
  * statistics: model-minus-baseline per image, equal-group bootstrap over the
    15 location groups (paired_gate.compare), each clip level separately.

Commands
    freeze   --prep <independent_eval_prep_20261019.json> --anchors <...json> --out protocol.json
    evaluate --protocol protocol.json --root <SI-HDR root> --checkpoint ckpt.pt --out <new dir>
    compare  --protocol protocol.json --a <dirA> --b <dirB> --out cmp.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_VERSION = "rudra-sihdr-v1.1"  # v1.1: compressed-EXR reader + partial-output handling; transforms and criteria unchanged from v1
CLIP_LEVELS = ("clip_95", "clip_97")
LUMA709 = np.array([0.2126, 0.7152, 0.0722])
LUMA2020 = np.array([0.2627, 0.6780, 0.0593])
HALF_MAX = 65504.0
DISPLAY = dict(anchor_nits=100.0, min_nits=0.005, max_nits=1000.0)
SATURATED_CODE = 250
MIN_REGION_PIXELS = 100
ACCEPTANCE = {
    "A1_full_frame_no_regression": "per clip level: lower 95% CI of model-minus-baseline full-frame PU21-PSNR >= -0.25 dB",
    "A2_highlight_expansion": "per clip level: lower 95% CI of model-minus-baseline saturated-region PU21-PSNR > 0 dB",
    "A3_unsaturated_preservation": "per clip level: lower 95% CI of model-minus-baseline unsaturated-region PU21-PSNR >= -0.25 dB",
    "A4_category_flag": "any category tag spanning >= 2 location groups with mean saturated-region gain < -0.5 dB is a flagged regression",
    "A5_visual": "the 10 lowest full-frame gains per clip level must be viewed on an HDR display before any claim; unviewed = incomplete",
    "interpretation": ("Independent captures with simulated SDR. The publisher reports ~3.5 dB PU21-PSNR "
                       "differences were needed before metrics reliably matched preference: a pass is a "
                       "no-regression / direction finding, not a perceptual or release pass."),
}


def code_sha256(path: Path) -> str:
    """Line-ending independent: a CRLF re-checkout on Windows is the same code."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ protocol
def freeze(prep_path: Path, anchors_path: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError(f"{out} exists; a frozen protocol is never rewritten")
    prep = json.loads(prep_path.read_text(encoding="utf-8"))["sihdr"]
    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    groups = prep["scene_groups_for_bootstrap"]
    ids = sorted(i for members in groups.values() for i in members)
    if len(ids) != len(set(ids)):
        raise ValueError("an image belongs to two groups")
    missing = [i for i in ids if i not in anchors or any(c not in anchors[i] for c in CLIP_LEVELS)]
    if missing:
        raise ValueError(f"anchors missing for {missing[:5]}")
    protocol = {
        "version": PROTOCOL_VERSION, "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_before_model_scores": True, "images": ids, "groups": groups,
        "categories": prep["category_tags_from_contact_sheet_review"],
        "category_note": "AI contact-sheet tags, provisional; not human-labelled ground truth",
        "anchors": {i: {c: float(anchors[i][c]["ref_maxchannel_threshold"]) for c in CLIP_LEVELS} for i in ids},
        "mask_ceiling_pixels": ["008", "080", "194"], "exclude_from_highlight_metrics": ["195"],
        "display": DISPLAY, "saturated_code": SATURATED_CODE, "min_region_pixels": MIN_REGION_PIXELS,
        "inference": {"input": "8-bit PNG as sRGB-coded Rec.709 -> BT.2020 canonical sRGB code",
                      "baseline_curve": "bt1886", "precision": "fp32", "tile_size": 512, "overlap": 64,
                      "preserve_outside": True},
        "colour": "luminance metrics only; Rec.709 reference interpretation is provisional",
        "acceptance": ACCEPTANCE,
        "sources": {"prep": str(prep_path), "prep_sha256": sha256_file(prep_path),
                    "anchors": str(anchors_path), "anchors_sha256": sha256_file(anchors_path)},
        "code_sha256": {"training/sihdr_eval.py": code_sha256(Path(__file__))},
        "release_authorized": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol


def load_protocol(path: Path, require_same_code: bool = True) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("version") != PROTOCOL_VERSION:
        raise ValueError("unknown protocol version")
    if require_same_code and protocol["code_sha256"]["training/sihdr_eval.py"] != code_sha256(Path(__file__)):
        raise ValueError("training/sihdr_eval.py changed after the protocol was frozen; "
                         "freeze a new protocol version instead of scoring with altered code")
    return protocol


# ------------------------------------------------------------------- scoring
def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2
    if px <= 0:
        return mask
    kernel = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def pu21(nits: np.ndarray) -> np.ndarray:
    from rudra.delivery.bench import pu21_encode
    return pu21_encode(nits)


PU_PEAK = None


def _psnr(a_pu: np.ndarray, b_pu: np.ndarray, mask: np.ndarray) -> float:
    global PU_PEAK
    if PU_PEAK is None:
        PU_PEAK = float(pu21(np.array(10000.0)))
    if int(mask.sum()) < MIN_REGION_PIXELS:
        return float("nan")
    mse = float(np.mean((a_pu[mask] - b_pu[mask]) ** 2))
    return float("inf") if mse <= 0 else 10.0 * math.log10(PU_PEAK ** 2 / mse)


def score_image(reference709: np.ndarray, predictions2020: dict, input_u8: np.ndarray, anchor: float,
                mask_ceiling: bool = False, highlight_excluded: bool = False) -> dict:
    """Pure function: one image, one clip level, any number of named predictions.

    reference709: relative linear Rec.709 (H,W,3). predictions2020: name -> linear
    BT.2020 (H,W,3) in any consistent scale (exposure is fitted per prediction).
    """
    if not (math.isfinite(anchor) and anchor > 0):
        raise ValueError("anchor must be finite and positive")
    ref = np.asarray(reference709, dtype=np.float64)
    valid = np.isfinite(ref).all(axis=2)
    if mask_ceiling:
        valid &= ~(ref >= HALF_MAX * 0.999).any(axis=2)
    sat = _dilate((input_u8 >= SATURATED_CODE).any(axis=2), 2)
    unsat_core = ~_dilate((input_u8 >= SATURATED_CODE).any(axis=2), 4)
    ref_y = np.nan_to_num(ref @ LUMA709) * (DISPLAY["anchor_nits"] / anchor)
    ref_pu = pu21(np.clip(ref_y, DISPLAY["min_nits"], DISPLAY["max_nits"]))
    regions = {"full": valid, "saturated": valid & sat, "unsaturated": valid & unsat_core}
    out = {"pixels": {k: int(v.sum()) for k, v in regions.items()}, "methods": {}}
    fit_mask = regions["unsaturated"] & (ref_y > DISPLAY["min_nits"])
    for name, pred in predictions2020.items():
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != ref.shape or not np.isfinite(pred).all():
            raise ValueError(f"{name}: shape mismatch or nonfinite prediction")
        y = np.maximum(pred @ LUMA2020, 0.0)
        usable = fit_mask & (y > 1e-12)
        if int(usable.sum()) < MIN_REGION_PIXELS:
            raise ValueError(f"{name}: too few unsaturated pixels to fit exposure")
        scale = float(np.exp(np.median(np.log(ref_y[usable]) - np.log(y[usable]))))
        pred_pu = pu21(np.clip(y * scale, DISPLAY["min_nits"], DISPLAY["max_nits"]))
        scores = {k: _psnr(pred_pu, ref_pu, m) for k, m in regions.items()}
        if highlight_excluded:
            scores["saturated"] = float("nan")
        out["methods"][name] = {"exposure_scale": scale, "pu21_psnr_db": scores}
    return out


# ---------------------------------------------------------------- discovery
def discover(root: Path, ids: list[str]) -> dict:
    """{id: {"reference": exr, "clip_95": png, "clip_97": png}}; exactly one file each."""
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".exr", ".png")]
    found: dict = {i: {} for i in ids}
    for p in files:
        m = re.search(r"(?<!\d)(\d{3})(?!\d)", p.stem)
        if not m or m.group(1) not in found:
            continue
        parts = {x.lower() for x in p.parts}
        kind = ("reference" if p.suffix.lower() == ".exr"
                else next((c for c in CLIP_LEVELS if any(c in x for x in parts)), None))
        if kind is None:
            continue
        if kind in found[m.group(1)]:
            raise ValueError(f"ambiguous {kind} for {m.group(1)}: {found[m.group(1)][kind]} and {p}")
        found[m.group(1)][kind] = p
    incomplete = [i for i, v in found.items() if set(v) != {"reference", *CLIP_LEVELS}]
    if incomplete:
        raise ValueError(f"missing files for {len(incomplete)} images, e.g. {incomplete[:5]}")
    return found


# ---------------------------------------------------------------- inference
class Runner:
    def __init__(self, checkpoint: Path, device: str | None = None):
        import torch
        from training.export_bench_pairs import load_model
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_model(checkpoint, self.device)

    def predict(self, input_u8: np.ndarray) -> dict:
        torch = self.torch
        from rudra.delivery.colorspace import rgb_to_rgb_matrix
        from rudra.sdr2hdr import linear_to_srgb, sdr_to_baseline_hdr, srgb_to_linear
        from rudra.source_baseline import baseline_curve_index, correction_log2, knots_table
        from training.infer_sdr2hdr import predict_image
        with torch.inference_mode():
            x = torch.from_numpy(input_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
            m = torch.as_tensor(rgb_to_rgb_matrix("rec709", "rec2020"), device=self.device, dtype=x.dtype)
            x = linear_to_srgb(torch.einsum("ij,bjhw->bihw", m, srgb_to_linear(x))).clamp(0, 1)
            curve = baseline_curve_index("bt1886") if getattr(self.model, "baseline_source", False) else None
            model = predict_image(self.model, x, preserve_outside=True, tile_size=512, overlap=64,
                                  bf16=False, baseline_curve=curve)
            ids = torch.tensor([baseline_curve_index("bt1886")], device=x.device)
            base = sdr_to_baseline_hdr(x.float(), self.model.corpus_ev) * torch.exp2(
                correction_log2(x.float(), ids, knots_table(x.device)))
        to_np = lambda t: t[0].permute(1, 2, 0).float().cpu().numpy()
        return {"model": to_np(model), "baseline_bt1886": to_np(base)}


def _read_png_rgb(path: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint8 or img.ndim != 3:
        raise ValueError(f"{path}: expected 8-bit RGB PNG")
    return img[..., :3][..., ::-1].copy()


def evaluate(protocol_path: Path, root: Path, checkpoint: Path, out: Path, runner=None,
             limit: int | None = None) -> dict:
    from training.exr_any import read_rgb
    protocol = load_protocol(protocol_path)
    partial = out.with_name(out.name + ".partial")
    if out.exists() or partial.exists():
        raise ValueError(f"{out} (or {partial.name}) exists; model outputs are never overwritten")
    ids = protocol["images"][:limit] if limit else protocol["images"]
    files = discover(root, ids)
    ckpt_sha = sha256_file(checkpoint)
    read_rgb(files[ids[0]]["reference"])          # fail before any output if the reader cannot decode
    runner = runner or Runner(checkpoint)
    partial.mkdir(parents=True)
    rows = []
    with (partial / "per_image.jsonl").open("w", encoding="utf-8") as handle:
        for n, image in enumerate(ids, 1):
            ref = read_rgb(files[image]["reference"])
            for clip in CLIP_LEVELS:
                inp = _read_png_rgb(files[image][clip])
                if inp.shape != ref.shape:
                    raise ValueError(f"{image}/{clip}: geometry {inp.shape} vs reference {ref.shape}")
                result = score_image(ref, runner.predict(inp), inp, protocol["anchors"][image][clip],
                                     mask_ceiling=image in protocol["mask_ceiling_pixels"],
                                     highlight_excluded=image in protocol["exclude_from_highlight_metrics"])
                row = {"image": image, "clip": clip, **result}
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
            if n % 20 == 0 or n == len(ids):
                print(f"[sihdr] {n}/{len(ids)} images", flush=True)
    if sha256_file(checkpoint) != ckpt_sha:
        raise ValueError("checkpoint changed during evaluation")
    summary = summarize(protocol, rows, "model", "baseline_bt1886")
    summary.update(checkpoint=str(checkpoint), checkpoint_sha256=ckpt_sha,
                   protocol=str(protocol_path), protocol_sha256=sha256_file(protocol_path),
                   images=len(ids), complete=len(ids) == len(protocol["images"]))
    (partial / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    partial.rename(out)                            # only a complete evaluation carries the final name
    return summary


def _group_of(protocol: dict) -> dict:
    return {i: g for g, members in protocol["groups"].items() for i in members}


def paired(protocol: dict, rows_a: list[dict], rows_b: list[dict], name_a: str, name_b: str) -> dict:
    """Equal-group bootstrap of A - B for every clip level and region."""
    from training.paired_gate import compare
    group = _group_of(protocol)
    index_b = {(r["image"], r["clip"]): r for r in rows_b}
    result = {}
    for clip in CLIP_LEVELS:
        result[clip] = {}
        for region in ("full", "saturated", "unsaturated"):
            a, b, scenes = {}, {}, {}
            for r in rows_a:
                if r["clip"] != clip:
                    continue
                other = index_b[(r["image"], clip)]
                va = r["methods"][name_a]["pu21_psnr_db"][region]
                vb = other["methods"][name_b]["pu21_psnr_db"][region]
                if math.isfinite(va) != math.isfinite(vb):
                    raise ValueError(f"{r['image']}/{clip}/{region}: asymmetric coverage")
                a[r["image"]], b[r["image"]] = {"m": va}, {"m": vb}
                scenes[r["image"]] = group[r["image"]]
            stats = compare(a, b, metrics=("m",), require_same_frames=True, scenes=scenes)["m"]
            result[clip][region] = stats
    return result


def summarize(protocol: dict, rows: list[dict], model: str, baseline: str) -> dict:
    stats = paired(protocol, rows, rows, model, baseline)
    group = _group_of(protocol)
    verdict = {}
    for clip, regions in stats.items():
        lo = lambda r: regions[r].get("ci95", [float("-inf")])[0]
        verdict[clip] = {"A1_full_frame_no_regression": lo("full") >= -0.25,
                         "A2_highlight_expansion": lo("saturated") > 0,
                         "A3_unsaturated_preservation": lo("unsaturated") >= -0.25}
    categories = {}
    for tag, members in protocol["categories"].items():
        per_clip = {}
        for clip in CLIP_LEVELS:
            gains = [r["methods"][model]["pu21_psnr_db"]["saturated"] - r["methods"][baseline]["pu21_psnr_db"]["saturated"]
                     for r in rows if r["clip"] == clip and r["image"] in members]
            gains = [g for g in gains if math.isfinite(g)]
            groups = {group[i] for i in members if i in group}
            per_clip[clip] = {"images": len(gains), "groups": len(groups),
                              "mean_saturated_gain_db": float(np.mean(gains)) if gains else None,
                              "flagged": bool(gains and len(groups) >= 2 and np.mean(gains) < -0.5)}
        categories[tag] = per_clip
    worst = {}
    for clip in CLIP_LEVELS:
        g = sorted(((r["methods"][model]["pu21_psnr_db"]["full"] - r["methods"][baseline]["pu21_psnr_db"]["full"],
                     r["image"]) for r in rows if r["clip"] == clip))
        worst[clip] = [dict(image=i, full_gain_db=d) for d, i in g[:10]]
    return {"protocol_version": PROTOCOL_VERSION, "model_minus_baseline": stats, "verdict": verdict,
            "A4_category_flags": [t for t, v in categories.items() if any(c["flagged"] for c in v.values())],
            "categories": categories, "A5_visual_review_required": worst,
            "A5_visual_review_done": False, "acceptance": protocol["acceptance"],
            "evidence_class": "independent HDR captures, SIMULATED SDR (not graded real SDR)",
            "release_authorized": False}


def compare_runs(protocol_path: Path, run_a: Path, run_b: Path) -> dict:
    protocol = load_protocol(protocol_path, require_same_code=False)
    read = lambda d: [json.loads(l) for l in (d / "per_image.jsonl").read_text().splitlines()]
    sa, sb = (json.loads((d / "summary.json").read_text()) for d in (run_a, run_b))
    if sa["protocol_sha256"] != sb["protocol_sha256"] or not (sa["complete"] and sb["complete"]):
        raise ValueError("runs must be complete and use the same frozen protocol")
    if sa["checkpoint_sha256"] == sb["checkpoint_sha256"]:
        raise ValueError("same checkpoint bytes")
    stats = paired(protocol, read(run_a), read(run_b), "model", "model")
    return {"a": sa["checkpoint"], "b": sb["checkpoint"], "a_minus_b": stats,
            "release_authorized": False, "promotion": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--prep", type=Path, required=True)
    f.add_argument("--anchors", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--protocol", type=Path, required=True)
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--limit", type=int)
    c = sub.add_parser("compare")
    c.add_argument("--protocol", type=Path, required=True)
    c.add_argument("--a", type=Path, required=True)
    c.add_argument("--b", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "freeze":
        result = freeze(args.prep, args.anchors, args.out)
        print(json.dumps({"frozen": str(args.out), "images": len(result["images"])}))
    elif args.cmd == "evaluate":
        result = evaluate(args.protocol, args.root, args.checkpoint, args.out, limit=args.limit)
        print(json.dumps({"verdict": result["verdict"], "flags": result["A4_category_flags"]}, indent=2))
    else:
        if args.out.exists():
            raise SystemExit("output exists")
        result = compare_runs(args.protocol, args.a, args.b)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({k: {r: v[r].get("ci95") for r in v} for k, v in result["a_minus_b"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
