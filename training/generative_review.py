"""Review kit for the MAIN mission: generated SDR video / images -> HDR.

Generated content has no HDR original, so nothing here is a fidelity score.
It measures, per item and per method, what can be measured without ground truth
and lays everything out for an HDR-display review:

  methods     analytic bt1886 baseline (the app's model-free path) and every
              --checkpoint given (label=path), same inference settings as the
              production kits (fp32, 512 tiles / 64 overlap, preserve_outside).
  dark        near-black input (max code <= 16): fraction of pixels the method
              lifts > 2 stops over the baseline, and log-opponent colour shift
              vs the baseline (the measured run02 defect: coloured blobs).
  highlights  clipped input (any channel >= 250): median / p99 nits and the
              share of the frame above 1000 nits (how much it expands).
  seams       gradient energy of log-luminance on the tile-boundary centre
              lines divided by the frame's average (1.0 = invisible; a ratio
              well above 1 is a tiling seam).
  flicker     videos are sampled in bursts of consecutive frames; per method,
              mean |frame-to-frame change of mean log2 luminance| beyond what
              the baseline does (EV). Global only; use `rudra video` reports
              for full-clip numbers.
  sheets      one PNG per item: SDR | baseline | each method | log2(method /
              baseline) heat map, same log tone map for all, for side-by-side
              review; plus review_sheet.md with the questions to answer on an
              HDR display. Flags are thresholds for attention, not verdicts.

    python training/generative_review.py --inputs <folder of generated clips/images> \
        --checkpoint run02=G:\\...\\best.pt [--checkpoint run03=...] --out <new dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
LUMA2020 = np.array([0.2627, 0.6780, 0.0593])
NEAR_BLACK, CLIPPED = 16, 250
TILE, OVERLAP = 512, 64
SEAM_VS_BASELINE = 1.2
VISIBLE_NITS = 0.5
# v2 (22 Oct, after the first public generated-clip run and BEFORE any run04 scoring): the
# relative dark measures stay in the report but no longer flag; flags use visible lift/colour.
FLAGS = {"dark_visible_lift_fraction": 0.01, "dark_visible_chroma_shift": 0.5, "seam_ratio": 1.5,
         "flicker_excess_ev": 0.02}


# ------------------------------------------------------------------ inputs
def list_items(folder: Path) -> list[Path]:
    items = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT | VIDEO_EXT)
    if not items:
        raise ValueError(f"no images or videos under {folder}")
    return items


def read_image(path: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)        # 8-bit BGR; alpha dropped, 16-bit reduced
    if img is None:
        raise ValueError(f"cannot decode {path}")
    return img[..., ::-1].copy()


def read_video_bursts(path: Path, bursts: int = 4, length: int = 3) -> list[list[np.ndarray]]:
    """`bursts` runs of `length` consecutive frames spread over the clip (8-bit RGB)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or n <= 0:
        raise ValueError(f"cannot decode {path}")
    starts = sorted({int(round(s)) for s in np.linspace(0, max(n - length, 0), bursts)})
    out = []
    for s in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        run = []
        for _ in range(length):
            ok, frame = cap.read()
            if not ok:
                break
            run.append(frame[..., ::-1].copy())
        if run:
            out.append(run)
    cap.release()
    if not out:
        raise ValueError(f"no frames decoded from {path}")
    return out


# --------------------------------------------------------------- inference
class Methods:
    """Analytic baseline plus models; every output in nits, linear BT.2020."""

    def __init__(self, checkpoints: dict[str, Path], device: str | None = None):
        import torch
        from training.export_bench_pairs import load_model
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = {label: load_model(path, self.device) for label, path in checkpoints.items()}
        self.corpus_ev = next(iter(self.models.values())).corpus_ev if self.models else 0.0

    def canonical(self, rgb8: np.ndarray):
        from rudra.delivery.colorspace import rgb_to_rgb_matrix
        from rudra.sdr2hdr import linear_to_srgb, srgb_to_linear
        torch = self.torch
        x = torch.from_numpy(rgb8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(self.device)
        m = torch.as_tensor(rgb_to_rgb_matrix("rec709", "rec2020"), device=self.device, dtype=x.dtype)
        return linear_to_srgb(torch.einsum("ij,bjhw->bihw", m, srgb_to_linear(x))).clamp(0, 1)

    def run(self, rgb8: np.ndarray) -> dict[str, np.ndarray]:
        from rudra.sdr2hdr import sdr_to_baseline_hdr
        from rudra.source_baseline import baseline_curve_index, correction_log2, knots_table
        from training.infer_sdr2hdr import predict_image
        torch = self.torch
        with torch.inference_mode():
            x = self.canonical(rgb8)
            ids = torch.tensor([baseline_curve_index("bt1886")], device=x.device)
            base = sdr_to_baseline_hdr(x.float(), self.corpus_ev) * torch.exp2(
                correction_log2(x.float(), ids, knots_table(x.device)))
            out = {"baseline_bt1886": base}
            for label, model in self.models.items():
                curve = baseline_curve_index("bt1886") if getattr(model, "baseline_source", False) else None
                out[label] = predict_image(model, x, preserve_outside=True, tile_size=TILE, overlap=OVERLAP,
                                           bf16=False, baseline_curve=curve)
        return {k: np.maximum(v[0].permute(1, 2, 0).float().cpu().numpy(), 0) * 10000.0 for k, v in out.items()}


# ----------------------------------------------------------------- metrics
def luminance(nits: np.ndarray) -> np.ndarray:
    return nits @ LUMA2020


def frame_metrics(rgb8: np.ndarray, pred: np.ndarray, base: np.ndarray) -> dict:
    """Pure: one frame, one method, against the analytic baseline."""
    y_p, y_b = luminance(pred), luminance(base)
    floor = 1e-3
    out = {}
    dark = rgb8.max(axis=2) <= NEAR_BLACK
    out["dark_pixels_fraction"] = float(dark.mean())
    if dark.sum() >= 64:
        lift = np.log2((y_p[dark] + floor) / (y_b[dark] + floor))
        out["dark_lift_fraction"] = float((lift > 2.0).mean())
        out["dark_lift_p99_stops"] = float(np.percentile(lift, 99))
        lp, lb = np.log(pred[dark] + floor), np.log(base[dark] + floor)
        opp = lambda l: np.stack([l[:, 0] - l[:, 1], l[:, 2] - l[:, 1]], 1)
        out["dark_chroma_shift"] = float(np.abs(opp(lp) - opp(lb)).mean())
        # Relative lift alone over-reports: 0.0001 -> 0.001 nit is +3 stops and invisible.
        # Visible lift: the method puts a near-black pixel above VISIBLE_NITS where the
        # baseline kept it below half that (grey/coloured haze on an OLED in a dim room).
        out["dark_visible_lift_fraction"] = float(((y_p[dark] > VISIBLE_NITS) & (y_b[dark] < VISIBLE_NITS / 2)).mean())
        out["dark_p99_nits"] = float(np.percentile(y_p[dark], 99))
        seen = np.minimum(y_p[dark], y_b[dark]) > VISIBLE_NITS / 10   # colour visible in both; lifts are caught above
        out["dark_visible_chroma_shift"] = (float(np.abs(opp(lp[seen]) - opp(lb[seen])).mean())
                                            if seen.sum() >= 64 else 0.0)
    clipped = rgb8.max(axis=2) >= CLIPPED
    out["clipped_pixels_fraction"] = float(clipped.mean())
    if clipped.sum() >= 16:
        out["clipped_median_nits"] = float(np.median(y_p[clipped]))
        out["clipped_p99_nits"] = float(np.percentile(y_p[clipped], 99))
    out["frame_above_1000_nits"] = float((y_p > 1000).mean())
    out["peak_nits_p9999"] = float(np.percentile(y_p, 99.99))
    out["seam_ratio"] = seam_ratio(y_p)
    return out


def seam_ratio(y: np.ndarray) -> float | None:
    """Log-luminance gradient on tile-overlap centre lines vs the frame average."""
    from rudra.inference import _tile_starts
    ly = np.log2(y + 1e-3)
    gx, gy = np.abs(np.diff(ly, axis=1)), np.abs(np.diff(ly, axis=0))
    xs = [s + OVERLAP // 2 for s in _tile_starts(y.shape[1], TILE, OVERLAP)[1:]]
    ys = [s + OVERLAP // 2 for s in _tile_starts(y.shape[0], TILE, OVERLAP)[1:]]
    lines = [gx[:, x - 1] for x in xs if 0 < x < gx.shape[1]] + [gy[y_ - 1, :] for y_ in ys if 0 < y_ < gy.shape[0]]
    if not lines:
        return None                                           # single tile: no seams possible
    return float(np.mean(np.concatenate(lines)) / max(float(np.mean(gx)), 1e-9))


def flicker_excess(bursts: list[list[dict]], label: str) -> float | None:
    """Mean |d mean-log2-Y| between consecutive frames, method minus baseline (EV)."""
    diffs = []
    for run in bursts:
        for a, b in zip(run, run[1:]):
            m = lambda r, k: float(np.mean(np.log2(luminance(r[k]) + 1e-3)))
            diffs.append(abs(m(b, label) - m(a, label)) - abs(m(b, "baseline_bt1886") - m(a, "baseline_bt1886")))
    return float(np.mean(diffs)) if diffs else None


# ------------------------------------------------------------------ sheets
def tone(nits: np.ndarray) -> np.ndarray:
    """Shared log display mapping: 0.01 nit -> black, 4000 nits -> white (review only)."""
    t = (np.log2(np.maximum(nits, 1e-4) / 0.01)) / np.log2(4000 / 0.01)
    return np.clip(t, 0, 1)


def heat(ratio_stops: np.ndarray) -> np.ndarray:
    """Diverging map, -3..+3 stops: grey = unchanged, red = brighter than baseline, blue = darker."""
    t = np.clip(ratio_stops / 3.0, -1, 1)[..., None]
    grey, red, blue = np.array([0.5, 0.5, 0.5]), np.array([1.0, 0.15, 0.1]), np.array([0.15, 0.35, 1.0])
    return np.where(t >= 0, grey + t * (red - grey), grey - t * (blue - grey))


def write_sheet(path: Path, rgb8: np.ndarray, preds: dict[str, np.ndarray], width: int = 640) -> None:
    import cv2
    h = int(rgb8.shape[0] * width / rgb8.shape[1])
    small = lambda img: cv2.resize(img.astype(np.float32), (width, h), interpolation=cv2.INTER_AREA)
    panels, names = [small(rgb8 / 255.0)], ["SDR input"]
    for k, v in preds.items():
        panels.append(small(tone(v)))
        names.append(k)
    for k, v in preds.items():
        if k != "baseline_bt1886":
            panels.append(small(heat(np.log2((luminance(v) + 1e-3) / (luminance(preds["baseline_bt1886"]) + 1e-3)))))
            names.append(f"{k} vs baseline: red brighter, blue darker (+/-3 stops)")
    tiles = []
    for img, name in zip(panels, names):
        canvas = np.zeros((h + 24, width, 3), np.uint8)       # OpenCV 5 draws text on 8-bit only
        canvas[24:] = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
        cv2.putText(canvas, name, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.ascontiguousarray(grid[..., ::-1])):
        raise IOError(f"cannot write {path}")


# --------------------------------------------------------------------- run
def review(inputs: Path, checkpoints: dict[str, Path], out: Path, methods=None, bursts: int = 4) -> dict:
    if out.exists():
        raise ValueError(f"{out} exists; reviews are never overwritten")
    items = list_items(inputs)
    methods = methods or Methods(checkpoints)
    out.mkdir(parents=True)
    report = {"inputs": str(inputs), "checkpoints": {k: str(v) for k, v in checkpoints.items()},
              "flag_thresholds": FLAGS, "items": [], "evidence_class":
              "generated SDR, no HDR ground truth: artefact measurements and review sheets only",
              "release_authorized": False}
    for item in items:
        is_video = item.suffix.lower() in VIDEO_EXT
        try:
            runs = read_video_bursts(item, bursts) if is_video else [[read_image(item)]]
        except ValueError as exc:
            report["items"].append({"item": str(item), "error": str(exc)})
            continue
        frames = [[{"rgb8": f, **methods.run(f)} for f in run] for run in runs]
        labels = [k for k in frames[0][0] if k not in ("rgb8",)]
        entry = {"item": str(item.relative_to(inputs)), "kind": "video" if is_video else "image",
                 "frames_sampled": sum(len(r) for r in frames), "size": list(runs[0][0].shape[:2]), "methods": {}}
        for label in labels:
            per = [frame_metrics(f["rgb8"], f[label], f["baseline_bt1886"]) for run in frames for f in run]
            agg = {}
            for key in per[0]:
                vals = [p[key] for p in per if p.get(key) is not None]
                if vals:
                    agg[key] = float(np.max(vals)) if key in ("dark_lift_fraction", "dark_chroma_shift", "seam_ratio", "dark_visible_lift_fraction",
                                                             "dark_visible_chroma_shift", "dark_p99_nits",
                                                             "dark_lift_p99_stops") else float(np.mean(vals))
            if is_video and label != "baseline_bt1886":
                agg["flicker_excess_ev"] = flicker_excess(frames, label)
            agg["flags"] = [k for k, t in FLAGS.items() if agg.get(k) is not None and label != "baseline_bt1886"
                            and agg[k] > t]
            entry["methods"][label] = agg
        # The seam metric also responds to content that happens to lie on a tile line; the
        # baseline has no tiles, so a method's seam flag needs a ratio well above the baseline's.
        base_seam = entry["methods"].get("baseline_bt1886", {}).get("seam_ratio")
        if base_seam is not None:
            for label, agg in entry["methods"].items():
                if "seam_ratio" in agg.get("flags", []) and agg["seam_ratio"] <= SEAM_VS_BASELINE * base_seam:
                    agg["flags"].remove("seam_ratio")
        mid = frames[len(frames) // 2][0]
        sheet = out / "sheets" / (item.relative_to(inputs).as_posix().replace("/", "__") + ".png")
        write_sheet(sheet, mid["rgb8"], {k: mid[k] for k in labels})
        entry["sheet"] = str(sheet.relative_to(out))
        report["items"].append(entry)
        print(f"[review] {entry['item']}: " + ", ".join(f"{k} flags={v['flags']}" for k, v in entry["methods"].items()
                                                         if k != "baseline_bt1886"), flush=True)
    (out / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_review_sheet(out / "review_sheet.md", report)
    return report


def write_review_sheet(path: Path, report: dict) -> None:
    lines = ["# Generated-content HDR review", "",
             "Measured flags point at frames to inspect; they are not verdicts. No HDR ground truth exists for",
             "generated content. Review on a calibrated HDR display at its real peak, with the app's View peak",
             "set to match, and mark each item for every method.", "",
             "| Item | Method | Flags | Highlights natural? | Blacks clean (no blobs/lift)? | Skin/mid-tones unchanged? | Banding? | Flicker/pumping? | Prefer over baseline? |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for it in report["items"]:
        for m, v in it.get("methods", {}).items():
            if m != "baseline_bt1886":
                lines.append(f"| {it['item']} | {m} | {', '.join(v['flags']) or '-'} |  |  |  |  |  |  |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--checkpoint", action="append", default=[], help="label=path; repeatable; none = baseline only")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bursts", type=int, default=4)
    ap.add_argument("--device")
    a = ap.parse_args()
    ckpts = {}
    for spec in a.checkpoint:
        label, _, path = spec.partition("=")
        if not label or not path or label == "baseline_bt1886":
            raise SystemExit(f"bad --checkpoint {spec!r}; use label=path")
        ckpts[label] = Path(path)
    report = review(a.inputs, ckpts, a.out, methods=Methods(ckpts, a.device), bursts=a.bursts)
    flagged = sum(1 for it in report["items"] for m, v in it.get("methods", {}).items() if v.get("flags"))
    print(json.dumps({"items": len(report["items"]), "method_items_flagged": flagged,
                      "review": str(a.out / "review.json"), "sheets": str(a.out / "sheets")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
