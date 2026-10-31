"""Tears of Steel temporal kit: synthetic SDR from relative scene-linear composites.

Tears of Steel (CC BY 3.0, (CC) Blender Foundation | mango.blender.org) frames
are ungraded linear Rec.709 EXR composites with RELATIVE scale. There is no SDR
release of these plates, so the SDR here is rendered by RUDRA's own ACES 2.1
pipeline and is SYNTHETIC. The HDR reference is the same plate through the
same family's 1000-nit PQ transform. Results are "motion behaviour on mixed
live-action/CG composites with pipeline SDR" -- never real-SDR evidence.

Frozen recipe v1 (fixed before any model output; never re-exposed per model)
  * sections 03_2a (dim CG interior), 04_5f (live action, practical lamp, skin,
    handheld), 06_1a (moving sunlit metal specular): 48 contiguous frames each.
    00_1a is excluded (non-finite source frames); 04_4k is sparse stills and is
    used only as an explicitly SYNTHETIC cut pair, never as flicker evidence.
  * scene exposure 1.0 (Blender working-space values used as-is), Rec.709 ->
    BT.2020, ACES 2.1 SDR 100 nits (Rec.709) / HDR 1000 nits (P3-D65 lim),
    trim 0 EV. Conditions: clean (8-bit, no codec) and hard (the evaluation's
    seeded camera/codec degradation, export_bench_pairs.degrade_like_eval).
  * 04_5f lamp plateau (source exactly 11.80 +/- 0.01 in any channel, dilated
    3 px) is masked from highlight scoring: it is a source clip, not detail.
  * native source resolution only; nothing is upscaled.

Acceptance (engineering thresholds, frozen 21 Oct 2026, calibrated on the
existing synthetic-flicker smoke: 5% alternating gain -> 0.0007-0.0017 PQ,
25% highlight pop -> 0.012-0.017 PQ). Model vs the analytic bt1886 baseline on
the same frames, flow and masks, per section and condition:
  T1 mean excess error (model - baseline) <= 0.001 PQ
  T2 mean highlight excess (model - baseline) <= 0.005 PQ, where scored
  T3 no pair with model excess > baseline excess + 0.012 PQ (popping)
  T4 image inference is stateless: a post-cut frame scored alone equals the
     in-sequence frame (reset_check); the stateful `rudra video` path needs its
     own run.
PQ luminance, DIS flow with flat regions masked: not CVVDP, not a perceptual
pass, and smooth sky is under-sampled (use RAFT flow for final evidence).

    python training/tos_temporal_eval.py build --tos <ToS exr root> --out <new kit>
    python training/tos_temporal_eval.py evaluate --kit <kit> --checkpoint c.pt --out <new dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECIPE_VERSION = "rudra-tos-temporal-v1"
SECTIONS = {"03_2a": (133, 180), "04_5f": (1798, 1845), "06_1a": (96, 143)}
RECIPE = {"family": "aces2.1_1000", "config": "studio-config-v5.0.0_aces-v2.1_ocio-v2.6",
          "sdr_display": ["Rec.1886 Rec.709 - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"],
          "hdr_display": ["Rec.2100-PQ - Display", "ACES 2.0 - HDR 1000 nits (P3 D65)"],
          "hdr_peak_nits": 1000.0, "trim_ev": 0.0, "codec": "none", "quality": None}
PLATEAU = {"04_5f": (11.79, 11.81)}
THRESHOLDS = {"T1_mean_excess_pq": 0.001, "T2_highlight_excess_pq": 0.005, "T3_pair_pop_pq": 0.012}
DIFFUSE_WHITE = 203.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _frame_files(tos: Path, section: str) -> list[Path]:
    lo, hi = SECTIONS[section]
    files = {}
    for p in tos.rglob("*.exr"):
        if section not in str(p):
            continue
        digits = "".join(c for c in p.stem.split("_")[-1] if c.isdigit()) or "".join(c for c in p.stem if c.isdigit())
        n = int(digits[-6:]) if digits else -1
        if lo <= n <= hi:
            if n in files:
                raise ValueError(f"{section}: two files for frame {n}")
            files[n] = p
    if sorted(files) != list(range(lo, hi + 1)):
        raise ValueError(f"{section}: expected frames {lo}-{hi}, found {len(files)}")
    return [files[n] for n in range(lo, hi + 1)]


def _read_exr(path: Path) -> np.ndarray:
    from training.exr_any import read_rgb
    arr = read_rgb(path)
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: non-finite source values")
    return arr


def plateau_mask(linear709: np.ndarray, section: str) -> np.ndarray:
    import cv2
    if section not in PLATEAU:
        return np.zeros(linear709.shape[:2], bool)
    lo, hi = PLATEAU[section]
    m = ((linear709 >= lo) & (linear709 <= hi)).any(axis=2).astype(np.uint8)
    return cv2.dilate(m, np.ones((7, 7), np.uint8)).astype(bool)


def build(tos: Path, out: Path, render=None) -> dict:
    """render(linear2020, recipe) -> (sdr uint16 canonical, hdr linear 1.0=203 nits, code8)."""
    import cv2
    from rudra.delivery.colorspace import rgb_to_rgb_matrix
    from rudra.delivery.exr import write_exr
    if render is None:
        from pipeline.render_pipeline_pairs import render_pair as render
    final = out
    out = out.with_name(out.name + ".partial")
    if final.exists() or out.exists():
        raise ValueError(f"{final} (or {out.name}) exists; frozen kits are immutable")
    if render is None or getattr(render, "__module__", "") == "pipeline.render_pipeline_pairs":
        try:                                                               # the frozen OCIO config must exist
            import PyOpenColorIO as ocio
            ocio.Config.CreateFromBuiltinConfig(RECIPE["config"])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenColorIO cannot load the frozen config {RECIPE['config']} "
                               f"({type(exc).__name__}: {exc}); use the Python environment that rendered the "
                               "validation pairs (OCIO 2.6 built-ins), e.g. run_independent_eval.ps1 -Python <path>") from exc
    files = {section: _frame_files(tos, section) for section in SECTIONS}   # all frames present before writing
    _read_exr(files[next(iter(SECTIONS))][0])                              # and decodable
    m = rgb_to_rgb_matrix("rec709", "rec2020").astype(np.float32)
    out.mkdir(parents=True)
    rows = []
    for section in SECTIONS:
        for index, src in enumerate(files[section]):
            lin709 = _read_exr(src)
            sdr16, hdr, _ = render(np.maximum(lin709, 0) @ m.T, RECIPE)
            stem = f"{section}_{index:03d}"
            local = {"sdr": Path(section) / "sdr" / f"{stem}.png", "hdr": Path(section) / "hdr" / f"{stem}.exr",
                     "mask": Path(section) / "mask" / f"{stem}.png"}
            paths = {k: out / v for k, v in local.items()}
            for p in paths.values():
                p.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(paths["sdr"]), np.ascontiguousarray(sdr16[..., ::-1])):
                raise IOError(paths["sdr"])
            write_exr(paths["hdr"], hdr.astype(np.float32), half=False)
            if not cv2.imwrite(str(paths["mask"]), plateau_mask(lin709, section).astype(np.uint8) * 255):
                raise IOError(paths["mask"])
            rows.append({"section": section, "index": index, "source": str(src), "source_sha256": sha256_file(src),
                         "width": lin709.shape[1], "height": lin709.shape[0],
                         **{f"{k}_path": str(final / local[k]) for k in paths},
                         **{f"{k}_sha256": sha256_file(p) for k, p in paths.items()}})
    (out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    contract = {"version": RECIPE_VERSION, "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "frozen_before_model_scores": True, "recipe": RECIPE, "scene_exposure": 1.0,
                "sections": {s: {"frames": SECTIONS[s][1] - SECTIONS[s][0] + 1, "source_frames": list(SECTIONS[s])}
                             for s in SECTIONS},
                "excluded": {"00_1a": "non-finite source frames 66/71/78", "04_4k": "sparse stills; synthetic cuts only"},
                "masks": {"04_5f": "lamp plateau 11.80 (source clip), highlight scoring only"},
                "thresholds": THRESHOLDS, "conditions": ["clean", "hard"],
                "frames_sha256": sha256_file(out / "frames.jsonl"),
                "code_sha256": {"training/tos_temporal_eval.py": code_sha256(Path(__file__))},
                "evidence_class": "mixed live-action/CG composites, SYNTHETIC pipeline SDR, relative scale",
                "licence": "CC BY 3.0, (CC) Blender Foundation | mango.blender.org",
                "release_authorized": False}
    (out / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    out.rename(final)                              # media paths in frames.jsonl point at the final name
    return contract


def load_kit(kit: Path) -> tuple[dict, list[dict]]:
    contract = json.loads((kit / "contract.json").read_text(encoding="utf-8"))
    if contract.get("version") != RECIPE_VERSION:
        raise ValueError("unknown kit version")
    if sha256_file(kit / "frames.jsonl") != contract["frames_sha256"]:
        raise ValueError("frozen frame list changed")
    if contract["code_sha256"]["training/tos_temporal_eval.py"] != code_sha256(Path(__file__)):
        raise ValueError("training/tos_temporal_eval.py changed after the kit was frozen; build a new version")
    rows = [json.loads(l) for l in (kit / "frames.jsonl").read_text().splitlines()]
    for r in rows:
        for k in ("sdr", "hdr", "mask"):
            if sha256_file(Path(r[f"{k}_path"])) != r[f"{k}_sha256"]:
                raise ValueError(f"kit media changed: {r[f'{k}_path']}")
    return contract, rows


def section_verdict(model: dict, base: dict) -> dict:
    """Pure: thresholds applied to two evaluate_sequence results on identical frames/flow."""
    ms, bs = model["summary"], base["summary"]
    if ms["pairs_scored"] == 0 or ms["pairs_scored"] != bs["pairs_scored"]:
        return {"scored": False, "reason": "no or unequal scored pairs"}
    d_mean = ms["excess"]["mean"] - bs["excess"]["mean"]
    out = {"scored": True, "pairs": ms["pairs_scored"], "mean_excess_minus_baseline_pq": d_mean,
           "T1": d_mean <= THRESHOLDS["T1_mean_excess_pq"]}
    if ms.get("highlight_excess", {}).get("pairs"):
        d_h = ms["highlight_excess"]["mean"] - bs["highlight_excess"]["mean"]
        out.update(highlight_excess_minus_baseline_pq=d_h, T2=d_h <= THRESHOLDS["T2_highlight_excess_pq"])
    else:
        out.update(T2=None, T2_note="no highlight pixels above 203 nits scored")
    by_t = {r["t"]: r for r in base["pairs"] if "excess" in r}
    pops = [r["t"] for r in model["pairs"] if "excess" in r and r["t"] in by_t
            and r["excess"] > by_t[r["t"]]["excess"] + THRESHOLDS["T3_pair_pop_pq"]]
    out.update(T3=not pops, popping_pairs=pops)
    return out


def evaluate(kit: Path, checkpoint: Path, out: Path, predictor=None) -> dict:
    import cv2
    import torch
    from training.temporal_stability import evaluate_sequence, reset_check, _finite_or_none
    from rudra.delivery.bench import pu_psnr
    contract, rows = load_kit(kit)
    final, out = out, out.with_name(out.name + ".partial")
    if final.exists() or out.exists():
        raise ValueError(f"{final} (or {out.name}) exists; never overwritten")
    ckpt_sha = sha256_file(checkpoint)
    if predictor is None:
        from training.export_bench_pairs import degrade_like_eval, load_model
        from training.infer_sdr2hdr import predict_image
        from rudra.sdr2hdr import sdr_to_baseline_hdr
        from rudra.source_baseline import baseline_curve_index, correction_log2, knots_table
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(checkpoint, device)

        def predictor(sdr01, condition, index):
            x = torch.from_numpy(sdr01).permute(2, 0, 1)[None]
            if condition == "hard":
                x = degrade_like_eval(x[0], index)[None]
            x = x.to(device)
            with torch.inference_mode():
                curve = baseline_curve_index("bt1886") if getattr(model, "baseline_source", False) else None
                y = predict_image(model, x, preserve_outside=True, tile_size=512, overlap=64, bf16=False,
                                  baseline_curve=curve)
                ids = torch.tensor([baseline_curve_index("bt1886")], device=device)
                b = sdr_to_baseline_hdr(x.float(), model.corpus_ev) * torch.exp2(
                    correction_log2(x.float(), ids, knots_table(device)))
            n = lambda t: t[0].permute(1, 2, 0).float().cpu().numpy() * 10000.0   # network units -> nits
            return n(y), n(b)
    out.mkdir(parents=True)
    report = {"kit": str(kit), "contract_sha256": sha256_file(kit / "contract.json"), "checkpoint": str(checkpoint),
              "checkpoint_sha256": ckpt_sha, "sections": {}, "evidence_class": contract["evidence_class"],
              "thresholds": THRESHOLDS, "release_authorized": False}
    global_index = 0
    for section in contract["sections"]:
        frames = [r for r in rows if r["section"] == section]
        refs, masks, sdrs = [], [], []
        for r in frames:
            refs.append(_read_exr(Path(r["hdr_path"])) * DIFFUSE_WHITE)
            masks.append(cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE) > 0)
            sdrs.append(cv2.imread(r["sdr_path"], cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32) / 65535.0)
        for condition in contract["conditions"]:
            preds, bases, pu = [], [], []
            for i, sdr in enumerate(sdrs):
                y, b = predictor(sdr, condition, global_index + i)
                if not (np.isfinite(y).all() and np.isfinite(b).all()):
                    raise ValueError(f"{section}/{condition}/{i}: non-finite prediction")
                preds.append(y)
                bases.append(b)
                pu.append({"model": pu_psnr(y, refs[i]), "baseline": pu_psnr(b, refs[i])})
            # Plateau pixels: set identical in output, baseline and reference so they never score.
            hl_refs = [np.where(m[..., None], 0.0, r) for r, m in zip(refs, masks)]
            seq_m = evaluate_sequence([np.where(m[..., None], 0.0, p) for p, m in zip(preds, masks)], hl_refs, cuts=[])
            seq_b = evaluate_sequence([np.where(m[..., None], 0.0, p) for p, m in zip(bases, masks)], hl_refs, cuts=[])
            reset = reset_check(predictor(sdrs[len(sdrs) // 2], condition, global_index + len(sdrs) // 2)[0],
                                preds[len(sdrs) // 2])
            gains = [p["model"] - p["baseline"] for p in pu]
            report["sections"].setdefault(section, {})[condition] = _finite_or_none({
                "frames": len(frames), "pu21_psnr_gain_db_mean": float(np.mean(gains)),
                "pu21_psnr_gain_db_min": float(np.min(gains)),
                "temporal_model": seq_m["summary"], "temporal_baseline": seq_b["summary"],
                "verdict": {**section_verdict(seq_m, seq_b), "T4_stateless_reset": reset["reset_ok"]}})
        global_index += len(frames)
    if sha256_file(checkpoint) != ckpt_sha:
        raise ValueError("checkpoint changed during evaluation")
    (out / "temporal_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    out.rename(final)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tos", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--kit", type=Path, required=True)
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        print(json.dumps(build(a.tos, a.out)["sections"], indent=2))
    else:
        r = evaluate(a.kit, a.checkpoint, a.out)
        print(json.dumps({s: {c: v["verdict"] for c, v in d.items()} for s, d in r["sections"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
