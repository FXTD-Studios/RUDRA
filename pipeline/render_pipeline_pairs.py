"""Matched SDR / HDR display renders of FXTD footage through real ACES output
transforms (17 Oct 2026).

There is no graded SDR/HDR version of FXTD's footage, so this makes the next
best thing: each shot rendered TWICE from the same scene-linear plate, through
the Academy's own output transforms as OpenColorIO ships them (ACES 1.3 and
ACES 2.1 studio configs, built into OCIO 2.6), once to an SDR display and once
to a PQ HDR display -- the way an ACES-managed show produces its SDR and HDR
deliverables when nobody trims. The pair is display-referred on both sides,
like the Netflix pairs, and unlike the corpus's rendered rows (whose target is
the scene-linear plate).

These are NOT real grades: no colourist chose anything. They are rendered rows
(``sdr_kind`` "rendered", ``render_family`` "pipeline") and must never be used
as real-SDR evidence. Their value is a second, industry-standard way SDR and
HDR relate, on independent owned scenes.

Input: rudra_fxtd_camera (pipeline/ingest_fxtd_footage.py). Its HDR targets
are the camera plates in linear BT.2020 with 1.0 = camera 1.0 (18 % grey at
0.18), exactly what an ACES input transform produces, so they feed OCIO's
"Linear Rec.2020" directly. Same shots, same scene_id, same split.

Per shot (a show is mastered once per shot), drawn from a hash of the scene:
  family    ACES 1.3 (SDR Video / HDR Video 1000 nits Rec.2020 lim) or
            ACES 2.1 (SDR 100 nits Rec.709 / HDR 1000 or 4000 nits, P3-D65 lim)
  trim      exposure +/- TRIM_EV stops, shared by SDR and HDR (a grade's offset)
  codec     SDR 8-bit, then half the shots through JPEG / H.264 / HEVC / AV1

Conversions are fetch_netflix_pairs.py's, so the rows sit in the same
convention as the real pairs:
  SDR  BT.1886 Rec.709 code -> 8 bits (-> codec) -> inverse BT.709 OETF
       (rudra/video.py 'rec709') -> 709 -> 2020 -> sRGB OETF -> uint16 PNG
  HDR  PQ code (Rec.2100 container) -> nits -> /203 -> log2_extended uint16 PNG

Censoring: an ACES HDR output saturates at its peak, and above a sensor clip
the plate itself is unknown. ``ceiling_nits`` is CEILING_MARGIN x the lower of
the display peak and (for a clipped shot) the HDR render of the clip white.

    python pipeline/render_pipeline_pairs.py --camera G:/datasets/corpora/rudra_fxtd_camera_20261016 \
        --out G:/datasets/corpora/rudra_fxtd_pipeline_20261017
    python pipeline/render_pipeline_pairs.py --camera ... --out ... --finalize --path-root G:\\datasets\\corpora\\rudra_fxtd_pipeline_20261017
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline import sdr_render  # noqa: E402
from pipeline.hdr_io import (HDR_IO_VERSION, HDRStorage, decode_hdr_u16, encode_hdr_u16,  # noqa: E402
                             pq_eotf, selftest)
from pipeline.licences import annotate  # noqa: E402
from rudra.delivery.colorspace import rgb_to_rgb_matrix  # noqa: E402

CORPUS = "rudra_fxtd_pipeline"
STORAGE = HDRStorage(mode="log2_extended")
DIFFUSE_WHITE = 203.0
TRIM_EV = 0.5
CODEC_PROBABILITY = 0.5
CEILING_MARGIN = 0.97
SEED = 20261017
M_709_2020 = rgb_to_rgb_matrix("rec709", "rec2020").astype(np.float32)

ACES1 = "studio-config-v2.2.0_aces-v1.3_ocio-v2.4"
ACES2 = "studio-config-v5.0.0_aces-v2.1_ocio-v2.6"
# (name, config, (sdr display, view), (hdr display, view), hdr peak nits, weight)
FAMILIES = (
    ("aces1.3_1000", ACES1, ("Rec.1886 Rec.709 - Display", "ACES 1.0 - SDR Video"),
     ("Rec.2100-PQ - Display", "ACES 1.1 - HDR Video (1000 nits & Rec.2020 lim)"), 1000.0, 0.45),
    ("aces2.1_1000", ACES2, ("Rec.1886 Rec.709 - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"),
     ("Rec.2100-PQ - Display", "ACES 2.0 - HDR 1000 nits (P3 D65)"), 1000.0, 0.40),
    ("aces2.1_4000", ACES2, ("Rec.1886 Rec.709 - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"),
     ("Rec.2100-PQ - Display", "ACES 2.0 - HDR 4000 nits (P3 D65)"), 4000.0, 0.15),
)
_PROCESSORS: dict = {}


def processor(config: str, display: str, view: str):
    key = (config, display, view)
    if key not in _PROCESSORS:
        import PyOpenColorIO as ocio
        cfg = ocio.Config.CreateFromBuiltinConfig(config)
        t = ocio.DisplayViewTransform(src="Linear Rec.2020", display=display, view=view)
        _PROCESSORS[key] = cfg.getProcessor(t).getDefaultCPUProcessor()
    return _PROCESSORS[key]


def apply(proc, rgb: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(rgb, dtype=np.float32).copy()
    proc.applyRGB(img)          # in place, (H, W, 3) float32
    return img


def shot_recipe(scene_id: str, seed: int = SEED, codecs: list[str] | None = None) -> dict:
    rng = np.random.default_rng([seed, int(hashlib.sha256(scene_id.encode()).hexdigest()[:8], 16)])
    weights = np.array([f[-1] for f in FAMILIES])
    fam = FAMILIES[int(rng.choice(len(FAMILIES), p=weights / weights.sum()))]
    recipe = {"family": fam[0], "config": fam[1], "sdr_display": list(fam[2]), "hdr_display": list(fam[3]),
              "hdr_peak_nits": fam[4], "trim_ev": round(float(rng.uniform(-TRIM_EV, TRIM_EV)), 4),
              "codec": "none", "quality": None}
    if rng.random() < CODEC_PROBABILITY:
        pool = ["jpeg"] + list(codecs if codecs is not None else sdr_render.available_video_codecs())
        recipe["codec"] = str(rng.choice(pool))
        recipe["quality"] = int(rng.integers(40, 93)) if recipe["codec"] == "jpeg" else int(rng.integers(18, 36))
    return recipe


def _bt709_inverse_oetf(v: np.ndarray) -> np.ndarray:
    return np.where(v < 0.081, v / 4.5, np.power((v + 0.099) / 1.099, 1.0 / 0.45))


def _srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def sdr_canonical(code709_u8: np.ndarray) -> np.ndarray:
    """8-bit BT.709 display code -> canonical sRGB code in BT.2020, uint16 (fetch_netflix_pairs.convert_sdr)."""
    lin = _bt709_inverse_oetf(code709_u8.astype(np.float32) / 255.0).astype(np.float32)
    lin2020 = np.clip(lin @ M_709_2020.T, 0.0, 1.0)
    return np.rint(_srgb_oetf(lin2020) * 65535.0).astype(np.uint16)


def render_pair(linear2020: np.ndarray, recipe: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sdr uint16 canonical, hdr linear 1.0=203 nits in BT.2020, sdr 8-bit 709 code)."""
    x = np.maximum(linear2020, 0.0) * np.float32(2.0 ** recipe["trim_ev"])
    sdr = apply(processor(recipe["config"], *recipe["sdr_display"]), x)
    code8 = np.clip(np.rint(np.clip(sdr, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    if recipe["codec"] == "jpeg":
        code8 = sdr_render._jpeg(code8, recipe["quality"])
    elif recipe["codec"] != "none":
        code8 = sdr_render._video_codec(code8, recipe["codec"], recipe["quality"])
    pq = apply(processor(recipe["config"], *recipe["hdr_display"]), x)
    nits = pq_eotf(np.clip(pq, 0.0, 1.0)).astype(np.float32)
    return sdr_canonical(code8), nits / DIFFUSE_WHITE, code8


def clip_white_nits(recipe: dict, clip_linear: float) -> float:
    """HDR render (nits) of a neutral at the sensor clip, after the shot's trim."""
    x = np.full((1, 1, 3), clip_linear * 2.0 ** recipe["trim_ev"], np.float32)
    return float(pq_eotf(np.clip(apply(processor(recipe["config"], *recipe["hdr_display"]), x), 0, 1)).max())


def shot_ceiling(recipe: dict, camera_ceiling_nits: float | None) -> float:
    peak = recipe["hdr_peak_nits"]
    if camera_ceiling_nits:
        peak = min(peak, clip_white_nits(recipe, camera_ceiling_nits / CEILING_MARGIN / DIFFUSE_WHITE))
    return CEILING_MARGIN * peak


# ----------------------------------------------------------------------------
def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"failed to encode {path}")
    tmp = path.with_suffix(".tmp.png")
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    os.replace(tmp, path)


def read_camera_rows(camera: Path) -> list[dict]:
    rows = []
    for p in sorted(camera.glob("_records*.jsonl")):
        rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if not r.get("excluded")]


def process_row(cam_row: dict, camera: Path, cam_storage: HDRStorage, out: Path) -> dict:
    code = cv2.imread(str(camera / cam_row["hdr_path"]), cv2.IMREAD_UNCHANGED)
    if code is None:
        raise IOError(f"cannot read {camera / cam_row['hdr_path']}")
    linear = decode_hdr_u16(cv2.cvtColor(code, cv2.COLOR_BGR2RGB), cam_storage)
    recipe = shot_recipe(cam_row["scene_id"])
    sdr, hdr_lin, code8 = render_pair(linear, recipe)
    hdr, stats = encode_hdr_u16(hdr_lin, STORAGE)
    aid = cam_row["asset_id"] + "_pipe"
    ceiling = shot_ceiling(recipe, cam_row.get("ceiling_nits"))
    _write_png(out / "hdr" / f"{aid}.png", hdr)
    _write_png(out / "sdr" / f"{aid}.png", sdr)
    lum = hdr_lin @ np.array([0.2627, 0.6780, 0.0593], dtype=np.float32) * DIFFUSE_WHITE
    keep = ("scene_id", "split", "project", "source_id", "shot", "frame_index", "shot_frames",
            "source_path", "camera", "source_encoding", "source_gamut")
    return {
        "asset_id": aid, **{k: cam_row[k] for k in keep},
        "sdr_path": f"sdr/{aid}.png", "hdr_path": f"hdr/{aid}.png",
        "width": int(sdr.shape[1]), "height": int(sdr.shape[0]),
        "sdr_kind": "rendered", "render_family": "pipeline",
        "sdr_source": f"{recipe['family']}: {' / '.join(recipe['sdr_display'])}, 8-bit"
                      + ("" if recipe["codec"] == "none" else f", {recipe['codec']}"),
        "hdr_source": f"{recipe['family']}: {' / '.join(recipe['hdr_display'])}",
        "pipeline_recipe": recipe, "source_curve": "unknown", "sdr_curve": recipe["family"],
        "sdr_codec": recipe["codec"], "tonemap_ev": None, "sdr_encoding": "srgb2020_from_bt709",
        "hdr_encoding": STORAGE.mode, "camera_asset_id": cam_row["asset_id"],
        "ceiling_nits": ceiling,
        "peak_at_ceiling": bool(stats["peak_nits"] >= ceiling),
        "peak_nits": stats["peak_nits"], "clipped_fraction": stats["clipped_fraction"],
        "sdr_clipped_fraction": float((code8.max(-1) >= 254).mean()),
        "sdr_mean_code": float(sdr.mean() / 65535.0),
        "hdr_p50_nits": float(np.percentile(lum, 50)), "hdr_p99_nits": float(np.percentile(lum, 99)),
        "is_video": False, "excluded": None,
    }


def write_sentinel(out: Path) -> None:
    import PyOpenColorIO as ocio
    payload = {
        "corpus": CORPUS, "created": time.strftime("%Y-%m-%d"), "hdr_io_version": HDR_IO_VERSION,
        "storage": STORAGE.as_dict(), "tonemap": "ocio-aces-pipeline:v1", "tonemap_ev": None,
        "sdr_encoding": "canonical sRGB code in BT.2020 from 8-bit BT.1886 Rec.709 (as fetch_netflix_pairs)",
        "hdr_convention": "ACES HDR output, Rec.2100-PQ container -> nits -> /203 -> log2_extended",
        "families": [{"name": f[0], "config": f[1], "sdr": f[2], "hdr": f[3], "peak_nits": f[4], "weight": f[5]}
                     for f in FAMILIES],
        "ocio_version": ocio.__version__, "trim_ev": TRIM_EV, "codec_probability": CODEC_PROBABILITY,
        "seed": SEED, "warning": "rendered, not graded: never use as real-SDR evidence",
        "licence": "FXTD Studios own footage; ACES configs (Academy, BSD-3-Clause) via OpenColorIO",
        "tool": "pipeline/render_pipeline_pairs.py",
    }
    path = out / "_ingest_config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        for key in ("hdr_io_version", "storage", "tonemap", "families"):
            if old.get(key) != json.loads(json.dumps(payload[key])):    # tuples come back as lists
                raise SystemExit(f"{path}: {key} differs; use a fresh --out")
        return
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_records(out: Path) -> dict[str, dict]:
    recs = {}
    for p in sorted(out.glob("_records*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["asset_id"]] = r
    return recs


def _absolute(root: str, rel: str) -> str:
    windows = "\\" in root or (len(root) > 1 and root[1] == ":")
    sep = "\\" if windows else "/"
    return root.rstrip("\\/") + sep + rel.replace("/", sep)


def finalize(out: Path, root: str | None = None) -> dict:
    root = root or str(out.resolve())
    kept = sorted(load_records(out).values(), key=lambda r: (r["split"], r["asset_id"]))
    for r in kept:
        for k in ("sdr_path", "hdr_path"):
            if not (out / r[k]).exists():
                raise SystemExit(f"{r['asset_id']}: {r[k]} missing")
            r[k] = _absolute(root, r[k])
    annotate(kept)
    if any(r.get("licence_source") != "fxtd" for r in kept):
        raise SystemExit("licence classifier did not tag every row fxtd")
    manifest = out / "sdr_hdr_manifest.jsonl"
    with open(manifest, "w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    scenes = defaultdict(set)
    for r in kept:
        scenes[r["scene_id"]].add(r["split"])
    report = {
        "manifest": _absolute(root, "sdr_hdr_manifest.jsonl"),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "records": len(kept), "scenes": len(scenes),
        "scenes_straddling_splits": sorted(s for s, v in scenes.items() if len(v) > 1),
        "by_split": dict(Counter(r["split"] for r in kept)),
        "families": dict(Counter(r["pipeline_recipe"]["family"] for r in kept)),
        "codecs": dict(Counter(r["sdr_codec"] for r in kept)),
        "rows_peak_at_ceiling": sum(1 for r in kept if r["peak_at_ceiling"]),
        "hdr_peak_nits_p50": float(np.median([r["peak_nits"] for r in kept])) if kept else None,
        "sdr_clipped_rows": sum(1 for r in kept if r["sdr_clipped_fraction"] > 1e-3),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--camera", required=True, type=Path, help="rudra_fxtd_camera corpus dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sources", nargs="*", default=None, help="limit to these source ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--time-budget", type=float, default=0.0)
    ap.add_argument("--tag", default="all", help="records file suffix, so parallel workers do not share one")
    ap.add_argument("--shard", default="0/1", help="i/n: this worker takes every n-th row from i")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--path-root", default=None)
    args = ap.parse_args(argv)
    if args.finalize:
        print(json.dumps(finalize(args.out, args.path_root), indent=2))
        return 0
    selftest(STORAGE)
    write_sentinel(args.out)
    cam_storage = HDRStorage.from_dict(json.loads((args.camera / "_ingest_config.json").read_text())["storage"])
    rows = read_camera_rows(args.camera)
    if args.sources:
        rows = [r for r in rows if r["source_id"] in args.sources]
    i, n = map(int, args.shard.split("/"))
    rows = sorted(rows, key=lambda r: r["asset_id"])[i::n]
    if args.limit:
        rows = rows[: args.limit]
    done = set(load_records(args.out))
    t0, wrote = time.time(), 0
    with open(args.out / f"_records.{args.tag}.jsonl", "a", encoding="utf-8", newline="\n") as log:
        for r in rows:
            if r["asset_id"] + "_pipe" in done:
                continue
            if args.time_budget and time.time() - t0 > args.time_budget:
                print(f"time budget reached after {wrote}; re-run to resume", flush=True)
                return 3
            rec = process_row(r, args.camera, cam_storage, args.out)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            wrote += 1
    print(f"done: wrote {wrote}; run --finalize", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
