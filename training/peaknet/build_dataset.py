"""Self-supervised PeakNet regions from commercial scene-linear HDR (17 Oct 2026).

PeakNet does not reconstruct pixels. The shipped clip fill (native/core/src/
clip_fill.cpp) already fills every fully clipped region with a smooth level
from its edge plus a dome ``a * (2u - u^2)``; the only free number per region
is the dome height ``a`` (stops), which the app exposes as the region's peak
(ClipOverride nits = centre level * 2^a). PeakNet predicts ``a`` per region
from the SDR. Everything else stays exactly the shipped fill, so outside
fully clipped pixels nothing can change.

Self-supervised: no SDR grades are needed. For each clean HDR frame:
  1. render an SDR with pipeline/sdr_render: ACES (the curve the fill's
     baseline inverts), pushed +0..PUSH_MAX EV so it clips, sometimes a
     codec, 8-bit;
  2. run the real clip fill through native/tools/clip_fill_probe three times:
     default (what ships), dome 0 (the edge fill alone), dome ~1 (to read the
     dome's per-pixel profile p);
  3. per region, the error in stops inside it as a function of a,
        E(a) = mean_i | log2 min(Y0_i 2^(a p_i), 1) - log2 min(Y_i, 1) |,
     sampled on A_GRID (Y = true luma in network units, capped at 10,000
     nits, the PQ ceiling). Where the HDR itself sits at a sensor clip
     (FXTD rows' ceiling_nits) the error is one-sided (truth is >= that).
  4. store features, a 64 x 64 context patch, E(a), and the shipped default's
     own error E_def, so training can minimise exactly the gate's metric and
     report PeakNet against the default region by region and frame by frame.

Sources (strict commercial tier only):
  polyhaven_moves  G:\\rudra_v02\\pairs_moves (CC0 Poly Haven views), two frames
                   per clip; split by panorama slug, matching the strict
                   manifest's Poly Haven val/test scenes.
  fxtd_camera      rudra_fxtd_camera_20261016 rows of the strict manifest.

    python training/peaknet/build_dataset.py --strict <strict manifest> --moves G:/rudra_v02/pairs_moves \
        --probe build/clip_fill_probe --out G:/datasets/corpora/rudra_peaknet_v1_20261017
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline import sdr_render  # noqa: E402
from pipeline.hdr_io import HDRStorage, decode_hdr_u16  # noqa: E402

A_GRID = np.round(np.arange(0.0, 12.0001, 0.125), 4).astype(np.float32)   # 97 dome heights, stops
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)                      # clip_fill.cpp kLuma
NETWORK_NITS = 10000.0
DIFFUSE_WHITE = 203.0
PATCH = 64
MIN_PIXELS = 9
MAX_REGIONS = 32
PUSH_MAX = 2.5
EXPOSURES_PER_FRAME = 2
FEATURES = ("log2_pixels", "log2_fraction", "log2_radius", "large", "log2_edge_nits", "glow_ev_per_px",
            "log2_centre_nits", "a_default", "touches_border", "cx", "cy", "log2_long_side")
SEED = 20261017


# ----------------------------------------------------------------------------
# The native fill
# ----------------------------------------------------------------------------
def run_probe(probe: str, codes: np.ndarray, glow: float | None = None,
              overrides: list[tuple[float, float, float]] | None = None, tmp: Path | None = None
              ) -> tuple[np.ndarray, list[dict]]:
    """codes (H, W, 3) float32 in [0, 1] -> (filled (H, W, 3) network units, regions)."""
    tmp = Path(tmp or tempfile.gettempdir())
    h, w = codes.shape[:2]
    tag = f"{os.getpid()}_{time.monotonic_ns()}"
    fin, fout, freg, fov = (tmp / f"cfp_{tag}{s}" for s in (".in", ".out", ".tsv", ".ov"))
    fin.write_bytes(struct.pack("ii", h, w) + np.ascontiguousarray(codes.transpose(2, 0, 1), np.float32).tobytes())
    cmd = [probe, str(fin), str(fout), str(freg), "--corpus-ev", "0"]
    if glow is not None:
        cmd += ["--glow", str(glow)]
    if overrides:
        fov.write_text("".join(f"{x:.8f} {y:.8f} {n:.6f}\n" for x, y, n in overrides))
        cmd += ["--overrides", str(fov)]
    subprocess.run(cmd, check=True, capture_output=True)
    out = np.frombuffer(fout.read_bytes()[8:], np.float32).reshape(3, h, w).transpose(1, 2, 0)
    lines = freg.read_text().splitlines()
    keys = lines[0].split("\t")
    regions = [dict(zip(keys, (float(v) for v in line.split("\t")))) for line in lines[1:]]
    for f in (fin, fout, freg, fov):
        if f.exists():
            f.unlink()
    return out, regions


# ----------------------------------------------------------------------------
# Regions and targets
# ----------------------------------------------------------------------------
def full_clip_labels(codes: np.ndarray, clip_code: float = 0.995) -> tuple[int, np.ndarray]:
    """8-connected components of pixels whose every channel is at or above clip_code (clip_fill kFull)."""
    full = (codes >= clip_code).all(-1).astype(np.uint8)
    n, labels = cv2.connectedComponents(full, connectivity=8)
    return n, labels


def error_curve(y0: np.ndarray, p: np.ndarray, y_true: np.ndarray, censored: np.ndarray,
                grid: np.ndarray = A_GRID) -> np.ndarray:
    """E(a) for one region: mean |log2 pred - log2 truth| in stops, one-sided where censored."""
    lt = np.log2(np.clip(y_true, 1e-9, 1.0))[None, :]
    lp = np.minimum(np.log2(np.maximum(y0, 1e-9))[None, :] + grid[:, None] * p[None, :], 0.0)
    err = np.where(censored[None, :], np.maximum(lt - lp, 0.0), np.abs(lp - lt))
    return err.mean(1).astype(np.float32)


def region_error(y_pred: np.ndarray, y_true: np.ndarray, censored: np.ndarray) -> float:
    lt = np.log2(np.clip(y_true, 1e-9, 1.0))
    lp = np.log2(np.clip(y_pred, 1e-9, 1.0))
    return float(np.where(censored, np.maximum(lt - lp, 0.0), np.abs(lp - lt)).mean())


def patch_for(codes, y0_log, region_mask, any_full, box, size=PATCH) -> np.ndarray:
    """(6, size, size) float16: SDR RGB codes, this region, any full clip, base log2 luma (scaled)."""
    y0, x0, y1, x1 = box
    stack = np.concatenate([codes[y0:y1, x0:x1], region_mask[y0:y1, x0:x1, None].astype(np.float32),
                            any_full[y0:y1, x0:x1, None].astype(np.float32),
                            ((y0_log[y0:y1, x0:x1, None] + 14.0) / 14.0).astype(np.float32)], axis=-1)
    # OpenCV resizes at most 4 channels at a time.
    planes = [cv2.resize(np.ascontiguousarray(stack[..., c]), (size, size), interpolation=cv2.INTER_AREA)
              for c in range(stack.shape[-1])]
    return np.stack(planes).astype(np.float16)


def region_inputs(codes: np.ndarray, y0_log: np.ndarray, mask: np.ndarray, any_full: np.ndarray, r0: dict,
                  a_def: float) -> tuple[np.ndarray, np.ndarray]:
    """PeakNet's inputs for one region: FEATURES and the context patch. Shared by training and inference.

    ``r0``: the region as clip_fill_probe reports it with the dome off (glow 0); ``a_def``: the
    shipped default's dome height for it, log2(default peak / r0 peak)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    pad = max(16, int(0.75 * max(np.ptp(ys) + 1, np.ptp(xs) + 1)))
    box = (max(0, ys.min() - pad), max(0, xs.min() - pad), min(h, ys.max() + 1 + pad), min(w, xs.max() + 1 + pad))
    touches = bool(ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1)
    feats = np.array([np.log2(r0["pixels"]), np.log2(r0["fraction"]), np.log2(r0["radius_px"] + 1.0),
                      r0["large"], np.log2(max(r0["edge_nits"], 1e-3)), r0["glow_ev_per_px"],
                      np.log2(max(r0["peak_nits"], 1e-3)), a_def, float(touches), r0["x"] / w, r0["y"] / h,
                      np.log2(max(h, w))], np.float32)
    return feats, patch_for(codes, y0_log, mask, any_full, box)


def frame_regions(probe: str, lin: np.ndarray, recipe: dict, ceiling_nits: float | None, tmp: Path,
                  meta: dict, predicted: dict[int, float] | None = None) -> list[dict]:
    """All PeakNet samples of one rendered frame. ``lin``: scene-linear (H, W, 3), 1.0 = 203 nits.

    ``predicted`` {region index: dome height a}: also run the real fill with those
    peaks as overrides and record each region's error under it (``e_native``), the
    check that E(a) is what the shipped code actually produces."""
    sdr8 = sdr_render.apply_render(lin, recipe, 0.0)
    codes = sdr8.astype(np.float32) / 255.0
    n, labels = full_clip_labels(codes)
    if n <= 1:
        return []
    h, w = labels.shape
    y_true = (lin * (DIFFUSE_WHITE / NETWORK_NITS)) @ LUMA
    censor = (y_true >= 0.97 * ceiling_nits / NETWORK_NITS) if ceiling_nits else np.zeros_like(y_true, bool)
    filled_def, regs = run_probe(probe, codes, tmp=tmp)
    filled0, regs0 = run_probe(probe, codes, glow=0.0, tmp=tmp)
    if len(regs0) != len(regs):
        return []
    # Dome ~1 stop on every region, through the override path the app uses.
    ov = [((r["x"] + 0.5) / w, (r["y"] + 0.5) / h, 2.0 * r["peak_nits"]) for r in regs0]
    filled1, regs1 = run_probe(probe, codes, overrides=ov, tmp=tmp)
    y_def, y0, y1 = (f @ LUMA for f in (filled_def, filled0, filled1))
    y_nat = None
    if predicted:
        ovp = [((regs0[j]["x"] + 0.5) / w, (regs0[j]["y"] + 0.5) / h, regs0[j]["peak_nits"] * 2.0 ** a)
               for j, a in predicted.items() if j < len(regs0)]
        filled_p, _ = run_probe(probe, codes, overrides=ovp, tmp=tmp)
        y_nat = filled_p @ LUMA
    any_full = labels > 0
    y0_log = np.log2(np.maximum(y0, 1e-9))
    out = []
    order = sorted(range(len(regs0)), key=lambda i: -regs0[i]["pixels"])[:MAX_REGIONS]
    for i in order:
        r0, rd, r1 = regs0[i], regs[i], regs1[i]
        if r0["pixels"] < MIN_PIXELS or not r1["overridden"]:
            continue
        cx, cy = int(r0["x"]), int(r0["y"])
        lab = labels[cy, cx]
        if lab == 0:
            continue
        mask = labels == lab
        if int(mask.sum()) != int(r0["pixels"]):
            continue                      # our labelling and the fill's disagree: skip, never guess
        # Exposure is not PeakNet's job (the Source curve's is): put the truth
        # in the baseline's terms on the clean ring around the region, the
        # same ring the fill reads its edge and glow from, so E(a) measures
        # only the peak. Regions without a clean ring are skipped.
        span = int(np.clip(r0["radius_px"], 3, 32))
        near = cv2.dilate(mask.astype(np.uint8), np.ones((2 * span + 1, 2 * span + 1), np.uint8)) > 0
        touch = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ring = near & ~touch & (codes.max(-1) < 0.97) & (y_true > 0)
        if int(ring.sum()) < 8:
            continue
        offset = float(np.median(np.log2(np.maximum(y0[ring], 1e-12)) - np.log2(y_true[ring])))
        yt = y_true[mask] * np.float32(2.0 ** offset)
        dome1 = np.log2(max(y1[cy, cx], 1e-12) / max(y0[cy, cx], 1e-12))
        if dome1 < 0.5:
            continue
        p = np.clip((np.log2(np.maximum(y1[mask], 1e-12)) - np.log2(np.maximum(y0[mask], 1e-12))) / dome1, 0.0, 1.0)
        ecurve = error_curve(y0[mask], p, yt, censor[mask])
        e_def = region_error(y_def[mask], yt, censor[mask])
        a_def = float(np.log2(max(rd["peak_nits"], 1e-9) / max(r0["peak_nits"], 1e-9)))
        feats, patch = region_inputs(codes, y0_log, mask, any_full, r0, a_def)
        k = int(np.argmin(ecurve))
        e_native = region_error(y_nat[mask], yt, censor[mask]) if y_nat is not None and i in predicted else None
        out.append({
            "feats": feats, "patch": patch, "ecurve": ecurve,
            "e_default": e_def, "e_zero": float(ecurve[0]), "e_oracle": float(ecurve[k]),
            "a_oracle": float(A_GRID[k]), "a_default": a_def, "pixels": int(r0["pixels"]),
            "censored_frac": float(censor[mask].mean()), "centre_nits": float(r0["peak_nits"]),
            "ring_offset": offset, "e_native": e_native,
            "meta": {**meta, "region": i, "x": cx, "y": cy, "curve": recipe["curve"],
                     "render_ev": recipe["render_ev"], "codec": recipe["codec"]},
        })
    return out


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------
def _win_to_local(p: str, maps: dict[str, str]) -> Path:
    q = p.replace("\\", "/")
    for win, local in maps.items():
        w = win.replace("\\", "/").rstrip("/")
        if q.lower().startswith(w.lower()):
            return Path(local) / q[len(w):].lstrip("/")
    return Path(q)


VIEWS_PER_PANO = 4
VIEW_SIZE = (960, 540)


def plan_frames(strict_manifest: Path, moves_dir: Path | None, maps: dict[str, str],
                panos_dir: Path | None = None) -> list[dict]:
    rows = [json.loads(line) for line in strict_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    slug_split: dict[str, str] = {}
    for r in rows:
        if r.get("licence_source") == "polyhaven":
            m = re.search(r"PolyHaven::(.+?)_(\d+k)$", str(r["scene_id"]))
            if m and r["split"] != "train":
                slug_split[m.group(1)] = r["split"]
    frames = []
    for r in rows:
        if r.get("licence_source") == "fxtd" and r.get("render_family") != "pipeline":
            frames.append({"kind": "fxtd_camera", "hdr": str(_win_to_local(r["hdr_path"], maps)),
                           "scene": r["scene_id"], "split": r["split"], "id": r["asset_id"],
                           "ceiling_nits": r.get("ceiling_nits")})
    if moves_dir is not None:
        for p in sorted((moves_dir / "hdr").glob("*.png")):
            m = re.match(r"tif_\d+_(.+)_(\d+k)_(\d{5})$", p.stem)
            if not m or int(m.group(3)) not in (4, 13):          # the middle frame of each of the two clips
                continue
            slug = m.group(1)
            frames.append({"kind": "polyhaven_moves", "hdr": str(p), "scene": f"polyhaven::{slug}",
                           "split": slug_split.get(slug, "train"), "id": p.stem, "ceiling_nits": None})
    if panos_dir is not None:
        # v2 (18 Oct 2026): new, independent views straight from the CC0 panoramas, half of them
        # aimed at a bright spot, because v1's 2 views per panorama overfit (train 0.38, val 0.53).
        for p in sorted(panos_dir.glob("*.exr")):
            slug = re.sub(r"_\d+k$", "", p.stem)
            for j in range(VIEWS_PER_PANO):
                frames.append({"kind": "polyhaven_pano", "hdr": str(p), "scene": f"polyhaven::{slug}",
                               "split": slug_split.get(slug, "train"), "id": f"pano_{slug}_v{j}",
                               "ceiling_nits": None, "view": j})
    return frames


def _storage_for(hdr: Path) -> HDRStorage:
    for d in (hdr.parent, hdr.parent.parent):
        s = d / "_ingest_config.json"
        if s.exists():
            return HDRStorage.from_dict(json.loads(s.read_text())["storage"])
    raise SystemExit(f"no _ingest_config.json above {hdr}")


def load_linear(hdr: Path) -> np.ndarray:
    code = cv2.imread(str(hdr), cv2.IMREAD_UNCHANGED)
    if code is None:
        raise IOError(f"cannot read {hdr}")
    return decode_hdr_u16(cv2.cvtColor(code, cv2.COLOR_BGR2RGB), _storage_for(hdr))


_PANO_CACHE: dict[str, np.ndarray] = {}


def pano_view(f: dict) -> np.ndarray:
    """A perspective view (VIEW_SIZE) of an equirectangular panorama, linear light, as
    pipeline/render_hdri_moves.py projects it. Half the views aim near a bright spot."""
    from pipeline.render_hdri_moves import project, ray_grid, rotation
    if f["hdr"] not in _PANO_CACHE:
        import training.prepare_training_data as ptd
        _PANO_CACHE.clear()
        _PANO_CACHE[f["hdr"]] = np.asarray(ptd.read_exr(Path(f["hdr"])), np.float32)[..., :3]
    pano = _PANO_CACHE[f["hdr"]]
    rng = frame_rng(f["id"], 99)
    hfov = float(rng.uniform(45.0, 90.0))
    if rng.random() < 0.5:
        small = cv2.resize(pano, (512, 256), interpolation=cv2.INTER_AREA) @ LUMA
        weights = np.sqrt(np.maximum(small, 0)).ravel()
        k = int(rng.choice(weights.size, p=weights / weights.sum()))
        v, u = divmod(k, small.shape[1])
        yaw = (u + 0.5) / small.shape[1] * 360.0 - 180.0 + rng.uniform(-0.3, 0.3) * hfov
        pitch = 90.0 - (v + 0.5) / small.shape[0] * 180.0 + rng.uniform(-0.2, 0.2) * hfov
    else:
        yaw, pitch = float(rng.uniform(-180, 180)), float(rng.uniform(-20, 30))
    rot = rotation(yaw, float(np.clip(pitch, -80, 80)), float(rng.uniform(-5, 5)))
    return np.ascontiguousarray(project(pano, ray_grid(*VIEW_SIZE, hfov), rot), np.float32)


def frame_rng(frame_id: str, k: int) -> np.random.Generator:
    return np.random.default_rng([SEED, k, int(hashlib.sha256(frame_id.encode()).hexdigest()[:8], 16)])


def recipe_for(rng: np.random.Generator) -> dict:
    """The ACES render the shipped clip fill inverts (its baseline is the ACES inverse at corpus EV 0),
    pushed +0..PUSH_MAX EV so it clips, sometimes through a codec. Other curves are the Source
    curve's problem (CurveNet), not the peak's: under a mismatched curve no dome height can fix
    the fill, and the targets would teach PeakNet that curve error instead."""
    rec = sdr_render.draw_render(rng, ev_jitter=0.0, codec_probability=0.3)
    rec.update({"curve": "aces", "contrast": 1.0, "saturation": 1.0, "oetf": "srgb",
                "render_ev": round(float(rng.uniform(0.0, PUSH_MAX)), 4)})
    return rec


def frame_variant(f: dict, lin: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    """The k-th rendering of a frame: its scale (FXTD only) and its SDR recipe, deterministic per frame."""
    rng = frame_rng(f["id"], k)
    img = lin
    if f["kind"] == "fxtd_camera" and lin.shape[1] > 1280:
        s = float(rng.choice([1.0, 2.0 / 3.0, 0.5]))
        if s < 1:
            img = cv2.resize(lin, (round(lin.shape[1] * s), round(lin.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img, recipe_for(rng)


def shard_frames(frames: list[dict], i: int, n: int) -> list[dict]:
    """Worker i of n. Panorama views shard by panorama, so each worker reads each panorama once;
    everything else keeps v1's order (id order, every n-th), so a v1 run resumes unchanged."""
    other = sorted((f for f in frames if f["kind"] != "polyhaven_pano"), key=lambda f: f["id"])[i::n]
    panos = sorted((f for f in frames if f["kind"] == "polyhaven_pano"), key=lambda f: f["id"])
    slugs = sorted({f["scene"] for f in panos})
    mine = set(slugs[i::n])
    return other + [f for f in panos if f["scene"] in mine]


def save_shard(path: Path, samples: list[dict]) -> None:
    np.savez_compressed(
        path, feats=np.stack([s["feats"] for s in samples]), patch=np.stack([s["patch"] for s in samples]),
        ecurve=np.stack([s["ecurve"] for s in samples]),
        **{k: np.array([s[k] for s in samples], np.float32)
           for k in ("e_default", "e_zero", "e_oracle", "a_oracle", "a_default", "pixels", "censored_frac",
                     "centre_nits", "ring_offset")},
        meta=np.array([json.dumps(s["meta"]) for s in samples]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--strict", required=True, type=Path)
    ap.add_argument("--moves", type=Path, default=None)
    ap.add_argument("--panos", type=Path, default=None, help="equirectangular CC0 panoramas (G:/rudra_v02/panoramas)")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--map", action="append", default=[], metavar="WIN=LOCAL")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--time-budget", type=float, default=0.0)
    ap.add_argument("--tmp", type=Path, default=Path(tempfile.gettempdir()))
    args = ap.parse_args(argv)
    maps = dict(m.split("=", 1) for m in args.map)
    frames = plan_frames(args.strict, args.moves, maps, args.panos)
    i, n = map(int, args.shard.split("/"))
    frames = shard_frames(frames, i, n)
    if args.limit:
        frames = frames[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "shards").mkdir(exist_ok=True)
    cfg = args.out / "_peaknet_config.json"
    if not cfg.exists():
        cfg.write_text(json.dumps({"a_grid": A_GRID.tolist(), "features": FEATURES, "patch": PATCH,
                                   "patch_channels": ["sdr_r", "sdr_g", "sdr_b", "region", "any_full_clip",
                                                      "base_log2_luma_scaled"],
                                   "push_max_ev": PUSH_MAX, "exposures_per_frame": EXPOSURES_PER_FRAME,
                                   "min_pixels": MIN_PIXELS, "max_regions": MAX_REGIONS, "seed": SEED,
                                   "tier": "strict", "tool": "training/peaknet/build_dataset.py"}, indent=2))
    done_log = args.out / f"_frames.{i}of{n}.jsonl"
    done = set()
    for p in args.out.glob("_frames.*.jsonl"):
        done |= {json.loads(line)["id"] for line in p.read_text().splitlines() if line.strip()}
    t0, wrote, total = time.time(), 0, {"train": 0, "val": 0, "test": 0}
    buf: dict[str, list] = {"train": [], "val": [], "test": []}
    pending: list[dict] = []
    stamp = f"{i}of{n}_{int(time.time())}"
    part = 0

    def flush(log) -> None:
        # Samples reach disk BEFORE their frames are marked done, so a run killed
        # mid-way loses only unsaved work and resumes it, never skips it.
        nonlocal part
        for split, samples in buf.items():
            if samples:
                save_shard(args.out / "shards" / f"{split}_{stamp}_{part:03d}.npz", samples)
                total[split] += len(samples)
                samples.clear()
        for rec in pending:
            log.write(json.dumps(rec) + "\n")
        log.flush()
        pending.clear()
        part += 1

    with open(done_log, "a") as log:
        for f in frames:
            if f["id"] in done:
                continue
            if args.time_budget and time.time() - t0 > args.time_budget:
                break
            lin = pano_view(f) if f["kind"] == "polyhaven_pano" else load_linear(Path(f["hdr"]))
            got = 0
            for k in range(EXPOSURES_PER_FRAME):
                img, recipe = frame_variant(f, lin, k)
                meta = {"frame": f["id"], "scene": f["scene"], "split": f["split"], "kind": f["kind"],
                        "exposure": k, "size": [int(img.shape[1]), int(img.shape[0])]}
                samples = frame_regions(args.probe, img, recipe, f["ceiling_nits"], args.tmp, meta)
                buf[f["split"]] += samples
                got += len(samples)
            pending.append({"id": f["id"], "regions": got})
            wrote += 1
            if len(pending) >= 10:
                flush(log)
        flush(log)
    print(f"frames {wrote}, regions {sum(total.values())} "
          f"({', '.join(f'{k} {v}' for k, v in total.items())}) in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
