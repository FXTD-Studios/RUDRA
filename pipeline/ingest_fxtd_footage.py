"""FXTD Studios' own camera footage -> SDR/HDR training pairs (16 Oct 2026).

The corpus has plenty of rows and few independent scenes: without HdM the
commercially clean train set is ~14k records, 48 % of them one film
(research notes TASK_G_DRIVE_INVENTORY_2026_10_13). FXTD's own camera originals are
the independent, owned scenes docs/CORPUS.md asks for. This turns them into
rendered pairs in the corpus's existing conventions -- nothing new is invented:

  HDR  camera original -> scene-linear in camera primaries (RED Linear EXR as
       Baselight exported it; ARRI LogC3 / LogC4 ProRes decoded with
       training/prepare_training_data.py's own curves) -> camera gamut to
       linear BT.2020 (REDWideGamutRGB, AWG3, AWG4; all D65, no adaptation)
       -> area downsample in LINEAR light to a 1920-px longest side ->
       pipeline.hdr_io ``log2_extended`` -> uint16 PNG. Scene-linear 1.0 =
       203 nits, as for Sparks and Poly Haven: camera 18 % grey lands at 36.5
       nits.
  SDR  pipeline.sdr_render "mix:v1" at tonemap_ev 0 -- curve, exposure,
       contrast, saturation, OETF and codec drawn ONCE PER SHOT (a shot is
       graded once), recorded per row. Same render as corpus_v4c onward, so
       the rows mix with rudra_mix_v10 and corpus_ev_of() still reads 0 EV.

Sensor clip. A camera original is not scene-referred above its sensor clip:
a clipped pixel says ">= clip", exactly like a pixel on a grading ceiling.
Per shot, a channel whose maximum is a PLATEAU (many pixels bit-equal to it
across the sampled frames) is clipped, and the row's ``ceiling_nits`` is set
just under that clip in BT.2020, so sdr2hdr_loss treats those pixels as
censored. A shot with no plateau is scene-referred (ceiling None).

Splits are by shot, never by frame. Projects with >= MIN_SHOTS_FOR_VAL shots
hold out a contiguous block of ~VAL_FRACTION of their shots (contiguous, so
neighbouring setups do not straddle the split); smaller projects train.

Resumable: a frame whose two PNGs exist and whose record is in
``_records.jsonl`` is skipped. ``--finalize`` writes the manifest with
absolute paths under ``--path-root`` (where Windows will READ the corpus).

    python pipeline/ingest_fxtd_footage.py --out G:/datasets/corpora/rudra_fxtd_camera_20261016
    python pipeline/ingest_fxtd_footage.py --out ... --finalize --path-root G:\\datasets\\corpora\\rudra_fxtd_camera_20261016

On a machine where the footage is mounted elsewhere: --root manchale=/mnt/x ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline import sdr_render  # noqa: E402
from pipeline.hdr_io import HDR_IO_VERSION, HDRStorage, encode_hdr_u16, selftest  # noqa: E402
from pipeline.licences import annotate  # noqa: E402
from rudra.delivery.colorspace import npm  # noqa: E402

CORPUS = "rudra_fxtd_camera"
STORAGE = HDRStorage(mode="log2_extended")
DIFFUSE_WHITE = 203.0
TONEMAP_EV = 0.0                 # must match rudra_mix_v10 (corpus_ev_of refuses a mix)
LONG_SIDE = 1920
FRAMES_PER_SHOT = 10
VAL_FRACTION = 0.15
MIN_SHOTS_FOR_VAL = 20
SEED = 20261016
CEILING_MARGIN = 0.97            # censor from just under the clip, so blended clip edges count
PLATEAU_MIN_PIXELS = 64          # bit-equal pixels at a channel max that make it a clip
MIN_HDR_PEAK_NITS = 1.0          # same "no HDR content" rule as fetch_netflix_pairs
MIN_SDR_MEAN = 0.02

# ----------------------------------------------------------------------------
# Sources. Paths are where the footage lives on the FXTD workstation; --root
# overrides them per source id. Camera, curve and gamut are from the files
# themselves (EXR headers: uk.ltd.filmlight.InputColourSpace "RED: Linear /
# REDWideGamutRGB"; MOVs: ARRI clip names, sensor geometry, and the black
# floor at LogC code 0.093 measured on every clip, 16 Oct 2026). The MOV
# containers say bt709; that tag is wrong and is ignored.
# ----------------------------------------------------------------------------
SOURCES: dict[str, dict] = {
    "manchale": {
        "project": "IRAH Manchale song",
        "path": r"F:\CINEMA\IRAH\Manchale\Manchale_Song\(Footage)\Song\FOOTAGE",
        "kind": "exr", "glob": "*/4096x1716/*.exr", "shot_from": "grandparent",
        "encoding": "linear", "gamut": "rwg",
        "camera": "RED EPIC-X Dragon 6K S35, anamorphic; Baselight RED Linear EXR",
    },
    "lab": {
        "project": "IRAH Lab scene",
        "path": r"F:\CINEMA\IRAH\Lab_Scene\Project 4 folder\(Footage)\Footage",
        # Only the camera plates: the PNG/JPG trees beside them are keys and comps.
        "kind": "exr", "glob": "*/4096x1716/*.exr", "shot_from": "grandparent",
        "encoding": "linear", "gamut": "rwg",
        "camera": "RED EPIC-X Dragon 6K S35, anamorphic; Baselight RED Linear EXR",
    },
    "elmatareed": {
        "project": "ELMATAREED",
        "path": r"F:\CINEMA\ELMATAREED\RAW",
        "kind": "mov", "glob": "*.mov", "shot_regex": r"^([A-Z]\d{3}C\d{3})_",
        "encoding": "logc3", "gamut": "awg3",
        "camera": "ARRI ALEXA (LogC3 EI800 assumed, AWG3), ProRes 4444 12-bit UHD",
    },
    "mataha": {
        "project": "Mataha (de-aging)",
        "path": r"F:\work\deaging\Mataha",
        "kind": "mov", "glob": "*.mov", "shot_regex": r"^(A_\d{4}C\d{3})_",
        "encoding": "logc4", "gamut": "awg4",
        "camera": "ARRI ALEXA 35 4.6K open gate (LogC4, AWG4), ProRes 4444/XQ 12-bit",
    },
}

# Camera primaries (x, y for R, G, B; white). All three use D65.
CAMERA_PRIMARIES = {
    "rwg":  ((0.780308, 0.304253), (0.121595, 1.493994), (0.095612, -0.084589)),
    "awg3": ((0.6840, 0.3130), (0.2210, 0.8480), (0.0861, -0.1020)),
    "awg4": ((0.7347, 0.2653), (0.1424, 0.8576), (0.0991, -0.0308)),
}
D65 = (0.3127, 0.3290)


def camera_npm(gamut: str) -> np.ndarray:
    """RGB -> XYZ for a camera gamut (SMPTE RP 177)."""
    (rx, ry), (gx, gy), (bx, by) = CAMERA_PRIMARIES[gamut]
    xyz = lambda x, y: np.array([x / y, 1.0, (1.0 - x - y) / y])  # noqa: E731
    p = np.stack([xyz(rx, ry), xyz(gx, gy), xyz(bx, by)], axis=1)
    s = np.linalg.solve(p, xyz(*D65))
    return p * s


def camera_to_rec2020(gamut: str) -> np.ndarray:
    return (np.linalg.inv(npm("rec2020")) @ camera_npm(gamut)).astype(np.float32)


# ----------------------------------------------------------------------------
# Decode
# ----------------------------------------------------------------------------
def _ptd():
    import training.prepare_training_data as ptd
    return ptd


def decode_log(code: np.ndarray, encoding: str) -> np.ndarray:
    ptd = _ptd()
    if encoding == "logc3":
        return ptd.eotf_logc3(code)
    if encoding == "logc4":
        return ptd.eotf_logc4(code)
    raise ValueError(f"not a log encoding: {encoding}")


def probe_mov(path: Path) -> tuple[int, float, int, int]:
    """(frames, fps, width, height). The container's frame count, which ProRes
    MOVs carry; counting packets instead reads the whole file (18 GB for Mataha)."""
    def run(extra):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", *extra,
                              "-show_entries", "stream=nb_frames,nb_read_packets,r_frame_rate,width,height",
                              "-of", "json", str(path)], capture_output=True, text=True, check=True)
        return json.loads(out.stdout)["streams"][0]
    s = run([])
    n = s.get("nb_frames")
    if not n or n == "N/A":
        n = run(["-count_packets"])["nb_read_packets"]
    num, den = s["r_frame_rate"].split("/")
    return int(n), float(num) / float(den), int(s["width"]), int(s["height"])


def read_mov_frame(path: Path, index: int, fps: float, width: int, height: int) -> np.ndarray:
    """One frame as full-range code values in [0, 1]. ProRes is intra-only, so
    an input seek lands on the exact frame."""
    t = (index + 0.5) / fps
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.6f}", "-i", str(path),
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
                         capture_output=True, check=True).stdout
    if len(raw) != width * height * 6:
        raise IOError(f"{path.name} frame {index}: got {len(raw)} bytes")
    return np.frombuffer(raw, np.uint16).reshape(height, width, 3).astype(np.float32) / 65535.0


def camera_linear(src: dict, item: dict) -> np.ndarray:
    """(H, W, 3) scene-linear in CAMERA primaries, full resolution."""
    if src["kind"] == "exr":
        img = _ptd().read_exr(Path(item["path"]))
    else:
        code = read_mov_frame(Path(item["path"]), item["frame"], item["fps"], item["width"], item["height"])
        img = decode_log(code, src["encoding"])
    img = np.asarray(img, dtype=np.float32)[..., :3]
    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return img


def downsample(img: np.ndarray, long_side: int = LONG_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1.0:
        return img
    size = (max(2, round(w * s / 2) * 2), max(2, round(h * s / 2) * 2))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------------
def _frame_number(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else 0


def even_picks(n: int, k: int) -> list[int]:
    """k indices spread over 0..n-1, offset half a stride from both ends."""
    if n <= 0:
        return []
    k = min(k, n)
    return sorted({int((i + 0.5) * n / k) for i in range(k)})


def plan_source(sid: str, src: dict, root: Path, frames_per_shot: int) -> dict[str, list[dict]]:
    """{shot: [frame items]} for one source."""
    shots: dict[str, list[dict]] = {}
    if src["kind"] == "exr":
        seqs: dict[str, list[Path]] = defaultdict(list)
        for p in root.glob(src["glob"]):
            shot = p.parent.parent.name if src["shot_from"] == "grandparent" else p.parent.name
            seqs[shot].append(p)
        for shot, files in seqs.items():
            files.sort(key=_frame_number)
            shots[shot] = [{"path": str(files[i]), "frame": _frame_number(files[i]), "shot_frames": len(files)}
                           for i in even_picks(len(files), frames_per_shot)]
    else:
        clips: dict[str, list[Path]] = defaultdict(list)
        for p in sorted(root.glob(src["glob"])):
            m = re.match(src["shot_regex"], p.name)
            clips[m.group(1) if m else p.stem].append(p)
        for shot, files in clips.items():
            probed = [(p, *probe_mov(p)) for p in files]
            total = sum(n for _, n, *_ in probed)
            items = []
            for file_index, (p, n, fps, w, h) in enumerate(probed):
                k = max(1, round(frames_per_shot * n / total))
                items += [{"path": str(p), "file_index": file_index, "frame": i, "fps": fps, "width": w, "height": h,
                           "shot_frames": total} for i in even_picks(n, k)]
            shots[shot] = items
    return shots


def assign_splits(shots: list[str], seed: int = SEED, fraction: float = VAL_FRACTION,
                  min_shots: int = MIN_SHOTS_FOR_VAL) -> dict[str, str]:
    """A contiguous block of ~fraction of the (sorted) shots is val; the rest train."""
    order = sorted(shots, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])
    split = {s: "train" for s in order}
    if len(order) < min_shots:
        return split
    n_val = max(1, round(len(order) * fraction))
    start = np.random.default_rng(seed).integers(0, len(order) - n_val + 1)
    for s in order[start:start + n_val]:
        split[s] = "val"
    return split


# ----------------------------------------------------------------------------
# Clip and render
# ----------------------------------------------------------------------------
def frame_peak(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel maximum of a FULL-resolution camera frame and how many pixels sit on it."""
    flat = img.reshape(-1, 3)
    top = flat.max(0)
    return top.astype(np.float32), (flat == top[None, :]).sum(0).astype(np.int64)


def channel_plateaus(peaks: list[tuple[np.ndarray, np.ndarray]], min_pixels: int = PLATEAU_MIN_PIXELS
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(per-channel shot maximum, bool clipped). A channel is clipped when at least
    ``min_pixels`` pixels across the shot are bit-equal to its maximum: natural
    peaks do not repeat exactly, a sensor clip does. Only frames whose own maximum
    IS the shot maximum can hold such pixels, so per-frame counts suffice."""
    tops = np.stack([t for t, _ in peaks])
    top = tops.max(0)
    counts = np.zeros(3, dtype=np.int64)
    for t, n in peaks:
        counts += np.where(t == top, n, 0)
    return top.astype(np.float32), (counts >= min_pixels) & (top > 0.25)


def shot_ceiling_nits(top: np.ndarray, clipped: np.ndarray, matrix: np.ndarray) -> float | None:
    """Ceiling in nits for a clipped shot, in BT.2020 after the gamut matrix:
    the smallest clipped channel of the camera's clip white, less a margin."""
    if not clipped.any():
        return None
    white = matrix @ top
    return float(CEILING_MARGIN * white[clipped].min() * DIFFUSE_WHITE)


def shot_recipe(scene_id: str, seed: int = SEED) -> dict:
    rng = np.random.default_rng([seed, int(hashlib.sha256(scene_id.encode()).hexdigest()[:8], 16)])
    return sdr_render.draw_render(rng)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"failed to encode {path}")
    tmp = path.with_suffix(".tmp.png")
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    os.replace(tmp, path)


def process_shot(sid: str, src: dict, shot: str, items: list[dict], split: str, out: Path,
                 done: set[str]) -> list[dict]:
    scene_id = f"fxtd_{sid}_{re.sub(r'[^A-Za-z0-9]+', '_', shot).strip('_').lower()}"
    asset = lambda it: f"{scene_id}_" + (  # noqa: E731
        f"f{it['frame']:08d}" if src["kind"] == "exr" else f"s{it['file_index']}_f{it['frame']:06d}")
    todo = [it for it in items if asset(it) not in done]
    if not todo:
        return []
    matrix = camera_to_rec2020(src["gamut"])
    # Every pick is decoded (the clip test needs the whole shot); only the
    # downsampled frame is kept, so a 4K shot costs ~10 x 18 MB, not 10 x 84 MB.
    smalls, peaks, sizes = [], [], []
    for it in items:
        cam = camera_linear(src, it)
        peaks.append(frame_peak(cam))
        sizes.append(cam.shape[:2])
        smalls.append(downsample(cam))
        del cam
    top, clipped = channel_plateaus(peaks)
    ceiling = shot_ceiling_nits(top, clipped, matrix)
    recipe = shot_recipe(scene_id)
    records = []
    for it, small, (src_h, src_w) in zip(items, smalls, sizes):
        aid = asset(it)
        if aid in done:
            continue
        lin = np.maximum(small @ matrix.T, 0.0).astype(np.float32)
        hdr, stats = encode_hdr_u16(lin, STORAGE)
        sdr = sdr_render.apply_render(lin, recipe, TONEMAP_EV)
        lum = lin @ np.array([0.2627, 0.6780, 0.0593], dtype=np.float32) * DIFFUSE_WHITE
        sdr_mean = float(sdr.mean() / 255.0)
        dark = stats["peak_nits"] < MIN_HDR_PEAK_NITS or sdr_mean < MIN_SDR_MEAN
        rec = {
            "asset_id": aid, "scene_id": scene_id, "split": split,
            "project": src["project"], "source_id": sid, "shot": shot,
            "frame_index": int(it["frame"]), "shot_frames": int(it["shot_frames"]),
            "sdr_path": f"sdr/{aid}.png", "hdr_path": f"hdr/{aid}.png",
            "width": int(lin.shape[1]), "height": int(lin.shape[0]),
            "source_width": int(src_w), "source_height": int(src_h),
            "source_path": str(PureWindowsPath(src["path"]) / Path(it["path"]).relative_to(src["_root"])),
            "camera": src["camera"], "source_encoding": src["encoding"], "source_gamut": src["gamut"],
            "sdr_kind": "rendered", "sdr_encoding": "srgb", "hdr_encoding": STORAGE.mode,
            "tonemap_ev": TONEMAP_EV, "sdr_render": recipe,
            "sdr_curve": recipe["curve"], "source_curve": recipe["curve"],
            "render_ev": recipe["render_ev"], "sdr_codec": recipe["codec"],
            "sensor_clip_camera": [round(float(v), 5) for v in top],
            "sensor_clipped_channels": [bool(c) for c in clipped],
            "ceiling_nits": ceiling,
            "peak_at_ceiling": bool(ceiling is not None and stats["peak_nits"] >= ceiling),
            "peak_nits": stats["peak_nits"], "clipped_fraction": stats["clipped_fraction"],
            "sdr_clipped_fraction": float((sdr.max(-1) >= 255).mean()),
            "sdr_mean_code": sdr_mean,
            "hdr_p50_nits": float(np.percentile(lum, 50)), "hdr_p99_nits": float(np.percentile(lum, 99)),
            "dr_stops": float(np.log2(max(np.percentile(lum, 99.9), 1e-4) / max(np.percentile(lum, 1), 1e-4))),
            "is_video": False,
            "excluded": "near_black" if dark else None,
        }
        if not dark:
            _write_png(out / rec["hdr_path"], hdr)
            _write_png(out / rec["sdr_path"], sdr)
        records.append(rec)
    return records


# ----------------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------------
def write_sentinel(out: Path) -> None:
    payload = {
        "corpus": CORPUS, "created": time.strftime("%Y-%m-%d"),
        "hdr_io_version": HDR_IO_VERSION, "storage": STORAGE.as_dict(),
        "tonemap": sdr_render.RENDER_VERSION, "tonemap_ev": TONEMAP_EV, "sdr_encoding": "srgb",
        "ev_jitter": 1.5, "codec_probability": 0.5, "render_unit": "shot",
        "target_size": f"full frame, longest side {LONG_SIDE}, area-downsampled in linear light",
        "hdr_convention": "camera original -> scene-linear camera primaries -> BT.2020 (D65) -> "
                          "1.0 = 203 nits -> log2_extended",
        "sensor_clip": f"per-shot channel plateaus; ceiling_nits = {CEILING_MARGIN} x clip white in BT.2020",
        "licence": "FXTD Studios own footage (client productions; FXTD confirmed training rights 16 Oct 2026)",
        "sources": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in SOURCES.items()},
        "frames_per_shot": FRAMES_PER_SHOT, "val_fraction": VAL_FRACTION, "seed": SEED,
        "tool": "pipeline/ingest_fxtd_footage.py",
    }
    path = out / "_ingest_config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        for key in ("hdr_io_version", "storage", "tonemap", "tonemap_ev", "sdr_encoding"):
            if old.get(key) != payload[key]:
                raise SystemExit(f"{path}: {key} differs ({old.get(key)!r} vs {payload[key]!r}); use a fresh --out")
        return
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_records(out: Path) -> dict[str, dict]:
    """Every ``_records*.jsonl`` (one per source, so sources can run in parallel)."""
    recs = {}
    for path in sorted(out.glob("_records*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["asset_id"]] = r
    return recs


def _absolute(root: str, rel: str) -> str:
    windows = "\\" in root or (len(root) > 1 and root[1] == ":")
    sep = "\\" if windows else "/"
    return root.rstrip("\\/") + sep + rel.replace("/", sep)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finalize(out: Path, root: str | None = None) -> dict:
    root = root or str(out.resolve())
    kept = [r for r in load_records(out).values() if not r["excluded"]]
    kept.sort(key=lambda r: (r["split"], r["asset_id"]))
    for r in kept:
        for k in ("sdr_path", "hdr_path"):
            if not (out / r[k]).exists():
                raise SystemExit(f"{r['asset_id']}: {r[k]} missing on disk")
            r[k] = _absolute(root, r[k])
    annotate(kept)
    bad = [r["asset_id"] for r in kept if r.get("licence_source") != "fxtd"]
    if bad:
        raise SystemExit(f"licence classifier did not tag {len(bad)} rows as fxtd, e.g. {bad[:3]}")
    manifest = out / "sdr_hdr_manifest.jsonl"
    with open(manifest, "w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    scenes = defaultdict(set)
    for r in kept:
        scenes[r["scene_id"]].add(r["split"])
    report = {
        "manifest": str(manifest), "manifest_sha256": _sha256(manifest), "path_root": root,
        "records": len(kept), "scenes": len(scenes),
        "scenes_straddling_splits": sorted(s for s, v in scenes.items() if len(v) > 1),
        "by_split_source": {f"{s}/{p}": n for (s, p), n in sorted(Counter((r["split"], r["source_id"]) for r in kept).items())},
        "scenes_by_split": dict(Counter(next(iter(v)) for v in scenes.values())),
        "excluded_near_black": sum(1 for r in load_records(out).values() if r["excluded"]),
        "sdr_curves": dict(Counter(r["sdr_curve"] for r in kept)),
        "sdr_codecs": dict(Counter(r["sdr_codec"] for r in kept)),
        "rows_with_ceiling": sum(1 for r in kept if r["ceiling_nits"]),
        "rows_peak_at_ceiling": sum(1 for r in kept if r["peak_at_ceiling"]),
        "hdr_peak_nits_p50": float(np.median([r["peak_nits"] for r in kept])) if kept else None,
        "dr_stops_p50": float(np.median([r["dr_stops"] for r in kept])) if kept else None,
        "sdr_clipped_rows": sum(1 for r in kept if r["sdr_clipped_fraction"] > 1e-4),
        "commercial_ok": dict(Counter(str(r.get("commercial_ok")) for r in kept)),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--root", action="append", default=[], metavar="ID=PATH",
                    help="where a source is mounted on this machine (default: its Windows path)")
    ap.add_argument("--frames-per-shot", type=int, default=FRAMES_PER_SHOT)
    ap.add_argument("--max-shots", type=int, default=0, help="per source; 0 = all (smoke runs)")
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="stop cleanly after this many seconds; re-run to resume")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--path-root", default=None, help="absolute root written into the manifest")
    args = ap.parse_args(argv)

    if args.finalize:
        print(json.dumps(finalize(args.out, args.path_root), indent=2))
        return 0

    selftest(STORAGE)
    write_sentinel(args.out)
    roots = dict(r.split("=", 1) for r in args.root)
    done = set(load_records(args.out))
    t0 = time.time()
    for sid in args.sources:
        with open(args.out / f"_records.{sid}.jsonl", "a", encoding="utf-8", newline="\n") as log:
            src = dict(SOURCES[sid])
            src["_root"] = Path(roots.get(sid, src["path"]))
            if not src["_root"].is_dir():
                print(f"skip {sid}: {src['_root']} not found", file=sys.stderr)
                continue
            shots = plan_source(sid, src, src["_root"], args.frames_per_shot)
            splits = assign_splits(list(shots))
            names = sorted(shots)[: args.max_shots or None]
            print(f"{sid}: {len(shots)} shots, {sum(len(v) for v in shots.values())} picks, "
                  f"val {sum(1 for s in shots if splits[s] == 'val')} shots", flush=True)
            for shot in names:
                if args.time_budget and time.time() - t0 > args.time_budget:
                    print("time budget reached; re-run to resume", flush=True)
                    return 3
                for rec in process_shot(sid, src, shot, shots[shot], splits[shot], args.out, done):
                    log.write(json.dumps(rec) + "\n")
                    done.add(rec["asset_id"])
                log.flush()
                print(f"  {shot}: {len(done)} rows, {time.time() - t0:.0f}s", flush=True)
    print("all sources done; run --finalize", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
