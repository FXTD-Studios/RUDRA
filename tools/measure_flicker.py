"""Flicker of RUDRA's masters on any real clip, against the T contract (<= 0.02 EV).

The same measure as the Hunyuan48 check (tools/evaluate_aces_temporal.py
temporal_change, used by evaluate_aces_feather_video.py): each frame as a
36 x 64 grid of AP1-luminance cell means (adaptive average pooling); for each
pair of neighbouring frames, the master's change in stops minus the plain ACES
inverse's change, cell by cell; the mean absolute value is that transition's
excess flicker. A clip passes when every transition is <= 0.02 EV.

Inputs per clip: the source (a movie, or a folder of frames) and the folder of
EXR masters a delivery wrote from it (ACEScg or ACES2065-1, any names that sort
in frame order). The baseline is rebuilt here from the source: decoded as the
engine decodes it (native/media/src/aces_frames.cpp: swscale accurate rounding,
the stream's matrix and range, 8-bit sources rounded to whole codes), then the
ACES 2.0 SDR 100-nit inverse (rudra.aces_view). Masters with alpha are
premultiplied; the baseline is premultiplied by the same alpha, so an alpha
edge cannot read as flicker.

    python tools/measure_flicker.py --clip D:/clips/shot010.mov --masters D:/renders/shot010 --out flicker_shot010.json
    python tools/measure_flicker.py --list clips.json --out flicker_all.json
        clips.json: [{"name": "shot010", "clip": "...", "masters": "...", "first": 0}, ...]

Exit 0 when every clip passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LUMA_AP1 = np.array([.2722287168, .6740817658, .0536895174], np.float64)
# ACES2065-1 (AP0) to ACEScg (AP1), the Academy's matrix.
AP0_TO_AP1 = np.array([[1.4514393161, -0.2365107469, -0.2149285693],
                       [-0.0765537734, 1.1762296998, -0.0996759264],
                       [0.0083161484, -0.0060324498, 0.9977163014]], np.float64)
GRID = (36, 64)
LIMIT_EV = 0.02
FRAME_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".dpx", ".bmp"}


def adaptive_pool(y: np.ndarray, out=GRID) -> np.ndarray:
    """torch.nn.functional.adaptive_avg_pool2d for one plane, in numpy."""
    h, w = y.shape
    oh, ow = out
    res = np.empty(out, np.float64)
    for i in range(oh):
        y0, y1 = (i * h) // oh, -((-(i + 1) * h) // oh)
        for j in range(ow):
            x0, x1 = (j * w) // ow, -((-(j + 1) * w) // ow)
            res[i, j] = y[y0:y1, x0:x1].mean()
    return res


def cell_luminance(rgb_ap1: np.ndarray) -> np.ndarray:
    return adaptive_pool(np.asarray(rgb_ap1, np.float64) @ LUMA_AP1)


def temporal_change(previous, current, baseline_previous, baseline_current) -> dict:
    """tools/evaluate_aces_temporal.py temporal_change, the same arithmetic."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (previous, current, baseline_previous, baseline_current)]
    valid = np.logical_and.reduce([x > 0 for x in arrays])
    if not valid.any():
        return {"valid_cells": 0, "excluded_cells": int(valid.size), "mean_abs_excess_ev": None, "max_abs_excess_ev": None}
    p, c, bp, bc = [np.log2(x[valid]) for x in arrays]
    delta = (c - p) - (bc - bp)
    return {"valid_cells": int(valid.sum()), "excluded_cells": int((~valid).sum()),
            "mean_abs_excess_ev": float(np.abs(delta).mean()), "max_abs_excess_ev": float(np.abs(delta).max())}


def read_master(path: Path):
    """-> (rgb in AP1, alpha or None). ACES2065-1 masters (acesImageContainerFlag
    or AP0 chromaticities) are converted to AP1."""
    import OpenEXR
    with OpenEXR.File(str(path)) as f:
        ch = f.channels()
        # Read everything needed from the header now: it is empty once the file closes.
        hdr = dict(f.header())
        if "RGB" in ch:
            rgb = np.asarray(ch["RGB"].pixels, np.float64)
            alpha = None
        elif "RGBA" in ch:
            px = np.asarray(ch["RGBA"].pixels, np.float64)
            rgb, alpha = px[..., :3], px[..., 3]
        else:
            rgb = np.stack([np.asarray(ch[c].pixels, np.float64) for c in ("R", "G", "B")], -1)
            alpha = np.asarray(ch["A"].pixels, np.float64) if "A" in ch else None
    ap0 = bool(hdr.get("acesImageContainerFlag")) or "ACES2065" in str(hdr.get("rudra:colourSpace", ""))
    chroma = hdr.get("chromaticities")
    if not ap0 and chroma is not None:
        try:
            ap0 = abs(float(chroma[0][0]) - 0.7347) < 1e-3   # AP0 red x
        except (TypeError, IndexError, ValueError):
            pass
    if ap0:
        rgb = rgb @ AP0_TO_AP1.T
    return rgb, alpha


def _probe(movie: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,pix_fmt,color_range", "-of", "json", str(movie)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"][0]


def decode_movie(movie: Path, first: int, count: int) -> list[np.ndarray]:
    """Frames first..first+count-1 as display codes in [0, 1], as the engine reads them."""
    s = _probe(movie)
    w, h = int(s["width"]), int(s["height"])
    rng = "pc" if s.get("color_range") == "pc" else "tv"
    eight = not any(k in s.get("pix_fmt", "") for k in ("10", "12", "16"))
    vf = (f"select=gte(n\\,{first}),scale=in_color_matrix=bt709:in_range={rng}:out_range=pc:"
          "flags=accurate_rnd+full_chroma_int,format=rgb48le")
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-noautorotate", "-i", str(movie), "-map", "0:v:0",
                        "-vf", vf, "-fps_mode", "passthrough", "-frames:v", str(count), "-f", "rawvideo", "-"],
                       capture_output=True, check=True)
    raw = np.frombuffer(r.stdout, np.uint16).reshape(-1, h, w, 3)
    out = []
    for fr in raw:
        codes = fr.astype(np.float64) / 65535.0
        out.append(np.rint(codes * 255.0) / 255.0 if eight else codes)
    return out


def decode_frames(folder: Path, first: int, count: int) -> list[np.ndarray]:
    import cv2
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in FRAME_EXT)[first:first + count]
    out = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"cannot read {p}")
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, 2)
        scale = 65535.0 if img.dtype == np.uint16 else 255.0
        out.append(img[..., :3][..., ::-1].astype(np.float64) / scale)
    return out


def measure_clip(name: str, clip: Path, masters: Path, first: int = 0, view=None) -> dict:
    from rudra.aces_view import ACES2View
    view = view or ACES2View(threads=4)
    exrs = sorted(masters.glob("*.exr"))
    if len(exrs) < 2:
        return {"name": name, "error": f"fewer than 2 EXR masters in {masters}"}
    src = decode_frames(clip, first, len(exrs)) if clip.is_dir() else decode_movie(clip, first, len(exrs))
    if len(src) != len(exrs):
        return {"name": name, "error": f"{len(exrs)} masters but {len(src)} source frames from frame {first}"}
    rows, prev = [], None
    for i, (codes, exr) in enumerate(zip(src, exrs)):
        rgb, alpha = read_master(exr)
        if rgb.shape[:2] != codes.shape[:2]:
            return {"name": name, "error": f"{exr.name}: {rgb.shape[1]}x{rgb.shape[0]}, source {codes.shape[1]}x{codes.shape[0]}"}
        base = np.asarray(view.inverse(codes.astype(np.float32)), np.float64)
        if alpha is not None:
            base = base * alpha[..., None]   # as the master is premultiplied
        cur = (cell_luminance(rgb), cell_luminance(base))
        if prev is not None:
            t = temporal_change(prev[0], cur[0], prev[1], cur[1])
            t["from"], t["to"] = exrs[i - 1].name, exr.name
            rows.append(t)
        prev = cur
    vals = [r["mean_abs_excess_ev"] for r in rows if r["mean_abs_excess_ev"] is not None]
    worst = max(rows, key=lambda r: -1 if r["mean_abs_excess_ev"] is None else r["mean_abs_excess_ev"])
    return {"name": name, "clip": str(clip), "masters": str(masters), "frames": len(exrs),
            "transitions": rows, "max_transition_mean_ev": max(vals) if vals else None,
            "mean_transition_ev": float(np.mean(vals)) if vals else None,
            "worst": {"from": worst["from"], "to": worst["to"], "ev": worst["mean_abs_excess_ev"]},
            "limit_ev": LIMIT_EV, "pass": bool(vals) and max(vals) <= LIMIT_EV}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path)
    ap.add_argument("--masters", type=Path)
    ap.add_argument("--first", type=int, default=0, help="the source frame the first master was made from")
    ap.add_argument("--list", type=Path, help="JSON list of {name, clip, masters, first}")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.list:
        jobs = json.loads(a.list.read_text(encoding="utf-8"))
    elif a.clip and a.masters:
        jobs = [{"name": a.clip.stem, "clip": str(a.clip), "masters": str(a.masters), "first": a.first}]
    else:
        ap.error("give --clip and --masters, or --list")
    if any(not Path(j["clip"]).is_dir() for j in jobs) and not shutil.which("ffmpeg"):
        print("FAILED: ffmpeg and ffprobe are needed to decode movies")
        return 1
    from rudra.aces_view import ACES2View
    view = ACES2View(threads=4)
    clips = []
    for j in jobs:
        r = measure_clip(j.get("name", Path(j["clip"]).stem), Path(j["clip"]), Path(j["masters"]), int(j.get("first", 0)), view)
        clips.append(r)
        if "error" in r:
            print(f"{r['name']:30s} ERROR {r['error']}")
        else:
            print(f"{r['name']:30s} {r['frames']:5d} frames  max {r['max_transition_mean_ev']:.4f} EV  "
                  f"mean {r['mean_transition_ev']:.4f}  worst {r['worst']['from']} -> {r['worst']['to']}  "
                  f"{'pass' if r['pass'] else 'FAIL'}")
    ok = all(c.get("pass") for c in clips)
    report = {"schema": "rudra_flicker_v1", "measure": "mean |master dlog2 - plain inverse dlog2| over 36x64 AP1-luminance cells, per transition",
              "limit_ev": LIMIT_EV, "clips": clips, "passed": sum(bool(c.get("pass")) for c in clips),
              "total": len(clips), "all_pass": ok,
              "enough_clips": len(clips) >= 10}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{report['passed']} of {report['total']} clips pass (<= {LIMIT_EV} EV every transition)"
          + ("" if report["enough_clips"] else "; the roadmap asks for at least 10 clips") + f"  -> {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
