"""Source decoding the engine's way, shared by the region re-render (rudra.genfill.run) and
tools/measure_flicker.py. It lives in the rudra package because packaged RUDRA ships only
python/rudra and python/config: an import from tools/ fails there (ModuleNotFoundError: tools).

A movie goes through ffmpeg (swscale accurate rounding, the stream's matrix and range, 8-bit
sources rounded to whole codes, as native/media/src/aces_frames.cpp); stills through OpenCV.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

FRAME_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".dpx", ".bmp", ".exr"}
# 1.3.2: an EXR frame holds linear light: read as Rec.709 SDR, clipped at 1.0 and encoded with BT.1886 (code =
# L^(1/2.4)), as the app reads it (native/media/src/still.cpp). OpenCV reads EXR only when asked to.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


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
        if p.suffix.lower() == ".exr":
            lin = np.nan_to_num(img[..., :3][..., ::-1].astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
            out.append(np.clip(lin, 0.0, 1.0) ** (1.0 / 2.4))
            continue
        scale = 65535.0 if img.dtype == np.uint16 else 255.0
        out.append(img[..., :3][..., ::-1].astype(np.float64) / scale)
    return out
