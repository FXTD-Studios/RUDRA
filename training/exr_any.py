"""Read any scanline/tiled, compressed or uncompressed RGB EXR as float32 (H, W, 3).

rudra.delivery.exr.read_exr is a minimal uncompressed-only reader; the SI-HDR
references and Tears of Steel plates are compressed. This tries the OpenEXR
binding first (exact half/float values), then OpenCV's EXR codec. Values are
returned unchanged: no clamping, no colour conversion, no exposure.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def _openexr(path: Path) -> np.ndarray | None:
    try:
        import OpenEXR
    except ImportError:
        return None
    if not hasattr(OpenEXR, "File"):          # pre-3.3 binding: use OpenCV instead
        return None
    with OpenEXR.File(str(path)) as f:
        channels = f.channels()
        if "RGB" in channels:
            rgb = np.asarray(channels["RGB"].pixels)
        elif "RGBA" in channels:
            rgb = np.asarray(channels["RGBA"].pixels)[..., :3]
        elif all(c in channels for c in "RGB"):
            rgb = np.stack([np.asarray(channels[c].pixels) for c in "RGB"], axis=-1)
        else:
            raise ValueError(f"{path}: no R/G/B channels ({sorted(channels)})")
    return np.ascontiguousarray(rgb, dtype=np.float32)


def read_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    img = _openexr(path)
    if img is None:
        import cv2
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 3 or raw.shape[2] < 3:
            raise ValueError(f"{path}: OpenCV could not decode an RGB EXR")
        img = np.ascontiguousarray(raw[..., 2::-1], dtype=np.float32)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{path}: unexpected shape {img.shape}")
    return img
