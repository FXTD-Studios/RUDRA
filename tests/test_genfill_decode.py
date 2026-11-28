"""rudra/genfill/decode.py: the source frames AI fill reads (1.3.2, 27 Nov 2026: EXR sequences, user feedback)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from rudra.genfill import decode

cv2 = pytest.importorskip("cv2")


def test_an_exr_frame_is_linear_rec709_read_as_bt1886_codes(tmp_path):
    """As the app reads it (native/media/src/still.cpp): clipped at 1.0, code = L^(1/2.4)."""
    assert os.environ.get("OPENCV_IO_ENABLE_OPENEXR") == "1"
    lin = np.array([[[0.0, 0.0, 0.0], [0.18, 0.18, 0.18]], [[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]]], np.float32)
    try:
        wrote = cv2.imwrite(str(tmp_path / "f.0001.exr"), lin)
    except cv2.error:
        wrote = False
    if not wrote:
        pytest.skip("this OpenCV reads and writes no EXR")
    cv2.imwrite(str(tmp_path / "f.0002.png"), np.full((2, 2, 3), 128, np.uint8))
    frames = decode.decode_frames(tmp_path, 0, 2)
    assert len(frames) == 2
    exr = frames[0]
    assert exr[0, 0, 0] == 0.0
    assert abs(exr[0, 1, 1] - 0.18 ** (1 / 2.4)) < 1e-4
    assert exr[1, 0, 2] == 1.0 and exr[1, 1, 0] == 1.0       # white, and a highlight clipped as the SDR clipped it
    assert abs(frames[1][0, 0, 0] - 128 / 255) < 1e-9          # a PNG beside it as before
