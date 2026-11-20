"""1.2.7 (multi-role review of 1.2.6, 20 Nov 2026): the Python side of the fixes.

- rudra/delivery/exr.py reads the ZIP and ZIPS masters the native app writes since 1.2.6.
- AI fill v4.3 can leave thin and small bright parts to the master (--rerender-keep-detail,
  composite.detail_keep_weight); the native "Where it fills" draws the same rule.
"""

import numpy as np
import pytest

from rudra.delivery.exr import read_exr
from rudra.genfill import composite
from rudra.genfill.run import Rerender

OpenEXR = pytest.importorskip("OpenEXR")


@pytest.mark.parametrize("compression", ["ZIP_COMPRESSION", "ZIPS_COMPRESSION", "NO_COMPRESSION"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_zip_masters_are_read_back_exactly(tmp_path, compression, dtype):
    rng = np.random.default_rng(127)
    rgb = (rng.random((37, 53, 3)) * 6.0).astype(dtype)   # 37 lines: a last ZIP block of 5
    rgb[0, 0] = 0.0                                          # a flat start, a run the predictor sees
    path = tmp_path / f"m_{compression}.exr"
    with OpenEXR.File({"compression": getattr(OpenEXR, compression), "type": OpenEXR.scanlineimage}, {"RGB": rgb}) as f:
        f.write(str(path))
    img, attrs = read_exr(path)
    assert img.shape == rgb.shape
    np.testing.assert_array_equal(img, rgb.astype(np.float32))
    assert attrs["channel_order"] == ["R", "G", "B"]


def test_other_compressions_are_still_refused(tmp_path):
    rgb = np.ones((8, 8, 3), np.float32)
    path = tmp_path / "piz.exr"
    with OpenEXR.File({"compression": OpenEXR.PIZ_COMPRESSION, "type": OpenEXR.scanlineimage}, {"RGB": rgb}) as f:
        f.write(str(path))
    with pytest.raises(ValueError, match="compression"):
        read_exr(path)


def _frame(h=1080, w=1920):
    return np.full((h, w, 3), 13, np.uint8)


def test_keep_detail_leaves_a_grating_and_a_lamp_and_redraws_the_door():
    codes = _frame()
    codes[100:800, 600:1300] = 255                      # the door
    for x in range(600, 1300, 6):
        codes[900:1000, x:x + 2] = 255                  # a floor grating: thin bars
    codes[100:130, 100:120] = 255                       # a practical lamp, 600 px (< 0.05 % of the frame)
    w = composite.detail_keep_weight(codes, 240)
    assert w.dtype == np.float32 and w.shape == codes.shape[:2]
    assert w[450, 950] == pytest.approx(1.0)            # the door is redrawn
    assert w[950, 600] < 0.5                            # the grating keeps the master
    assert w[115, 110] == 0.0                           # the lamp too
    assert w[50, 50] == 0.0                             # outside the region: nothing
    assert float((w[100:800, 600:1300] >= 0.5).mean()) > 0.97


def test_keep_detail_is_off_unless_asked():
    r = Rerender(source="x.mp4")
    assert r.keep_detail is False


def test_the_run_takes_the_switch():
    from rudra.genfill import run
    src = open(run.__file__, encoding="utf-8").read()
    assert "--rerender-keep-detail" in src
    assert "detail_keep_weight(codes[i], ms.tau_high)" in src
