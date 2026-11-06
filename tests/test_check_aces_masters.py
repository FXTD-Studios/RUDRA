"""tools/check_aces_masters.py on masters built with the real OCIO view (1.0.3)."""
import json

import numpy as np
import pytest

ocio = pytest.importorskip("PyOpenColorIO")
OpenEXR = pytest.importorskip("OpenEXR")
cv2 = pytest.importorskip("cv2")

from rudra.aces_view import ACES2View
from tools.check_aces_masters import main


def _write_exr(path, rgb, mask=None):
    ch = {"RGB": np.ascontiguousarray(rgb, np.float32)}
    if mask is not None:
        ch["rudra.reconstructed"] = np.ascontiguousarray(mask, np.float32)
    with OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}, ch) as f:
        f.write(str(path))


@pytest.fixture
def frames(tmp_path):
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    codes[:6, :6] = 255                       # a clipped patch
    src = tmp_path / "src"; src.mkdir()
    cv2.imwrite(str(src / "f001.png"), codes[..., ::-1])
    view = ACES2View()
    master = np.asarray(view.inverse(codes.astype(np.float32) / 255.0), np.float32)
    master[:6, :6] *= 1.0                     # a valid master keeps the same view
    a = tmp_path / "a"; a.mkdir()
    _write_exr(a / "f001.exr", master, (codes.max(-1) == 255).astype(np.float32))
    return tmp_path, master, view


def test_a_view_consistent_master_passes(frames):
    root, master, _ = frames
    assert main(["--source", str(root / "src"), "--master", str(root / "a"), "--report", str(root / "r.json")]) == 0
    rep = json.loads((root / "r.json").read_text())
    assert rep["pass"] and rep["frames"][0]["R_round_trip"]["max"] < 1.0


def test_a_master_that_moves_the_picture_fails(frames):
    root, master, _ = frames
    bad = root / "bad"; bad.mkdir()
    _write_exr(bad / "f001.exr", master * 1.25)
    assert main(["--source", str(root / "src"), "--master", str(bad), "--report", str(root / "r.json")]) == 1


def test_backends_within_and_beyond_the_tolerance(frames):
    root, master, view = frames
    near = root / "near"; near.mkdir()
    _write_exr(near / "f001.exr", master * (1 + 1e-5))
    far = root / "far"; far.mkdir()
    _write_exr(far / "f001.exr", master * 1.1)
    args = ["--source", str(root / "src"), "--master", str(root / "a"), "--b-limit", "1.0", "--report", str(root / "r.json")]
    assert main(args + ["--other", str(near)]) == 0
    assert main(args + ["--other", str(far)]) == 1


def test_a_missing_frame_fails(frames):
    root, _, _ = frames
    empty = root / "empty"; empty.mkdir()
    assert main(["--source", str(root / "src"), "--master", str(empty), "--report", str(root / "r.json")]) == 1


def test_non_finite_values_fail(frames):
    root, master, _ = frames
    nan = root / "nan"; nan.mkdir()
    m = master.copy(); m[0, 0, 0] = np.nan
    _write_exr(nan / "f001.exr", m)
    assert main(["--source", str(root / "src"), "--master", str(nan), "--report", str(root / "r.json")]) == 1


def test_a_backend_failure_says_where_and_what(frames, tmp_path):
    root, master, _ = frames
    b = root / "b"; b.mkdir()
    other = master.copy(); other[10, 20] *= 1.2
    _write_exr(b / "f001.exr", other)
    assert main(["--source", str(root / "src"), "--master", str(root / "a"), "--other", str(b), "--b-limit", "0.25",
                 "--report", str(root / "r.json")]) == 1
    w = json.loads((root / "r.json").read_text())["frames"][0]["B_backends"]
    assert (w["worst_pixel"]["x"], w["worst_pixel"]["y"]) == (20, 10) and w["pixels_over_limit"] >= 1


def test_one_pixel_on_the_constraint_threshold_is_not_a_backend_failure(tmp_path):
    """PR #48 (6 Nov 2026): one pixel of a frame half a code apart, the rest identical, is
    the 1-code backtrack taking another side in another runtime: the limit holds at p99.9
    and the max within 1 code. Over --b-max anywhere still fails."""
    rng = np.random.default_rng(11)
    codes = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)   # > 1000 pixels: one is under 0.1 %
    src = tmp_path / "src"; src.mkdir()
    cv2.imwrite(str(src / "f001.png"), codes[..., ::-1])
    view = ACES2View()
    disp = codes.astype(np.float32) / 255.0
    a = tmp_path / "a"; a.mkdir()
    _write_exr(a / "f001.exr", np.asarray(view.inverse(disp), np.float32))
    nudged = disp.copy(); nudged[12, 3] = np.clip(nudged[12, 3] + 0.55 / 255.0, 0, 1)
    one = tmp_path / "one"; one.mkdir()
    _write_exr(one / "f001.exr", np.asarray(view.inverse(nudged), np.float32))
    args = ["--source", str(src), "--master", str(a), "--b-limit", "0.25", "--report", str(tmp_path / "r.json")]
    assert main(args + ["--other", str(one)]) == 0
    b = json.loads((tmp_path / "r.json").read_text())["frames"][0]["B_backends"]
    assert 0.25 < b["max"] <= 1.0 and b["p999"] <= 0.25
    assert main(args + ["--other", str(one), "--b-max", "0.25"]) == 1
