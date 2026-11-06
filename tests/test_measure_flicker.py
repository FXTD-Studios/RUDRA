"""tools/measure_flicker.py: the T contract on any clip."""
import json

import numpy as np
import pytest

ocio = pytest.importorskip("PyOpenColorIO")
OpenEXR = pytest.importorskip("OpenEXR")
cv2 = pytest.importorskip("cv2")

from rudra.aces_view import ACES2View
from tools.measure_flicker import AP0_TO_AP1, adaptive_pool, main, temporal_change

H, W, N = 36, 64, 6


def _write_exr(path, rgb, alpha=None, ap0=False):
    ch = {"RGB": np.ascontiguousarray(rgb, np.float32)} if alpha is None else \
         {"RGBA": np.ascontiguousarray(np.concatenate([rgb, alpha[..., None]], -1), np.float32)}
    hdr = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    if ap0:
        hdr["acesImageContainerFlag"] = 1
    with OpenEXR.File(hdr, ch) as f:
        f.write(str(path))


@pytest.fixture
def clip(tmp_path):
    """N 8-bit frames of a slow pan with a clipped window, and the plain inverse of each."""
    view = ACES2View()
    src = tmp_path / "frames"
    src.mkdir()
    bases = []
    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(N):
        codes = np.stack([0.25 + 0.4 * (xx + i) / (W + N), 0.3 + 0.3 * yy / H, np.full((H, W), 0.35)], -1)
        codes[8:20, 20 + i:34 + i] = 1.0   # a moving clipped window
        codes = np.rint(codes * 255) / 255
        cv2.imwrite(str(src / f"f{i:04d}.png"), np.rint(codes[..., ::-1] * 255).astype(np.uint8))
        bases.append(np.asarray(view.inverse(codes.astype(np.float32)), np.float64))
    return tmp_path, src, bases


def _run(tmp_path, src, masters, name):
    out = tmp_path / f"{name}.json"
    rc = main(["--clip", str(src), "--masters", str(masters), "--out", str(out)])
    return rc, json.loads(out.read_text())["clips"][0]


def test_steady_reconstruction_passes(tmp_path):
    # A static shot with a static rebuild: no excess change at all. (A rebuilt area
    # that MOVES across the grid changes its cells' mix, which the T measure counts;
    # the Hunyuan48 check behaves the same.)
    view = ACES2View()
    src = tmp_path / "static"
    src.mkdir()
    m = tmp_path / "static_m"
    m.mkdir()
    yy, xx = np.mgrid[0:H, 0:W]
    codes = np.stack([0.25 + 0.4 * xx / W, 0.3 + 0.3 * yy / H, np.full((H, W), 0.35)], -1)
    codes[8:20, 20:34] = 1.0
    codes = np.rint(codes * 255) / 255
    base = np.asarray(view.inverse(codes.astype(np.float32)), np.float64)
    r = base.copy()
    r[8:20, 20:34] *= 4.0
    for i in range(N):
        cv2.imwrite(str(src / f"f{i:04d}.png"), np.rint(codes[..., ::-1] * 255).astype(np.uint8))
        _write_exr(m / f"m.{i:06d}.exr", r)
    rc, c = _run(tmp_path, src, m, "steady")
    assert rc == 0 and c["pass"]
    assert c["max_transition_mean_ev"] < 1e-6


def test_injected_flicker_fails_by_its_size(clip):
    tmp, src, bases = clip
    m = tmp / "flicker"
    m.mkdir()
    for i, b in enumerate(bases):
        _write_exr(m / f"m.{i:06d}.exr", b * (2 ** (0.1 if i % 2 else 0.0)))   # whole frame +0.1 EV every other frame
    rc, c = _run(tmp, src, m, "flicker")
    assert rc == 1 and not c["pass"]
    assert c["max_transition_mean_ev"] == pytest.approx(0.1, abs=2e-3)


def test_aces2065_masters_read_as_acescg(clip):
    tmp, src, bases = clip
    m = tmp / "ap0"
    m.mkdir()
    ap1_to_ap0 = np.linalg.inv(AP0_TO_AP1)
    for i, b in enumerate(bases):
        _write_exr(m / f"m.{i:06d}.exr", b @ ap1_to_ap0.T, ap0=True)
    rc, c = _run(tmp, src, m, "ap0")
    assert rc == 0 and c["max_transition_mean_ev"] < 1e-4


def test_alpha_does_not_read_as_flicker(clip):
    tmp, src, bases = clip
    m = tmp / "alpha"
    m.mkdir()
    for i, b in enumerate(bases):
        a = np.clip((np.arange(W)[None, :] - 10 - 3 * i) / 20.0, 0, 1) * np.ones((H, 1))   # a moving matte edge
        _write_exr(m / f"m.{i:06d}.exr", b * a[..., None], alpha=a)   # premultiplied, as RUDRA writes
    rc, c = _run(tmp, src, m, "alpha")
    assert rc == 0 and c["max_transition_mean_ev"] < 1e-4


def test_mismatched_counts_are_an_error(clip, tmp_path):
    tmp, src, bases = clip
    m = tmp / "short"
    m.mkdir()
    _write_exr(m / "m.000000.exr", bases[0])
    rc, c = _run(tmp, src, m, "short")
    assert rc == 1 and "error" in c


def test_pool_and_change_match_the_hunyuan48_measure():
    y = np.random.default_rng(1).uniform(0.1, 2.0, (72, 128))
    assert np.allclose(adaptive_pool(y, (36, 64)), y.reshape(36, 2, 64, 2).mean((1, 3)))
    torch = pytest.importorskip("torch")
    z = np.random.default_rng(2).uniform(0.1, 2.0, (37, 70))
    want = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(z)[None, None], (36, 64))[0, 0].numpy()
    assert np.allclose(adaptive_pool(z), want)
    from tools.evaluate_aces_temporal import temporal_change as reference
    a, b, c, d = (np.random.default_rng(k).uniform(0.1, 2, (36, 64)) for k in range(4))
    assert temporal_change(a, b, c, d) == reference(a, b, c, d)
