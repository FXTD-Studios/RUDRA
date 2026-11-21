"""1.2.8 (QC of AI fill v4.3 on 09V, 21 Nov 2026): the two fixes on the Python side.

- composite.finish_rerender: the value carried from frame to frame is the fill's own, never the master's
  share of a region-edge pixel. 09V's antenna (2-3 px, dark against the sky) left a fan of faint copies of
  its old positions behind it (0.8, 0.64, ... of its depth).
- run.window_alignment / --rerender-align: with independent windows, each window is lined up with the one
  before it (a shift, measured on the frames they share) before the crossfade.
"""

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rudra.genfill import composite
from rudra.genfill.run import Rerender, align_frames, window_alignment


def _antenna_frame(x, h=1080, w=480):
    """A sky at code 252 with a thin rod at column x, as 09V's antenna against the door: motion-blurred, so
    only a few codes under the region's threshold (234-238), the master a stop darker there, and the region
    weight its ramp (partly on)."""
    codes = np.full((h, w, 3), 252, np.uint8)
    master = np.full((h, w, 3), 0.8, np.float32)
    for dx, code in ((-2, 238), (-1, 236), (0, 234), (1, 236), (2, 238)):
        codes[:, x + dx] = code
        master[:, x + dx] = 0.4
    t = np.clip((codes.max(-1).astype(np.float32) - 232.0) / 8.0, 0.0, 1.0)
    weight = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    fill = np.full((h, w, 3), 2.0, np.float32)            # the model: an even bright sky
    source = np.power(codes.astype(np.float32) / 255.0, 2.4)
    return fill, master, source, codes, weight


def test_a_moving_thin_dark_object_leaves_no_trail():
    state = {}
    xs = [60, 120, 180, 240, 300, 360]                    # the rod moves 60 px a frame (1080 lines)
    for x in xs:
        fill, master, source, codes, weight = _antenna_frame(x)
        out = composite.finish_rerender(fill, master, source, codes, weight, hue="source", colour=0.3, level=3.3,
                                        smooth=0.8, white=0.8, tau_high=240, state=state, dip=0.3, calm=0.9)
    y = np.log2(np.maximum(out @ composite.LUMA_AP1, 1e-6))
    sky = float(np.median(y[200:880, 420:470]))
    for x in xs[:-2]:                                    # where the rod was, now sky (clear of its edge rule)
        old = y[200:880, x - 2:x + 3]
        assert float(old.min()) > sky - 0.05, (x, float(old.min()), sky)


def test_the_master_share_is_this_frames_own():
    """The edge pixel's master share follows the master of THIS frame (no lag); only the fill's own value is smoothed."""
    state = {}
    fill, master, source, codes, weight = _antenna_frame(240)
    composite.finish_rerender(fill, master, source, codes, weight, hue="source", colour=0.3, level=3.3,
                              smooth=0.8, white=0.8, tau_high=240, state=state)
    assert state["log_y"].shape == codes.shape[:2]
    # what is carried is the fill's own level, not darkened by the rod's anti-aliased edge
    edge = state["log_y"][200:880, 238]
    sky = state["log_y"][200:880, 440]
    assert float(np.abs(edge - sky).max()) < 0.5


def _texture(h=232, w=416, seed=3):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 3) * 4
    t[150:] += 0.5                                       # a horizon
    return np.dstack([t] * 3)


def test_window_alignment_lines_up_a_slipped_window():
    img = _texture()
    h, w = img.shape[:2]
    m = np.float32([[1, 0, -0.7], [0, 1, 1.6]])            # the next window drew it 1.6 px lower, 0.7 px left
    slipped = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    region = np.zeros((h, w), bool)
    region[30:210, 80:340] = True
    warp, how = window_alignment([img] * 3, [slipped] * 3, [region] * 3)
    assert how["aligned"] and warp is not None and how["pairs"] == 3
    assert abs(how["dy_px"] - 1.6) < 0.1 and abs(how["dx_px"] + 0.7) < 0.1
    back = align_frames(slipped[None], warp)[0]
    assert np.abs(back - img)[region].mean() < 0.2 * np.abs(slipped - img)[region].mean()


def test_window_alignment_leaves_a_different_picture_alone():
    img = _texture()
    h, w = img.shape[:2]
    region = np.zeros((h, w), bool)
    region[30:210, 80:340] = True
    far = cv2.warpAffine(img, np.float32([[1, 0, 0], [0, 1, 40]]), (w, h), borderMode=cv2.BORDER_REFLECT)
    warp, how = window_alignment([img], [far], [region])
    assert warp is None and not how["aligned"]
    rng = np.random.default_rng(11)                       # another picture altogether: no match
    other = np.dstack([cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 3) * 4] * 3)
    warp, how = window_alignment([img], [other], [region])
    assert warp is None and not how["aligned"]
    flat = np.ones_like(img)
    warp, how = window_alignment([flat], [flat], [region])
    assert warp is None and how["reason"] == "nothing to measure on"


def test_align_is_off_unless_asked_and_the_run_takes_it():
    assert Rerender(source="x.mp4").align is False
    from rudra.genfill import run
    src = open(run.__file__, encoding="utf-8").read()
    assert "--rerender-align" in src and "rerender_align" in src
    assert "window_alignment(" in src
    assert math.isclose(run.ALIGN_MAX_SHIFT, 0.03)
