"""tools/colour_gate.py: the shot check's colour gate (1.3.1, 25 Nov 2026)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from rudra.delivery.colorspace import rgb_to_rgb_matrix

_spec = importlib.util.spec_from_file_location("colour_gate", Path(__file__).resolve().parents[1] / "tools" / "colour_gate.py")
cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cg)


def sdr(seed: int = 1, h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(12 / 255, 245 / 255, size=(h, w, 1))
    tint = rng.uniform(0.55, 1.0, size=(h, w, 3))
    return np.clip(base * tint / tint.max(-1, keepdims=True), 0, 1)


def as_master(light_709: np.ndarray, gain: float = 3.0, primaries: str = "ap1") -> np.ndarray:
    return (light_709 * gain) @ rgb_to_rgb_matrix("rec709", primaries).T


def test_the_sdrs_own_light_passes_at_any_exposure_and_in_any_primaries():
    c = sdr()
    for prim in ("ap1", "rec2020", "ap0"):
        r = cg.summarise([cg.measure(c, as_master(c ** 2.4, 5.0, prim), prim)])
        assert r["pass"] is True, (prim, r)
        for v in r["ranges"].values():
            assert abs(v["saturation_change"]) < 0.002 and v["hue_median_deg"] < 0.05, v


def test_the_srgb_reading_under_bt1886_fails_in_the_shadows():
    """1.3.0's analytic master: BT.1886 level, sRGB colour."""
    c = sdr()
    light = c ** 2.4
    srgb = cg.sdr_light(c, "srgb")
    luma = np.array([0.2126, 0.7152, 0.0722])
    master = srgb * ((light @ luma) / np.maximum(srgb @ luma, 1e-9))[..., None]
    r = cg.summarise([cg.measure(c, as_master(master), "ap1")])
    assert r["pass"] is False
    assert r["ranges"]["shadows"]["saturation_change"] < -0.10
    assert r["ranges"]["shadows"]["saturation_change"] < r["ranges"]["mid"]["saturation_change"]


def test_a_hue_shift_fails():
    c = sdr()
    rot = np.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]])
    r = cg.summarise([cg.measure(c, as_master((c ** 2.4) @ rot.T), "ap1")])
    assert r["pass"] is False
    assert max(v["hue_median_deg"] for v in r["ranges"].values()) > 3.0


def test_too_few_pixels_is_not_a_verdict():
    c = np.full((8, 8, 3), 0.5)
    r = cg.summarise([cg.measure(c, as_master(c ** 2.4), "ap1")])
    assert r["pass"] is None and r["ranges"]["mid"]["pass"] is None
