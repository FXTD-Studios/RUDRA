"""pipeline/render_pipeline_pairs.py: ACES SDR/HDR pipeline pairs (17 Oct 2026). Needs PyOpenColorIO."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("PyOpenColorIO")
from pipeline import render_pipeline_pairs as rp  # noqa: E402

FAMS = {f[0]: f for f in rp.FAMILIES}


def recipe(name, trim=0.0):
    f = FAMS[name]
    return {"family": f[0], "config": f[1], "sdr_display": list(f[2]), "hdr_display": list(f[3]),
            "hdr_peak_nits": f[4], "trim_ev": trim, "codec": "none", "quality": None}


def test_recipe_is_per_shot_and_deterministic():
    a, b = rp.shot_recipe("fxtd_lab_x", codecs=[]), rp.shot_recipe("fxtd_lab_x", codecs=[])
    assert a == b and abs(a["trim_ev"]) <= rp.TRIM_EV
    assert len({rp.shot_recipe(f"s{i}", codecs=[])["family"] for i in range(60)}) == len(rp.FAMILIES)


@pytest.mark.parametrize("name", list(FAMS))
def test_sdr_and_hdr_rise_together_and_respect_the_peak(name):
    ramp = np.geomspace(1e-3, 200.0, 64, dtype=np.float32)[None, :, None].repeat(3, 2)
    sdr, hdr, code8 = rp.render_pair(ramp, recipe(name))
    y_sdr, y_hdr = sdr[0, :, 1].astype(float), hdr[0, :, 1] * rp.DIFFUSE_WHITE
    assert np.all(np.diff(y_sdr) >= 0) and np.all(np.diff(y_hdr) >= -1e-3)
    assert y_hdr.max() <= FAMS[name][4] * 1.01 and code8.max() >= 250


def test_aces2_grey_lands_where_the_transform_puts_it():
    grey = np.full((1, 1, 3), 0.18, np.float32)
    _, hdr, _ = rp.render_pair(grey, recipe("aces2.1_1000"))
    assert hdr[0, 0, 1] * 203 == pytest.approx(14.5, abs=0.5)


def test_sdr_canonical_endpoints():
    assert rp.sdr_canonical(np.full((1, 1, 3), 255, np.uint8)).min() == 65535
    assert rp.sdr_canonical(np.zeros((1, 1, 3), np.uint8)).max() == 0


def test_ceiling_is_the_lower_of_peak_and_rendered_clip():
    r = recipe("aces1.3_1000")
    assert rp.shot_ceiling(r, None) == pytest.approx(970.0)
    clipped = rp.shot_ceiling(r, 0.97 * 9.27 * 203)
    assert clipped < 970.0 and clipped == pytest.approx(0.97 * rp.clip_white_nits(r, 9.27), rel=1e-4)
