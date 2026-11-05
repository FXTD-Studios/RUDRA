"""The reconstruction region oracle (composite spec revision 2, rudra/delivery/region.py)."""
import numpy as np
import pytest

from rudra.delivery.region import (ReconstructionRegion, reconstruction_gate, region_ramp_down,
                                   region_ramp_up)

Y = np.linspace(0.0, 1.0, 2001, dtype=np.float32)


def _rev1(y, sw=1.0):
    hp = 1.0 / (1.0 + np.exp(-((y - np.float32(0.82)) * np.float32(24.0))))
    sp = 1.0 / (1.0 + np.exp(-((np.float32(0.10) - y) * np.float32(24.0)))) * np.float32(sw)
    return np.maximum(hp, sp).astype(np.float32)


def test_off_is_revision_one():
    g = reconstruction_gate(Y, 0.8, "all", 1.0, ReconstructionRegion())
    assert np.array_equal(g, _rev1(Y, 0.8))


def test_ramps_have_compact_support_and_are_monotone():
    up = region_ramp_up(Y, 0.8, 0.1)
    assert np.all(up[Y <= np.float32(0.75)] == 0.0) and np.all(up[Y >= np.float32(0.85)] == 1.0)
    assert np.all(np.diff(up) >= 0)
    dn = region_ramp_down(Y, 0.2, 0.1)
    assert np.all(dn[Y >= np.float32(0.25)] == 0.0) and np.all(dn[Y <= np.float32(0.15)] == 1.0)
    assert np.all(np.diff(dn) <= 0)
    assert np.array_equal(region_ramp_up(Y, 0.5, 0.0), (Y >= np.float32(0.5)).astype(np.float32))


def test_region_only_narrows_the_reconstruction():
    r = ReconstructionRegion(enabled=True)
    g = reconstruction_gate(Y, 1.0, "all", 1.0, r)
    assert np.all(g <= _rev1(Y) + 1e-7)
    mid = (Y > 0.25) & (Y < 0.65)
    assert np.all(g[mid] == 0.0)          # mid-tones preserved exactly


def test_strengths_are_independent():
    r = ReconstructionRegion(enabled=True, shadow_strength=0.0)
    g = reconstruction_gate(Y, 1.0, "all", 1.0, r)
    assert np.all(g[Y < 0.05] == 0.0) and g[-1] > 0.9
    r = ReconstructionRegion(enabled=True, highlight_strength=0.0)
    g = reconstruction_gate(Y, 1.0, "all", 1.0, r)
    assert np.all(g[Y > 0.9] == 0.0) and g[0] > 0.5


def test_band_isolates_a_range():
    r = ReconstructionRegion(enabled=True, band=True, band_low=0.85, band_high=0.95, band_softness=0.02)
    g = reconstruction_gate(Y, 1.0, "all", 1.0, r)
    assert np.all(g[(Y < 0.84) | (Y > 0.96)] == 0.0)
    assert np.all(g[(Y > 0.87) & (Y < 0.93)] > 0.0)


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        reconstruction_gate(Y, 1.0, "everything", 1.0, ReconstructionRegion())
