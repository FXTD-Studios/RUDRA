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


# ---- RUDRA 1.1 priority 2: the region reaches the AI fill ----

from rudra.delivery.region import fill_region_weight, region_from_dict  # noqa: E402


def test_region_from_dict_reads_native_json_and_refuses_bad_values():
    r = region_from_dict({"band": True, "band_low": 0.6, "band_high": 0.9, "band_softness": 0.05,
                          "highlight_from": 0.75, "highlight_softness": 0.1, "highlight_strength": 1.0,
                          "shadow_below": 0.15, "shadow_softness": 0.1, "shadow_strength": 1.0})
    assert r.enabled and r.band and r.band_low == pytest.approx(0.6)
    assert not region_from_dict(None).enabled and not region_from_dict({}).enabled
    for bad in ({"band_low": 0.9, "band_high": 0.5}, {"highlight_softness": 0.8},
                {"shadow_strength": float("nan")}, {"typo": 1}):
        with pytest.raises(ValueError):
            region_from_dict(bad)


def test_fill_region_weight():
    codes = np.zeros((2, 3, 3), np.uint8)
    codes[0, 0] = 255                       # clipped white
    codes[0, 1] = 128                       # mid grey
    codes[1, 2] = 2                         # crushed black
    assert fill_region_weight(codes, region_from_dict(None)) is None
    w = fill_region_weight(codes, region_from_dict({"highlight_strength": 2.0}))
    assert w.dtype == np.float32 and w[0, 0] == 1.0      # strength 2 caps at a blend weight of 1
    assert w[0, 1] == 0.0
    assert w[1, 2] == 1.0                                # shadows on by default in the region
    w = fill_region_weight(codes, region_from_dict({"shadow_strength": 0.0}))
    assert w[1, 2] == 0.0 and w[0, 0] == 1.0


def test_run_checks_its_new_arguments():
    pytest.importorskip("cv2")
    from rudra.genfill.run import checked_prompt_strength, checked_shape_gain, parse_region, request_region
    assert checked_prompt_strength(1.0) == 1.0 and checked_shape_gain("0.5") == 0.5
    for f in (checked_prompt_strength, checked_shape_gain):
        for bad in (-0.1, 2.5, float("inf")):
            with pytest.raises(ValueError):
                f(bad)
    assert parse_region("") is None and parse_region('{"band": false}').enabled

    class A:
        region = '{"shadow_strength": 0.0}'
    assert request_region({}, A()).shadow_strength == 0.0
    assert request_region({"region": None}, A()) is None
    assert request_region({"region": {"band": True}}, A()).band


def test_the_fill_never_takes_a_masters_mattes_as_frames(tmp_path):
    from rudra.genfill.run import list_frames
    for n in ("m.000001.exr", "m.000001.mattes.exr", "m.000002.exr", "m.000002.MATTES.exr", ".hidden.exr"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in list_frames(tmp_path)] == ["m.000001.exr", "m.000002.exr"]
    assert [p.name for p in list_frames(tmp_path, "m.[0-9][0-9][0-9][0-9][0-9][0-9].exr")] == ["m.000001.exr", "m.000002.exr"]
