"""RUDRA 1.1: the AI fill's prompt presets and the suggestion (rudra/genfill/prompts.py)."""
import json
import math

import numpy as np
import pytest

pytest.importorskip("cv2")

from rudra.genfill import prompts as P  # noqa: E402

H, W = 180, 320


def base(colour=(40, 40, 40)):
    return np.full((H, W, 3), colour, np.uint8)


def disc(img, cy, cx, r, ring=None):
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    if ring is not None:
        img[(d2 < (r + 6) ** 2)] = ring
    img[d2 < r * r] = 255
    return img


def top(img):
    return [s["id"] for s in P.suggest([img])["suggestions"]]


def test_blue_sky():
    im = base((150, 180, 240))
    im[: H // 3] = 255
    assert top(im)[0] == "sky_blue"


def test_sunset_sky_and_overcast():
    warm = base((250, 150, 90))
    warm[: H // 3] = 255
    assert top(warm)[0] == "sky_sunset"
    grey = base((190, 190, 190))
    grey[: H // 3] = 255
    assert top(grey)[0] == "sky_overcast"


def test_sun_and_lamp():
    assert top(disc(base((90, 60, 40)), 90, 200, 10, ring=(250, 160, 80)))[0] == "sun_disc"
    assert top(disc(base(), 90, 200, 10, ring=(200, 200, 200)))[0] == "light_source"


def test_window():
    im = base((30, 30, 30))
    im[40:110, 180:270] = 255
    assert top(im)[0] == "window_daylight"


def test_glints():
    im = base((60, 60, 70))
    for cy, cx in ((20, 30), (50, 90), (80, 150), (110, 210), (140, 260), (160, 40)):
        im[cy:cy + 3, cx:cx + 3] = 255
    assert top(im)[0] == "specular_glints"


def test_a_flame_shows_itself_by_moving():
    frames = []
    for t in range(6):
        im = base((30, 20, 15))
        r = 12 + 6 * math.sin(t * 1.9)
        yy, xx = np.mgrid[0:H, 0:W]
        blob = ((yy - 120) ** 2 / (2.2 * r) ** 2 + (xx - 160) ** 2 / r ** 2) < 1
        im[np.roll(blob, t, 1) & (np.random.default_rng(t).random((H, W)) > 0.2)] = 255
        glow = ((yy - 120) ** 2 + (xx - 160) ** 2) < (3 * r) ** 2
        im[glow & (im.max(-1) < 255)] = (240, 140, 50)
        frames.append(im)
    assert "flame" in [s["id"] for s in P.suggest(frames)["suggestions"]]


def test_nothing_clipped_suggests_nothing_and_shadows_need_tau_low():
    im = base((100, 100, 100))
    assert P.suggest([im])["suggestions"] == []
    dark = base((0, 0, 0))
    assert P.suggest([dark])["suggestions"] == []                 # shadows off by default
    assert P.suggest([dark], tau_low=4)["suggestions"][0]["id"] == "shadow_detail"


def test_cache_key_is_the_prompt_banks(tmp_path):
    torch = pytest.importorskip("torch")
    from rudra.genfill.model import PromptBank
    for p in P.PRESETS:
        assert P.cache_key(p.text) == PromptBank.key(p.text)
    (tmp_path / f"{P.cache_key(P.PRESETS[0].text)}.pt").write_bytes(b"x")
    assert P.cached(P.PRESETS[0].text, tmp_path) and not P.cached(P.PRESETS[1].text, tmp_path)
    del torch


def test_cli_json(tmp_path, capsys):
    import cv2
    im = base((150, 180, 240))
    im[: H // 3] = 255
    f = tmp_path / "sky.png"
    cv2.imwrite(str(f), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    assert P.main(["suggest", str(f), "--json", "--cache", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["suggestions"][0]["id"] == "sky_blue" and len(out["presets"]) == len(P.PRESETS)
    assert out["suggestions"][0]["cached"] is False
    assert P.main(["suggest", str(tmp_path / "missing.png"), "--json"]) == 2
