"""RUDRA 1.1: the AI fill's prompt presets and the suggestion (rudra/genfill/prompts.py)."""
import json
import math
from pathlib import Path

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


# ---- 1.2.6 (18 Nov 2026): the prompt from what the shot shows (Florence-2 captions) ----
def test_shot_prompt_says_what_the_bright_area_is_and_nothing_in_front_of_it():
    text, pid = P.shot_prompt("Two soldiers standing in the open cargo door of a military aircraft, looking out at a "
                              "bright, hazy sky over the ocean.", "A bright white hazy sky seen through an open door.")
    assert pid == "sky_overcast"
    assert text == ("bright hazy sky, soft even white light, seen through an open cargo door, over the ocean, "
                    "scene-referred high dynamic range footage, physically plausible highlights")
    assert "soldier" not in text and "aircraft" not in text
    assert P.shot_prompt("A beach at sunset.", "An orange sky with clouds above the sea.")[1] == "sky_sunset"
    assert P.shot_prompt("A living room.", "A lamp with a white shade glowing.")[1] == "light_source"
    assert P.shot_prompt("A campfire at night in a forest.", "Flames of a fire.")[1] == "flame"
    t, pid = P.shot_prompt("A man by a window in a dark room.", "A bright window with trees outside.")
    assert pid == "window_daylight" and "trees outside, scene-referred" in t
    t, pid = P.shot_prompt("A clear blue sky over mountains.", "Blue sky with white clouds.")
    assert pid == "sky_blue" and "a few thin high clouds" in t and "mountains" in t
    assert P.shot_prompt("A street with cars.", "", fallback="light_source")[1] == "light_source"   # nothing named: the rules' pick


def test_bright_crop_is_round_the_largest_bright_area():
    im = base()
    im[40:80, 100:200] = 252
    im[5:8, 5:8] = 255
    c = P.bright_crop(im)
    assert c is not None and c.shape[:2] == (80, 200)            # 40 x 100 grown by half each side
    assert P.bright_crop(base()) is None


class _FakeDescriber:
    def __init__(self):
        self.seen = []

    def caption(self, rgb, task="<DETAILED_CAPTION>"):
        self.seen.append(rgb.shape)
        return "A hazy sky over the sea." if len(self.seen) > 1 else "People standing in a doorway."


def test_describe_suggestion_is_first_and_says_why():
    im = base()
    im[: H // 2, 100:220] = 253
    d = _FakeDescriber()
    s = P.describe_suggestion(im, describer=d)
    assert s["id"] == "shot" and s["label"] == "From this shot" and s["region"] == "highlight"
    assert s["text"] == ("bright hazy sky, soft even white light, seen through an open doorway, over the sea, "
                         "scene-referred high dynamic range footage, physically plausible highlights")
    assert "Florence-2" in s["reason"] and len(d.seen) == 2


def test_cli_describe_falls_back_without_the_model(tmp_path, capsys, monkeypatch):
    import cv2
    im = base((150, 180, 240))
    im[: H // 3] = 255
    f = tmp_path / "sky.png"
    cv2.imwrite(str(f), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    def no_model(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(P, "Describer", no_model)
    assert P.main(["suggest", str(f), "--json", "--describe", "auto"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["suggestions"][0]["id"] == "sky_blue" and "no network" in out["describe_error"]
    monkeypatch.setattr(P, "Describer", lambda *a, **k: _FakeDescriber())
    assert P.main(["suggest", str(f), "--json", "--describe", "auto"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["suggestions"][0]["id"] == "shot" and out["suggestions"][1]["id"] == "sky_blue"


# ---- 1.3.2 (26 Nov 2026): the wording ends with the words the LoRA was trained under ----
def _base_prompt():
    """model.BASE_PROMPT without importing torch: read from model.py."""
    import ast
    src = (Path(P.__file__).parent / "model.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "BASE_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("no BASE_PROMPT in model.py")


def test_every_suggestion_ends_with_the_training_words():
    base = _base_prompt()
    for tail in (P.HIGHLIGHT_TAIL, P.SHADOW_TAIL):
        for phrase in tail.split(", "):
            assert phrase in base, (phrase, base)
    for p in P.PRESETS:
        assert p.text.endswith(P.HIGHLIGHT_TAIL if p.region == "highlight" else P.SHADOW_TAIL), p.id
        assert p.text.count("scene-referred") == 1
    for scene, area in (("A beach at sunset.", "An orange sky with clouds above the sea."),
                        ("A living room.", "A lamp with a white shade glowing."),
                        ("A street with cars.", "")):
        assert P.shot_prompt(scene, area)[0].endswith(P.HIGHLIGHT_TAIL)


def test_cli_describe_device_reaches_florence(tmp_path, capsys, monkeypatch):
    import cv2
    im = base((150, 180, 240))
    im[: H // 3] = 255
    f = tmp_path / "sky.png"
    cv2.imwrite(str(f), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    seen = []

    def describer(model_id, device="auto"):
        seen.append(device)
        return _FakeDescriber()
    monkeypatch.setattr(P, "Describer", describer)
    assert P.main(["suggest", str(f), "--json", "--describe", "auto", "--describe-device", "cpu"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert seen == ["cpu"] and out["suggestions"][0]["id"] == "shot"
