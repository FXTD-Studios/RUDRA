"""rudra/genfill/ofx_prompt.py (1.3.2, 26 Nov 2026): the OFX plug-in's Suggest prompt and the prompt read while the
frames are collected. Torch-free: Florence-2, PyTorch and the text encoder are stand-ins."""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from rudra.genfill import ofx_job, ofx_prompt
from rudra.genfill import prompts as P

REPO = Path(__file__).resolve().parents[1]
cv2 = pytest.importorskip("cv2")


def frame(tmp: Path, name: str = "frame.tif") -> Path:
    """A 16-bit TIFF as the plug-in writes it: a dark frame with a clipped sky at the top, blue around it."""
    im = np.full((180, 320, 3), 40, np.uint8)
    im[:] = (150, 180, 240)
    im[:60] = 255
    f = tmp / name
    cv2.imwrite(str(f), cv2.cvtColor(im.astype(np.uint16) * 257, cv2.COLOR_RGB2BGR))
    return f


def read(out: Path) -> dict:
    return json.loads(out.read_text(encoding="utf-8"))


def test_suggest_without_florence_ranks_the_presets_and_says_why(tmp_path, capsys):
    out = tmp_path / "suggest.json"
    assert ofx_prompt.main(["suggest", "--repo", str(REPO), "--frame", str(frame(tmp_path)), "--out", str(out)]) == 0
    r = read(out)
    assert r["ok"] and r["suggestions"][0]["id"] == "sky_blue"
    assert "Florence-2 is not set up" in r["describe_error"]
    assert r["suggestions"][0]["text"].endswith(P.HIGHLIGHT_TAIL) and r["suggestions"][0]["cached"] is False
    assert capsys.readouterr().out == ""                       # nothing on stdout: the plug-in does not read it
    assert not (tmp_path / "suggest.json.part").exists()


class _Florence:
    devices = []

    def __init__(self, model_id, device="auto"):
        _Florence.devices.append(device)

    def caption(self, rgb, task="<DETAILED_CAPTION>"):
        return "A hazy sky over the sea." if rgb.shape[0] < 180 else "People in a doorway."


def test_suggest_with_florence_puts_from_this_shot_first(tmp_path, monkeypatch):
    florence = tmp_path / "Florence-2-base"
    florence.mkdir()
    (florence / "config.json").write_text("{}")
    monkeypatch.setattr(P, "Describer", _Florence)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    out = tmp_path / "suggest.json"
    _Florence.devices = []
    assert ofx_prompt.main(["suggest", "--repo", str(REPO), "--frame", str(frame(tmp_path)), "--out", str(out),
                            "--florence", str(florence), "--cache", str(cache), "--device", "cpu"]) == 0
    r = read(out)
    assert r["ok"] and r["suggestions"][0]["id"] == "shot" and r["suggestions"][1]["id"] == "sky_blue"
    assert "describe_error" not in r and _Florence.devices == ["cpu"]
    import os
    assert os.environ.get("HF_HUB_OFFLINE") == "1"             # from its folder only, never the network


def test_suggest_that_fails_says_why_in_the_file(tmp_path):
    out = tmp_path / "suggest.json"
    assert ofx_prompt.main(["suggest", "--repo", str(REPO), "--frame", str(tmp_path / "missing.tif"), "--out", str(out)]) == 2
    r = read(out)
    assert r["ok"] is False and "cannot read" in r["error"]


def _fake_torch_and_model(monkeypatch, record):
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float32 = "bf16", "f32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    model = types.ModuleType("rudra.genfill.model")

    class Bank:
        def __init__(self, base, cache, device="cpu", dtype=None):
            record.update(base=base, cache=Path(cache), device=device, dtype=dtype, got=[])
            self.encoder_device = device

        def get(self, text):
            record["locked"] = (record["cache"] / ofx_job.EMBED_LOCK).exists()
            record["got"].append(text)
            (record["cache"] / f"{P.cache_key(text)}.pt").write_bytes(b"e")

        def release(self):
            record["released"] = True
    model.PromptBank = Bank
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "rudra.genfill.model", model)


def test_embed_reads_only_what_is_missing_under_the_lock(tmp_path, monkeypatch):
    lora = tmp_path / "models" / "rudra_genfill.safetensors"
    (lora.parent / "prompt_cache").mkdir(parents=True)
    base_prompt = ofx_prompt.model_base_prompt(REPO)
    assert base_prompt.startswith("scene-referred")
    (lora.parent / "prompt_cache" / f"{P.cache_key(base_prompt)}.pt").write_bytes(b"b")   # the app's: copied in
    cache = tmp_path / "ofx_fill" / "prompt_cache"
    record = {}
    _fake_torch_and_model(monkeypatch, record)
    args = ["embed", "--repo", str(REPO), "--base", "B", "--lora", str(lora), "--cache", str(cache), "--text", " my sky "]
    assert ofx_prompt.main(args) == 0
    assert record["got"] == ["my sky"] and record["locked"] and record["released"]
    assert record["device"] == "cpu" and record["dtype"] == "f32"
    assert (cache / f"{P.cache_key(base_prompt)}.pt").is_file() and not (cache / ofx_job.EMBED_LOCK).exists()
    record.clear()
    assert ofx_prompt.main(args) == 0 and record == {}         # everything cached: PyTorch is not even loaded


def test_embed_leaves_it_to_a_reading_that_runs(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / ofx_job.EMBED_LOCK).write_text("{}")
    record = {}
    _fake_torch_and_model(monkeypatch, record)
    assert ofx_prompt.main(["embed", "--repo", str(REPO), "--base", "B", "--lora", str(tmp_path / "l.safetensors"),
                            "--cache", str(cache), "--text", "new"]) == 0
    assert record == {} and (cache / ofx_job.EMBED_LOCK).exists()   # its lock left alone


def test_embed_that_fails_lets_go_of_the_lock(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float32 = "bf16", "f32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    model = types.ModuleType("rudra.genfill.model")

    class Broken:
        def __init__(self, *a, **k):
            raise OSError("no text encoder")
    model.PromptBank = Broken
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "rudra.genfill.model", model)
    t0 = time.time()
    assert ofx_prompt.main(["embed", "--repo", str(REPO), "--base", "B", "--lora", str(tmp_path / "l.safetensors"),
                            "--cache", str(cache), "--text", "new"]) == 2
    assert not (cache / ofx_job.EMBED_LOCK).exists() and time.time() - t0 < 10
