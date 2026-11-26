"""rudra/genfill/ofx_job.py: AI fill for the OFX plug-in (phase 3, 26 Nov 2026).

The job runs AI fill v5 with the app's own command line on the frames the plug-in stored, and reports in
status.json. Here a stand-in fill (a small Python script) speaks rudra.genfill.run's JSON lines."""
from __future__ import annotations

import json
import re
import sys
import textwrap
import threading
import time
from pathlib import Path

from rudra.genfill import ofx_job

REPO = Path(__file__).resolve().parents[1]

FAKE_RUN = textwrap.dedent('''
    import json, os, sys
    a = sys.argv[1:]
    out = a[a.index("--out") + 1]
    mode = os.environ.get("FAKE_RUN", "ok")
    say = lambda **e: print(json.dumps(e), flush=True)
    say(event="ready", device="cpu")
    if mode == "cancel":
        for line in sys.stdin:
            if line.strip() == "cancel":
                sys.exit(1)
    if mode == "error":
        say(event="error", ok=False, error="RuntimeError", detail="boom")
        sys.exit(2)
    os.makedirs(out)
    for i in range(2):
        open(os.path.join(out, f"f.{i:06d}.exr"), "w").write("fill")
        say(event="frame", index=i + 1, of=2)
    say(event="done", ok=True, generated_fraction=0.25, fill_flicker_p95_ev=0.03, fallback_frames=0, seconds=1.0)
''')


def stored_shot(tmp: Path, frames: int = 2, condition: str = "bt1886") -> Path:
    d = tmp / "shot"
    (d / "codes").mkdir(parents=True)
    (d / "master").mkdir()
    for i in range(frames):
        (d / "codes" / f"f.{i:06d}.tif").write_bytes(b"tif")
        (d / "master" / f"f.{i:06d}.exr").write_bytes(b"exr")
    (d / "request.json").write_text(json.dumps({"frames": frames, "condition": condition, "seed": 3, "prompt": "sky",
                                                "region": {"enabled": True}}), encoding="utf-8")
    return d


def run_job(tmp: Path, d: Path, mode: str, monkeypatch) -> int:
    script = tmp / "fake_run.py"
    script.write_text(FAKE_RUN, encoding="utf-8")
    monkeypatch.setenv("FAKE_RUN", mode)
    return ofx_job.main(["--dir", str(d), "--repo", str(REPO), "--base", "base", "--lora", "lora", "--skip-lora-check"],
                        runner=[sys.executable, str(script)])


def status(d: Path) -> dict:
    return json.loads((d / "status.json").read_text(encoding="utf-8"))


def test_the_v5_settings_are_the_apps():
    header = (REPO / "native" / "app" / "genfill_job.hpp").read_text(encoding="utf-8")
    for name, value in (("kGenFillV5Sha256", ofx_job.V5_SHA256), ("kGenFillV5Hold", ofx_job.V5_HOLD),
                        ("kGenFillV5BlendStill", ofx_job.V5_BLEND_STILL), ("kGenFillV5HoldCalm", ofx_job.V5_HOLD_CALM)):
        m = re.search(name + r'\s*=\s*"([^"]*)"', header)
        assert m and m.group(1) == value, name
    assert ofx_job.STEPS == 30   # 8 steps drew stripes on 09V (1.3.1)


def test_the_command_line_is_the_apps_region_rerender_with_v5():
    a = ofx_job.recipe("M", "O", "S", "B", "L", "bt1886", 3, " sky ", '{"enabled": true}')
    assert a[:4] == ["-m", "rudra.genfill.run", "--master", "M"]
    for flag, value in (("--source", "S"), ("--condition", "bt1886"), ("--tau-high", "240"), ("--tau-low", "-1"),
                        ("--feather", "8"), ("--steps", "30"), ("--seed", "3"), ("--prompt-highlight", "sky"),
                        ("--region", '{"enabled": true}'), ("--glob", "f.*.exr"), ("--rerender-hold-clipped", "thin"),
                        ("--rerender-blend-still", "0.95"), ("--rerender-hold-calm", "0.95"), ("--rerender-calm", "0.9")):
        assert a[a.index(flag) + 1] == value, flag
    for flag in ("--region-rerender", "--rerender-independent", "--rerender-full-last", "--json"):
        assert flag in a
    assert "--rerender-cap-master" not in a and "--mask-first-frame" not in a
    learned = ofx_job.recipe("M", "O", "S", "B", "L", "aces", 0, "", "")
    assert "--rerender-cap-master" in learned and "--mask-first-frame" in learned
    assert "--prompt-highlight" not in learned and "--region" not in learned


def test_a_fill_that_finishes_is_renamed_whole_and_says_done(tmp_path, monkeypatch):
    d = stored_shot(tmp_path)
    assert run_job(tmp_path, d, "ok", monkeypatch) == 0
    s = status(d)
    assert s["state"] == "done" and s["done"] == 2 and s["of"] == 2
    assert s["stats"]["fill_flicker_p95_ev"] == 0.03
    assert (d / "fill" / "f.000001.exr").is_file() and not (d / "fill.part").exists()
    log = (d / "job.log").read_text(encoding="utf-8")
    assert "--prompt-highlight sky" in log and "--region" in log


def test_a_fill_that_fails_says_why(tmp_path, monkeypatch):
    d = stored_shot(tmp_path)
    assert run_job(tmp_path, d, "error", monkeypatch) == 2
    s = status(d)
    assert s["state"] == "failed" and "boom" in s["message"]
    assert not (d / "fill").exists() and not (d / "fill.part").exists()


def test_cancel_stops_the_fill(tmp_path, monkeypatch):
    d = stored_shot(tmp_path)
    threading.Timer(1.5, lambda: (d / "cancel").write_text("Cancel pressed\n")).start()
    t0 = time.time()
    assert run_job(tmp_path, d, "cancel", monkeypatch) == 1
    assert time.time() - t0 < 30
    assert status(d)["state"] == "cancelled"
    assert not (d / "fill").exists()


def test_missing_frames_fail_before_anything_runs(tmp_path, monkeypatch):
    d = stored_shot(tmp_path)
    (d / "master" / "f.000001.exr").unlink()
    assert run_job(tmp_path, d, "ok", monkeypatch) == 2
    s = status(d)
    assert s["state"] == "failed" and "1 of 2 frames stored in master" in s["message"]


def test_a_lora_that_is_not_v5_is_refused(tmp_path):
    d = stored_shot(tmp_path)
    lora = tmp_path / "other.safetensors"
    lora.write_bytes(b"not v5")
    assert ofx_job.main(["--dir", str(d), "--repo", str(REPO), "--base", "b", "--lora", str(lora)]) == 2
    assert "is not AI fill v5" in status(d)["message"]


def test_steps_and_the_shared_prompt_cache(tmp_path):
    """27 Nov 2026: the plug-in's Steps reach the fill (10..50), and a prompt is read from one cache for every OFX
    fill, seeded from the cache beside the LoRA (the app's presets)."""
    assert ofx_job.steps_of({"steps": 20}) == 20
    assert ofx_job.steps_of({"steps": 3}) == 10 and ofx_job.steps_of({"steps": 99}) == 50
    assert ofx_job.steps_of({}) == ofx_job.STEPS and ofx_job.steps_of({"steps": "x"}) == ofx_job.STEPS
    a = ofx_job.recipe("M", "O", "S", "B", "L", "bt1886", 0, "sky", "", 20, "C")
    assert a[a.index("--steps") + 1] == "20" and a[a.index("--prompt-cache") + 1] == "C"
    assert "--prompt-cache" not in ofx_job.recipe("M", "O", "S", "B", "L", "bt1886", 0, "", "")
    lora = tmp_path / "models" / "rudra_genfill.safetensors"
    (lora.parent / "prompt_cache").mkdir(parents=True)
    (lora.parent / "prompt_cache" / "abc.pt").write_bytes(b"x")
    folder = tmp_path / "ofx_fill" / "shot" / "key"
    folder.mkdir(parents=True)
    cache = ofx_job.prompt_cache_for(folder, lora)
    assert cache == tmp_path / "ofx_fill" / "prompt_cache" and (cache / "abc.pt").read_bytes() == b"x"


def test_prompt_strength_reaches_the_fill(tmp_path, monkeypatch):
    """1.3.2: the plug-in's Prompt strength (0..2) goes to the fill as the app's Deliver passes it: only when not 1,
    and only with a prompt."""
    assert ofx_job.strength_of({"prompt_strength": 1.5}) == 1.5
    assert ofx_job.strength_of({"prompt_strength": 9}) == 2.0 and ofx_job.strength_of({"prompt_strength": -1}) == 0.0
    assert ofx_job.strength_of({}) == 1.0 and ofx_job.strength_of({"prompt_strength": "x"}) == 1.0
    assert ofx_job.strength_of({"prompt_strength": float("nan")}) == 1.0
    a = ofx_job.recipe("M", "O", "S", "B", "L", "bt1886", 0, "sky", "", 30, "", 1.5)
    assert a[a.index("--prompt-strength") + 1] == "1.5"
    assert "--prompt-strength" not in ofx_job.recipe("M", "O", "S", "B", "L", "bt1886", 0, "sky", "", 30, "", 1.0)
    assert "--prompt-strength" not in ofx_job.recipe("M", "O", "S", "B", "L", "bt1886", 0, "", "", 30, "", 1.5)
    d = stored_shot(tmp_path)
    req = json.loads((d / "request.json").read_text(encoding="utf-8"))
    req["prompt_strength"] = 0.5
    (d / "request.json").write_text(json.dumps(req), encoding="utf-8")
    assert run_job(tmp_path, d, "ok", monkeypatch) == 0
    assert "--prompt-strength 0.5" in (d / "job.log").read_text(encoding="utf-8")


class _Status:
    def __init__(self):
        self.messages = []

    def set(self, **kw):
        self.messages.append(kw.get("message", ""))


def test_the_fill_waits_while_the_prompt_is_read(tmp_path):
    """1.3.2: Generate starts reading a new prompt (ofx_prompt.py embed) while the frames are collected; the fill
    waits for its lock, ignores a lock left by a process that ended, and gives up waiting after the limit."""
    cache = tmp_path / "prompt_cache"
    cache.mkdir()
    assert ofx_job.wait_for_embedding(cache, _Status(), poll=0.05)               # no lock: at once
    lock = cache / ofx_job.EMBED_LOCK
    lock.write_text("{}")
    threading.Timer(1.0, lock.unlink).start()
    st, t0 = _Status(), time.time()
    assert ofx_job.wait_for_embedding(cache, st, poll=0.05)
    assert 0.8 < time.time() - t0 < 20 and "reading the prompt" in st.messages[-1]
    lock.write_text("{}")
    old = time.time() - ofx_job.EMBED_STALE - 5
    import os
    os.utime(lock, (old, old))                                                    # its process ended long ago
    assert not ofx_job.embedding_running(cache) and ofx_job.wait_for_embedding(cache, _Status(), poll=0.05)
    os.utime(lock, None)
    assert not ofx_job.wait_for_embedding(cache, _Status(), limit=0.2, poll=0.05)  # gave up
    lock.unlink()


def test_the_seed_skips_the_lock_and_hidden_files(tmp_path):
    lora = tmp_path / "models" / "rudra_genfill.safetensors"
    beside = lora.parent / "prompt_cache"
    beside.mkdir(parents=True)
    (beside / "abc.pt").write_bytes(b"x")
    (beside / ofx_job.EMBED_LOCK).write_text("{}")
    cache = ofx_job.seed_cache(tmp_path / "cache", lora)
    assert (cache / "abc.pt").is_file() and not (cache / ofx_job.EMBED_LOCK).exists()


def test_the_plug_ins_time_estimate_knows_the_prompt_pass():
    """1.3.2: the plug-in counts a second transformer pass a step for a blended prompt (fill_store.hpp
    kSinglePromptPass), which must be run.py's PROMPT_PASS."""
    from rudra.genfill import run
    hpp = (REPO / "native" / "ofx" / "fill_store.hpp").read_text(encoding="utf-8")
    m = re.search(r"kSinglePromptPass\s*=\s*(true|false)", hpp)
    assert m and (m.group(1) == "true") == (run.PROMPT_PASS == "single")
