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
