"""Packaged RUDRA ships only python/rudra and python/config. Every rudra module the app runs must
import with nothing else on the path: 1.3.1's region re-render imported tools.measure_flicker and
the app's AI fill preview failed with ModuleNotFoundError: tools."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTSIDE = re.compile(r"^\s*(from|import)\s+(tools|scripts|tests)\b", re.M)


def test_no_rudra_module_imports_the_dev_folders():
    bad = [f"{p.relative_to(ROOT)}: {m.group(0).strip()}"
           for p in (ROOT / "rudra").rglob("*.py") for m in OUTSIDE.finditer(p.read_text(encoding="utf-8"))]
    assert not bad, "imports outside the packaged folders:\n" + "\n".join(bad)


def test_genfill_imports_as_packaged(tmp_path):
    for name in ("rudra", "config"):
        if (ROOT / name).is_dir():
            shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    code = ("import sys; sys.path.insert(0, '.'); import rudra.genfill.run, rudra.genfill.ofx_job, rudra.genfill.ofx_prompt, rudra.genfill.prompts, rudra.genfill.decode as d;"
            "assert '.png' in d.FRAME_EXT and callable(d.decode_movie)")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    if "No module named 'torch'" in r.stderr or "No module named 'cv2'" in r.stderr:
        import pytest
        pytest.skip("heavy dependency missing in this Python")
    assert r.returncode == 0, r.stderr
