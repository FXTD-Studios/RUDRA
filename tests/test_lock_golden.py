"""native/tests/golden/aces_lock/lock_parity_v1.bin is what the Python lock gives today (7 Nov 2026).

The native lock is held to that file (native/tests/test_aces_lock.cpp). If the oracle
(training/aces_partial_tighten.py) changes, this fails until the golden is rewritten with
tools/make_lock_golden.py, so the two cannot drift apart unnoticed."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_lock_golden import main  # noqa: E402


def test_the_committed_golden_is_what_the_oracle_gives_now(tmp_path):
    out = tmp_path / "lock.bin"
    assert main([str(out)]) == 0
    assert out.read_bytes() == (ROOT / "native/tests/golden/aces_lock/lock_parity_v1.bin").read_bytes()
