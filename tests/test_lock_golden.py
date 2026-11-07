"""native/tests/golden/aces_lock/lock_parity_v1.bin is what the Python lock gives today (7 Nov 2026).

The native lock is held to that file (native/tests/test_aces_lock.cpp). If the oracle
(training/aces_partial_tighten.py) changes, this fails until the golden is rewritten with
tools/make_lock_golden.py, so the two cannot drift apart unnoticed. Structure, counts and
masks must match exactly; floating-point arrays within a tight tolerance, so a different
NumPy or BLAS build does not fail it on the last bit."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_lock_golden import main, read_golden  # noqa: E402

GOLDEN = ROOT / "native/tests/golden/aces_lock/lock_parity_v1.bin"


def assert_close(a, b, name):
    """Float arrays equal in shape and within 1e-6 relative (1e-9 absolute)."""
    assert a.shape == b.shape, name
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9, err_msg=name)


def test_the_committed_golden_is_what_the_oracle_gives_now(tmp_path):
    """Regenerate the golden and compare it with the committed one, component by component."""
    out = tmp_path / "lock.bin"
    assert main([str(out)]) == 0
    now, committed = read_golden(out.read_bytes()), read_golden(GOLDEN.read_bytes())
    assert now["header"] == committed["header"]
    assert np.array_equal(now["used"], committed["used"])
    for k in ("source", "constrained", "weight", "dequantized"):
        assert_close(now[k], committed[k], k)
    for i, (a, b) in enumerate(zip(now["cases"], committed["cases"])):
        assert a["meta"] == b["meta"], f"case {i}"
        assert a["strength"] == b["strength"], f"case {i}"
        assert_close(a["result"], b["result"], f"case {i} result")


def test_the_parser_refuses_a_bad_file():
    """Wrong magic, a truncated file and trailing bytes are all refused."""
    data = GOLDEN.read_bytes()
    with pytest.raises(ValueError, match="magic"):
        read_golden(b"XXXX" + data[4:])
    with pytest.raises(ValueError, match="truncated"):
        read_golden(data[:-1])
    with pytest.raises(ValueError, match="trailing"):
        read_golden(data + b"\0")
