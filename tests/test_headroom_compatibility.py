"""Unsupported policies must fail before legacy decoding, inference or output."""
import sys
from pathlib import Path

import pytest

from rudra.input_policy import require_legacy_input_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import server


@pytest.mark.parametrize("params", [
    {"headroom_policy": "bt1886-headroom-v1"},
    {"headroom_policy": "future"},
    {"headroom_policy": None},
    {"decode": {"headroom_policy": True}},
    {"decode": {"headroom_policy": "bt1886-headroom-v1"}},
    {"decode": None},
    {"rudra_project": 2, "decode": {"headroom_policy": "bt1886-headroom-v1"}},
])
def test_every_legacy_entry_refuses_before_work(params, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Legacy decoder must not run for an unsupported policy")
    monkeypatch.setattr(server, "decode_sdr", unexpected)
    params = dict(params, render_dir=str(tmp_path))
    for call in (
        lambda: require_legacy_input_policy(params),
        lambda: server.run_inference(None, b"", params, None),
        lambda: server.run_frame(None, b"", params, None),
        lambda: server.master_targets(params),
        lambda: server.run_master(None, b"", params, None),
        lambda: server._render_master(None, b"", params, None, tmp_path / "out.exr"),
    ):
        with pytest.raises(ValueError):
            call()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("params", [{}, {"headroom_policy": "disabled"}, {"decode": {"headroom_policy": "disabled"}}])
def test_legacy_and_disabled_requests_keep_their_existing_plan(params, tmp_path):
    require_legacy_input_policy(params)
    assert server.master_targets(dict(params, render_dir=str(tmp_path))) == [tmp_path / "master.exr"]
