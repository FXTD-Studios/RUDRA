"""Regression tests for the 9 Oct 2026 ship-path review.

reports/CODE_REVIEW_SHIP_PATH_2026-10-09.md has the findings these pin.
"""
from __future__ import annotations

import errno
import io
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))


# -- HLG: one system gamma, log10, every path --------------------------------

@pytest.mark.parametrize("peak", [600.0, 1000.0, 2000.0, 4000.0])
def test_deliver_hlg_puts_diffuse_white_at_203_nits_at_any_peak(peak):
    """`rudra deliver` used log2 for the system gamma: right at 1,000 nits
    only, +0.6 stop at 2,000 and +1.24 at 4,000."""
    from rudra.delivery import profiles
    from rudra.delivery.video import hlg_inverse_ootf, hlg_oetf
    white = np.full((1, 1, 3), 203.0 / peak)
    code = hlg_oetf(hlg_inverse_ootf(white, peak))
    shown = profiles.hlg_eotf(code, peak)
    assert shown[0, 0, 0] == pytest.approx(203.0, rel=1e-4)


@pytest.mark.parametrize("peak", [600.0, 2000.0, 4000.0])
def test_both_hlg_paths_agree(peak):
    """`rudra video` (profiles.encode_master) and `rudra deliver`
    (delivery.video) must code the same display light the same way."""
    from rudra.delivery import profiles
    from rudra.delivery.video import hlg_code
    rng = np.random.default_rng(9)
    nits = rng.uniform(0.5, 0.7 * peak, size=(8, 8, 3))
    # Below the knee, so the shoulder is identity and only the HLG coding differs.
    video_code, _ = profiles.encode_master(nits / 10_000.0, "hlg", peak, knee_nits=0.99 * peak)
    deliver_code = hlg_code(nits / peak, peak)
    np.testing.assert_allclose(video_code, deliver_code, atol=2e-4)


# -- Studio master: Region EV survives the anchor; linear is really 2020 -----

def _plate_png(value=(77, 77, 77), size=(64, 48)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, value).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def model():
    torch = pytest.importorskip("torch")
    from training.infer_sdr2hdr import load_models
    torch.set_num_threads(4)
    return load_models(str(ROOT / "checkpoints" / "sdr2hdr_shadow_v1.pt"), None,
                       torch.device("cpu"))[0]


def test_region_ev_lands_in_an_anchored_master(model, tmp_path):
    """The anchor sets every pixel below its knee to target/actual. Applied
    after Region EV it divided the push straight back out, while the sidecar
    said the grade was applied."""
    import server
    from rudra.delivery.exr import read_exr
    png = _plate_png()
    common = dict(render_dir=str(tmp_path), container="linear", anchor=True)
    band = {"label": "mids", "low_nits": 1.0, "high_nits": 100.0}
    server.run_master(model, png, dict(common, render_name="flat",
                                       regions=[dict(band, ev=0.0)]), SimpleNamespace())
    graded = server.run_master(model, png, dict(common, render_name="push",
                                                regions=[dict(band, ev=1.0)]), SimpleNamespace())
    flat, _ = read_exr(tmp_path / "flat.exr")
    push, _ = read_exr(tmp_path / "push.exr")
    assert graded["graded"] is True
    ratio = float(np.median(push[..., 1]) / np.median(flat[..., 1]))
    assert ratio == pytest.approx(2.0, rel=0.02), ratio


def test_linear_master_is_rec2020_and_says_so(model, tmp_path):
    """The linear container was labelled Rec.2020 but carried Rec.709 pixels
    and no chromaticities attribute."""
    import server
    from rudra.delivery.colorspace import REC2020_CHROMATICITIES
    from rudra.delivery.exr import read_exr
    png = _plate_png((200, 40, 40))
    base = dict(render_dir=str(tmp_path), anchor=False, carry_chroma=False,
                settle_grain=False)
    server.run_master(model, png, dict(base, render_name="lin", container="linear"),
                      SimpleNamespace())
    server.run_master(model, png, dict(base, render_name="aces", container="aces"),
                      SimpleNamespace())
    lin, attrs = read_exr(tmp_path / "lin.exr")
    np.testing.assert_allclose(attrs["chromaticities"], REC2020_CHROMATICITIES, atol=1e-6)
    sidecar = json.loads((tmp_path / "lin.json").read_text())
    assert "Rec.2020" in sidecar["container"]
    # Same picture as the ACES master, in 2020 primaries: compare through AP0.
    from rudra.delivery.colorspace import convert
    aces, _ = read_exr(tmp_path / "aces.exr")
    np.testing.assert_allclose(convert(lin, "rec2020", "ap0"), aces, rtol=2e-3, atol=1e-3)


# -- Studio master publication without hard links ----------------------------

def test_master_publishes_where_hard_links_do_not_exist(tmp_path, monkeypatch):
    import os
    import server

    def render(model, raw, params, args, out):
        out.write_bytes(b"exr")
        out.with_suffix(".json").write_text("{}")
        return {}

    def no_links(src, dst):
        raise OSError(errno.EPERM, "hard links not supported")

    monkeypatch.setattr(server, "_render_master", render)
    monkeypatch.setattr(os, "link", no_links)
    server.run_master(None, b"", dict(render_dir=str(tmp_path)), None)
    assert (tmp_path / "master.exr").read_bytes() == b"exr"
    assert (tmp_path / "master.json").is_file()
    with pytest.raises(ValueError, match="overwrite"):
        server.run_master(None, b"", dict(render_dir=str(tmp_path)), None)


def test_publish_without_links_never_replaces_an_existing_file(tmp_path, monkeypatch):
    import os
    import server
    staged = tmp_path / "staged.exr"
    staged.write_bytes(b"new")
    out = tmp_path / "out.exr"
    out.write_bytes(b"old")
    monkeypatch.setattr(os, "link", lambda s, d: (_ for _ in ()).throw(OSError(errno.EPERM, "x")))
    with pytest.raises(FileExistsError):
        server._publish_new(staged, out)
    assert out.read_bytes() == b"old"


# -- Studio Host check (DNS rebinding) ---------------------------------------

@pytest.mark.parametrize("host,ok", [
    ("localhost:8422", True), ("127.0.0.1:8422", True), ("[::1]:8422", True),
    ("192.168.1.20:8422", True), ("LOCALHOST", True),
    ("evil.example:8422", False), ("127.0.0.1.nip.io:8422", False), ("", False), (None, False),
])
def test_host_header_allowlist(host, ok, monkeypatch):
    import server
    monkeypatch.delenv("RUDRA_ALLOWED_HOSTS", raising=False)
    assert server.host_allowed(host) is ok


def test_named_hosts_can_be_allowed_explicitly(monkeypatch):
    import server
    monkeypatch.setenv("RUDRA_ALLOWED_HOSTS", "studio.lan, grade-01")
    assert server.host_allowed("studio.lan:8422")
    assert server.host_allowed("grade-01")
    assert not server.host_allowed("evil.example")


# -- encode_sequence failure paths --------------------------------------------

class _FakeProcess:
    """ffmpeg that has already exited: stdin raises like Windows does."""
    def __init__(self, args, **kw):
        self.args = args
        self.returncode = 1
        self.stdin = self
        self.stderr = io.BytesIO(b"[Eval] Undefined constant or missing '(' in 'bt2020nc'\n"
                                 b"Error opening output files: Invalid argument\n")
        self.closed = False
        self.killed = False

    def write(self, data):
        raise OSError(errno.EINVAL, "Invalid argument")

    def close(self):
        self.closed = True

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True


def test_windows_einval_surfaces_the_ffmpeg_error(tmp_path, monkeypatch):
    from rudra.delivery import video as video_mod
    monkeypatch.setattr(video_mod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(video_mod, "_ffmpeg_supports", lambda *a: False)
    monkeypatch.setattr(video_mod.subprocess, "Popen", _FakeProcess)
    frames = iter([np.full((16, 16, 3), 100.0)] * 2)
    with pytest.raises(video_mod.EncodeError, match="Undefined constant"):
        video_mod.encode_sequence(frames, tmp_path / "shot", target="hdr10")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_failed_sequence_leaves_no_file_behind(tmp_path):
    from rudra.delivery.video import EncodeError, encode_sequence

    def mixed():
        yield np.full((64, 64, 3), 100.0)
        yield np.full((32, 64, 3), 100.0)
    with pytest.raises(EncodeError, match="must not change size"):
        encode_sequence(mixed(), tmp_path / "x", target="hdr10")
    assert list(tmp_path.iterdir()) == []


# -- Tag readers do not read whole files --------------------------------------

def test_prores_frame_tags_reads_only_the_head(tmp_path, monkeypatch):
    from rudra.delivery import video as video_mod
    header = bytearray(20)
    header[0:2] = (148).to_bytes(2, "big")
    header[8:10] = (64).to_bytes(2, "big")
    header[10:12] = (32).to_bytes(2, "big")
    header[14], header[15], header[16] = 9, 16, 9
    path = tmp_path / "x.mov"
    path.write_bytes(b"\0" * 100 + b"icpf" + bytes(header) + b"\0" * 1000)
    monkeypatch.setattr(Path, "read_bytes",
                        lambda self: (_ for _ in ()).throw(AssertionError("whole-file read")))
    assert video_mod.prores_frame_tags(path) == {
        "color_primaries": "bt2020", "color_transfer": "smpte2084", "color_space": "bt2020nc"}


# -- Queue progress never fails a job ----------------------------------------

def test_queue_state_save_waits_out_a_windows_sharing_violation(tmp_path, monkeypatch):
    import os
    from rudra import batch
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "The process cannot access the file")
        return real(src, dst)

    monkeypatch.setattr(batch.os, "replace", flaky)
    monkeypatch.setattr(batch.time, "sleep", lambda s: None)
    target = tmp_path / "q.json.state.json"
    batch.save(target, {"ok": True})
    assert json.loads(target.read_text()) == {"ok": True}
    assert [p.name for p in tmp_path.iterdir()] == [target.name]


# -- EXR writes are atomic -----------------------------------------------------

def test_write_exr_leaves_nothing_at_the_target_when_it_fails(tmp_path, monkeypatch):
    from rudra.delivery import exr

    def boom(*a, **k):
        Path(a[0]).write_bytes(b"partial")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(exr, "_write_body", boom)
    with pytest.raises(OSError):
        exr.write_exr(tmp_path / "f.exr", np.zeros((4, 4, 3), np.float32))
    assert list(tmp_path.iterdir()) == []
