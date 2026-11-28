"""1.2.6 (readiness report, checklist item 7): the AI fill's offline install and hashed, pinned models."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from rudra.genfill import setup


def _model(root: Path, rev: str | None = "a" * 40) -> Path:
    (root / "transformer").mkdir(parents=True)
    (root / "transformer" / "config.json").write_text('{"_class_name": "WanVACETransformer3DModel"}')
    (root / "model_index.json").write_text("{}")
    (root / "vae.bin").write_bytes(b"\x01" * 1000)
    if rev:   # what huggingface_hub writes for a local_dir download
        meta = root / ".cache" / "huggingface" / "download" / "vae.bin.metadata"
        meta.parent.mkdir(parents=True)
        meta.write_text(f"{rev}\netag\n123.0\n")
    return root


def test_the_manifest_has_the_revision_and_every_file_but_the_cache(tmp_path):
    e = setup.manifest_entry(_model(tmp_path / "wan"), setup.REPO_ID)
    assert e["revision"] == "a" * 40
    assert sorted(e["files"]) == ["model_index.json", "transformer/config.json", "vae.bin"]
    assert e["files"]["vae.bin"]["size"] == 1000
    assert setup.verify(tmp_path / "wan", e) == []


def test_verify_names_a_missing_a_resized_and_a_changed_file(tmp_path):
    root = _model(tmp_path / "wan")
    e = setup.manifest_entry(root, setup.REPO_ID)
    (root / "model_index.json").unlink()
    (root / "vae.bin").write_bytes(b"\x02" * 1000)                   # same size, other bytes
    (root / "transformer" / "config.json").write_text("{}")         # other size
    p = setup.verify(root, e)
    assert any("missing model_index.json" in x for x in p)
    assert any("vae.bin: SHA-256" in x for x in p)
    assert any("transformer/config.json" in x and "bytes" in x for x in p)
    assert not any("vae.bin" in x for x in setup.verify(root, e, hashes=False))   # sizes only


def test_an_install_from_a_zip_is_checked_then_moved_into_place(tmp_path):
    good = _model(tmp_path / "good", rev=None)
    e = setup.manifest_entry(good, setup.REPO_ID)
    z = tmp_path / "wan.zip"
    with zipfile.ZipFile(z, "w") as f:
        for rel, p in setup.model_files(good):
            f.write(p, "Wan2.1-VACE-1.3B-diffusers/" + rel)   # one top folder inside
    dest = tmp_path / "models" / "Wan2.1-VACE-1.3B-diffusers"
    assert setup.install_from(z, dest, e) == []
    assert setup.have_model(dest) and not dest.with_name(dest.name + ".installing").exists()

    bad = dict(e, files={**e["files"], "vae.bin": {"size": 1000, "sha256": "0" * 64}})
    again = tmp_path / "other"
    p = setup.install_from(z, again, bad)
    assert p and not again.exists() and not again.with_name("other.installing").exists()


def test_an_unsafe_zip_is_refused(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../outside.txt", "x")
    assert "unsafe" in setup.install_from(z, tmp_path / "dest", {})[0]
    assert not (tmp_path / "outside.txt").exists()


def test_a_folder_install_keeps_the_model_that_was_there(tmp_path):
    good = _model(tmp_path / "good")
    e = setup.manifest_entry(good, setup.REPO_ID)
    dest = _model(tmp_path / "dest", rev=None)
    assert setup.install_from(good, dest, e) == []
    assert (tmp_path / "dest.before_install" / "vae.bin").is_file()


def test_make_manifest_then_offline_install_from_the_command_line(tmp_path, capsys):
    base = _model(tmp_path / "models" / "Wan2.1-VACE-1.3B-diffusers")
    fl = tmp_path / "models" / setup.FLORENCE_DIR
    fl.mkdir()
    (fl / "config.json").write_text("{}")
    man = tmp_path / "manifest.json"
    assert setup.main(["--base", str(base), "--make-manifest", "--manifest", str(man), "--json"]) == 0
    m = json.loads(man.read_text())
    assert m["schema"] == "rudra_model_manifest_v1" and m["wan"]["revision"] == "a" * 40
    assert m["florence"]["repo_id"] == setup.FLORENCE_ID and "config.json" in m["florence"]["files"]

    # No hashes for the model: an offline install is refused unless asked for.
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    assert setup.main(["--base", str(tmp_path / "x"), "--from", str(base), "--manifest", str(empty), "--json"]) == 1
    assert "no hashes" in capsys.readouterr().out


def _florence(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "florence2"}')
    (root / "model.safetensors").write_bytes(b"\x03" * 500)


def test_florence_only_downloads_at_the_pinned_revision_and_checks_it(tmp_path, monkeypatch, capsys):
    good = tmp_path / "good"
    _florence(good)
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"florence": setup.manifest_entry(good, setup.FLORENCE_ID, "b" * 40)}))
    asked = {}

    def fake_download(dest, repo_id=setup.REPO_ID, revision=None):
        asked.update(repo=repo_id, rev=revision)
        _florence(dest)

    monkeypatch.setattr(setup, "download_model", fake_download)
    base = tmp_path / "models" / "Wan2.1-VACE-1.3B-diffusers"   # no Wan, no GPU: not needed for Florence-2
    assert setup.main(["--base", str(base), "--florence-only", "--manifest", str(man), "--json"]) == 0
    assert asked == {"repo": setup.FLORENCE_ID, "rev": "b" * 40}
    assert (tmp_path / "models" / setup.FLORENCE_DIR / "config.json").is_file()
    assert '"ok": true' in capsys.readouterr().out.strip().splitlines()[-1]


def test_a_florence_download_that_does_not_match_is_removed(tmp_path, monkeypatch):
    good = tmp_path / "good"
    _florence(good)
    entry = setup.manifest_entry(good, setup.FLORENCE_ID, "d" * 40)   # pinned: downloads only from a pinned entry

    def tampered(dest, repo_id=setup.REPO_ID, revision=None):
        _florence(dest)
        (dest / "model.safetensors").write_bytes(b"\x04" * 500)

    monkeypatch.setattr(setup, "download_model", tampered)
    dest = tmp_path / "Florence-2-base"
    problems, warning = setup.setup_florence(dest, entry, None)
    assert problems and warning is None and not dest.exists()


def test_a_florence_download_that_fails_is_only_a_warning(tmp_path, monkeypatch):
    def offline(dest, repo_id=setup.REPO_ID, revision=None):
        raise OSError("no network")

    monkeypatch.setattr(setup, "download_model", offline)
    problems, warning = setup.setup_florence(tmp_path / "Florence-2-base", {}, None, allow_unverified=True)
    assert problems == [] and "presets" in warning


def test_nothing_is_downloaded_unchecked(tmp_path, monkeypatch, capsys):
    """Code review, 19 Nov 2026: a missing or unreadable manifest used to let a download go unchecked."""
    called = []
    monkeypatch.setattr(setup, "download_model", lambda *a, **k: called.append(a))
    problems, warning = setup.setup_florence(tmp_path / "Florence-2-base", {}, None)
    assert problems == [] and "not pinned" in warning and not called
    bad = tmp_path / "broken.json"
    bad.write_text("{not json")
    assert setup.main(["--base", str(tmp_path / "wan"), "--manifest", str(bad), "--json"]) == 1
    assert "cannot be read" in capsys.readouterr().out
    # --allow-unverified goes on without it (here to the check, which reports what is missing).
    setup.main(["--base", str(tmp_path / "wan"), "--manifest", str(bad), "--allow-unverified", "--check-only", "--json"])
    assert "cannot be read" not in capsys.readouterr().out


def test_a_florence_download_that_stopped_is_resumed(tmp_path, monkeypatch):
    good = tmp_path / "good"
    _florence(good)
    entry = setup.manifest_entry(good, setup.FLORENCE_ID, "c" * 40)
    dest = tmp_path / "Florence-2-base"
    dest.mkdir()
    (dest / "config.json").write_text('{"model_type": "florence2"}')   # what a stopped download leaves
    resumed = []

    def resume(d, repo_id=setup.REPO_ID, revision=None):
        resumed.append(revision)
        _florence(d)

    monkeypatch.setattr(setup, "download_model", resume)
    assert setup.setup_florence(dest, entry, None) == ([], None)
    assert resumed == ["c" * 40]
    # Whole: no download at all.
    resumed.clear()
    assert setup.setup_florence(dest, entry, None) == ([], None)
    assert resumed == []


# 1.3.2 (end-user report, 27 Nov 2026): "Set up automatically" failed on computers where it had stopped once.

def _ready_machine(monkeypatch):
    """A computer that can run the fill, with every package: only the model step is left to run."""
    monkeypatch.setattr(setup, "machine", lambda base: {"device": "cuda", "memory_gb": 16.0, "gpu": "Test GPU",
                                                        "torch": "2.9.0+cu128", "free_gb": 500.0})
    monkeypatch.setattr(setup, "missing_packages", lambda: [])


def test_a_wan_download_that_stopped_is_carried_on(tmp_path, monkeypatch, capsys):
    """A stopped download has its small files (transformer/config.json among them) and not the weights: it was
    taken for the model and refused for the files it lacked, on every run after. Now the download carries on."""
    good = _model(tmp_path / "good")
    entry = setup.manifest_entry(good, setup.REPO_ID)
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"wan": entry}))
    base = tmp_path / "models" / "Wan2.1-VACE-1.3B-diffusers"
    (base / "transformer").mkdir(parents=True)
    for rel in ("transformer/config.json", "model_index.json"):    # what came in before it stopped
        (base / rel).write_bytes((good / rel).read_bytes())
        meta = base / ".cache" / "huggingface" / "download" / (rel + ".metadata")
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text("a" * 40 + "\netag\n123.0\n")
    assert setup.stopped_download(base, entry) == ["missing vae.bin"]
    assert 0 < setup.download_gb(base, entry) < setup.MODEL_GB
    _ready_machine(monkeypatch)
    asked = []

    def finish(dest, repo_id=setup.REPO_ID, revision=None):
        asked.append(revision)
        (dest / "vae.bin").write_bytes((good / "vae.bin").read_bytes())

    monkeypatch.setattr(setup, "download_model", finish)
    assert setup.main(["--base", str(base), "--manifest", str(man), "--json", "--skip-packages"]) == 0
    out = capsys.readouterr().out
    assert asked == ["a" * 40] and "carries on" in out and "still to come" in out
    assert json.loads(out.strip().splitlines()[-1])["ok"] is True
    # Whole now: nothing to download the next time.
    asked.clear()
    assert setup.main(["--base", str(base), "--manifest", str(man), "--json", "--skip-packages"]) == 0
    assert asked == []


def test_a_model_of_another_size_is_not_downloaded_over(tmp_path, monkeypatch, capsys):
    """A folder that is not a stopped download of the pinned model (a file of another size) is refused, and the
    refusal says what to do; nothing is downloaded into it."""
    good = _model(tmp_path / "good")
    entry = setup.manifest_entry(good, setup.REPO_ID)
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"wan": entry}))
    base = _model(tmp_path / "other")
    (base / "vae.bin").write_bytes(b"\x01" * 999)
    assert setup.stopped_download(base, entry) == []
    _ready_machine(monkeypatch)
    monkeypatch.setattr(setup, "download_model", lambda *a, **k: pytest.fail("downloaded into another model"))
    assert setup.main(["--base", str(base), "--manifest", str(man), "--json", "--skip-packages"]) == 1
    done = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "does not match" in done["error"] and "Move" in done["error"] and "aside" in done["error"]


def test_a_stopped_download_of_another_revision_is_not_carried_on(tmp_path):
    good = _model(tmp_path / "good")
    entry = setup.manifest_entry(good, setup.REPO_ID)
    base = tmp_path / "base"
    (base / "transformer").mkdir(parents=True)
    (base / "transformer" / "config.json").write_bytes((good / "transformer" / "config.json").read_bytes())
    meta = base / ".cache" / "huggingface" / "download" / "transformer" / "config.json.metadata"
    meta.parent.mkdir(parents=True)
    meta.write_text("e" * 40 + "\netag\n123.0\n")
    assert setup.stopped_download(base, entry) == []


def test_a_setup_that_breaks_still_says_why(tmp_path, monkeypatch, capsys):
    """Anything that stops the setup ends with a done event the app shows (it was a traceback, and the app's
    status line kept saying the step before it)."""
    def broken(base):
        raise PermissionError("[WinError 5] Access is denied: 'D:\\\\models'")

    monkeypatch.setattr(setup, "machine", broken)
    assert setup.main(["--base", str(tmp_path / "wan"), "--json"]) == 1
    captured = capsys.readouterr()
    done = json.loads(captured.out.strip().splitlines()[-1])
    assert done["event"] == "done" and not done["ok"] and "PermissionError" in done["error"]
    assert "Traceback" in captured.err


def test_no_gpu_names_the_driver_or_the_runtime_pytorch_needs():
    driver = setup.no_gpu_reason({"torch": "2.9.0+cu128", "device": "cpu", "memory_gb": None,
                                  "nvidia": [{"name": "NVIDIA GeForce RTX 4080", "memory_gb": 16.0, "driver": "552.44"}],
                                  "cuda_warning": "CUDA initialization: The NVIDIA driver on your system is too old"})
    assert "RTX 4080" in driver and "552.44" in driver and "570" in driver and "too old" in driver
    dll = setup.no_gpu_reason({"torch": None, "device": "cpu", "memory_gb": None,
                               "torch_error": 'OSError: [WinError 126] Error loading "fbgemm.dll"'})
    assert "does not load" in dll and "vc_redist.x64.exe" in dll
    dll_c = setup.no_gpu_reason({"torch": None, "device": "cpu", "memory_gb": None,
                                 "torch_error": "ImportError: DLL load failed while importing _C: The specified module"})
    assert "vc_redist.x64.exe" in dll_c
    silent = setup.no_gpu_reason({"torch": "2.9.0+cu128", "device": "cpu", "memory_gb": None, "nvidia": [],
                                  "nvidia_error": "NVIDIA-SMI has failed because it couldn't communicate with the driver"})
    assert "does not answer" in silent
    small = setup.no_gpu_reason({"torch": "2.9.0+cu128", "device": "cuda", "memory_gb": 12.0, "gpu": "RTX 4070"})
    assert "RTX 4070 has 12.0 GB" in small and "needs 16 GB" in small


def test_nvidia_smi_is_read(monkeypatch):
    import subprocess
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 0, stdout="NVIDIA GeForce RTX 4080, 16376, 566.36\n", stderr=""))
    assert setup.nvidia_smi() == ([{"name": "NVIDIA GeForce RTX 4080", "memory_gb": 16.0, "driver": "566.36"}], None)
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 9, stdout="NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n", stderr=""))
    gpus, error = setup.nvidia_smi()
    assert gpus == [] and "couldn't communicate" in error
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    if setup.sys.platform != "win32":
        assert setup.nvidia_smi() == (None, None)


def test_a_pytorch_that_is_there_and_does_not_load_is_not_called_missing(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def broken(name, *a, **k):
        if name == "torch":
            raise OSError('[WinError 126] The specified module could not be found. Error loading "fbgemm.dll"')
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", broken)
    m = setup.machine(tmp_path)
    assert m["torch"] is None and "WinError 126" in m["torch_error"]
    assert "Visual C++" in setup.no_gpu_reason(m)

    def absent(name, *a, **k):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", absent)
    m = setup.machine(tmp_path)
    assert m["torch"] is None and "torch_error" not in m
    assert "PyTorch is not installed" in setup.no_gpu_reason(m)


def test_only_the_missing_packages_are_installed(monkeypatch):
    ran = []

    class Proc:
        stdout = iter(["Successfully installed opencv-python-headless\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(setup.subprocess, "Popen", lambda cmd, **k: ran.append(cmd) or Proc())
    assert setup.install_packages(["cv2"])
    assert ran[0][-1] == "opencv-python-headless>=4.9" and "diffusers>=0.35" not in ran[0]
    assert "cv2" in setup.IMPORTS and "numpy" in setup.IMPORTS
