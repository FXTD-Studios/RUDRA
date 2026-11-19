"""1.2.6 (readiness report, checklist item 7): the AI fill's offline install and hashed, pinned models."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

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
