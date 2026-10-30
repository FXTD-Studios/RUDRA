"""tools/make_models_json.py: the models.json the app downloads from (29 Oct 2026)."""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import make_models_json as mm  # noqa: E402


def _model(tmp_path: Path) -> Path:
    d = tmp_path / "aces_v2_demo"
    (d / "sub").mkdir(parents=True)
    (d / "manifest.json").write_text('{"name": "Demo"}', encoding="utf-8")
    (d / "sub" / "w.bin").write_bytes(b"x" * 4096)
    (d / ".rudra-download.json").write_text("{}", encoding="utf-8")   # a download marker is not published
    (d / "sub" / "w.bin.part").write_bytes(b"y")                       # nor a half download
    return d


def test_every_file_with_its_size_checksum_and_hub_url(tmp_path):
    out = tmp_path / "models.json"
    folder = _model(tmp_path)
    assert mm.main(["--repo", "fxtdstudios/RUDRA", "--revision", "v1.0.0", "--out", str(out),
                    "--model", str(folder), "--channel", "preview", "--kind", "aces_v2", "--licence", "RUDRA model terms"]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["version"] == 1
    (m,) = doc["models"]
    assert m["id"] == "aces_v2_demo" and m["title"] == "Demo" and m["folder"] == "aces_v2_demo"
    assert [f["path"] for f in m["files"]] == ["manifest.json", "sub/w.bin"]
    w = m["files"][1]
    assert w["size"] == 4096
    assert w["sha256"] == hashlib.sha256(b"x" * 4096).hexdigest()
    assert w["url"] == "https://huggingface.co/fxtdstudios/RUDRA/resolve/v1.0.0/models/aces_v2_demo/sub/w.bin"
    # What the app accepts (model_downloads.cpp, parse_model_index).
    for f in m["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["url"].startswith("https://")
        assert not f["path"].startswith("/") and ".." not in f["path"].split("/") and "\\" not in f["path"]


def test_merge_keeps_the_other_models(tmp_path):
    out = tmp_path / "models.json"
    out.write_text(json.dumps({"version": 1, "models": [{"id": "old", "files": []}, {"id": "aces_v2_demo", "files": []}]}))
    mm.main(["--repo", "r/x", "--out", str(out), "--merge", "--model", str(_model(tmp_path))])
    ids = [m["id"] for m in json.loads(out.read_text())["models"]]
    assert ids == ["old", "aces_v2_demo"]
