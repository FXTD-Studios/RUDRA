"""tools/update_models_index.py (7 Nov 2026): a published index changed in place."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from update_models_index import main, update  # noqa: E402

URL = "https://huggingface.co/fxtdstudios/rudra-models/resolve/{rev}/models/rudra_genfill_v3/{p}"


def published():
    return {"version": 1, "models": [
        {"id": "rudra_aces_2", "title": "RUDRA ACES 2", "version": "1.0", "channel": "release", "kind": "aces_v2",
         "summary": "s", "licence": "Apache 2.0", "licence_url": "", "folder": "RUDRA_ACES_2", "min_app": "1.0.0",
         "files": [{"path": "rudra_g1.ts", "url": "https://huggingface.co/fxtdstudios/rudra-models/resolve/v1.0.0/models/RUDRA_ACES_2/rudra_g1.ts",
                    "size": 1, "sha256": "a" * 64}]},
        {"id": "rudra_genfill_v3", "title": "AI fill v3 (experimental)", "version": "3", "channel": "preview",
         "kind": "genfill", "summary": "g", "licence": "Apache 2.0", "licence_url": "", "folder": "rudra_genfill_v3",
         "min_app": "1.0.0", "files": [{"path": "rudra_genfill.safetensors", "url": URL.format(rev="v1.0.0", p="rudra_genfill.safetensors"),
                                        "size": 3, "sha256": "b" * 64}]},
    ]}


def test_the_channel_changes_and_nothing_else():
    old = published()
    new = update(json.loads(json.dumps(old)), {"rudra_aces_2": "preview"}, {}, None)
    assert new["models"][0]["channel"] == "preview"
    new["models"][0]["channel"] = "release"
    assert new == old


def test_a_refresh_lists_the_new_files_at_the_same_place(tmp_path):
    folder = tmp_path / "rudra_genfill_v3"
    (folder / "prompt_cache").mkdir(parents=True)
    (folder / "rudra_genfill.safetensors").write_bytes(b"abc")
    (folder / "prompt_cache" / "k1.pt").write_bytes(b"x" * 10)
    new = update(published(), {}, {"rudra_genfill_v3": folder}, "v1.1.0")
    g = new["models"][1]
    assert [f["path"] for f in g["files"]] == ["prompt_cache/k1.pt", "rudra_genfill.safetensors"]
    assert g["files"][0]["url"] == URL.format(rev="v1.1.0", p="prompt_cache/k1.pt")
    assert (g["id"], g["version"], g["channel"], g["min_app"]) == ("rudra_genfill_v3", "3", "preview", "1.0.0")
    assert new["models"][0] == published()["models"][0]


def test_refusals(tmp_path):
    with pytest.raises(SystemExit):
        update(published(), {"nope": "preview"}, {}, None)
    with pytest.raises(SystemExit):
        update(published(), {"rudra_aces_2": "beta"}, {}, None)
    wrong = tmp_path / "other_name"
    wrong.mkdir()
    (wrong / "x").write_bytes(b"1")
    with pytest.raises(SystemExit):
        update(published(), {}, {"rudra_genfill_v3": wrong}, None)
    lacking = tmp_path / "rudra_genfill_v3"
    lacking.mkdir()
    (lacking / "brightness.json").write_text("{}")
    with pytest.raises(SystemExit):                   # a published file missing from the folder
        update(published(), {}, {"rudra_genfill_v3": lacking}, None)


def test_the_command_line(tmp_path):
    src = tmp_path / "in.json"
    src.write_text(json.dumps(published()))
    out = tmp_path / "out.json"
    assert main(["--in", str(src), "--out", str(out), "--channel", "rudra_aces_2=preview"]) == 0
    assert json.loads(out.read_text())["models"][0]["channel"] == "preview"
