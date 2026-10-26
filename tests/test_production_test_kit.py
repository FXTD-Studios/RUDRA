import json

import pytest

from training import production_test_kit as tk
from training.production_eval_kit import digest


def rows(tmp_path):
    out = []
    for i, split in enumerate(["train", "val", "test", "test", "test"]):
        for kind in ("sdr", "hdr"):
            (tmp_path / f"{kind}{i}.png").write_bytes(f"{kind}{i}".encode())
        out.append(dict(asset_id=f"a{i}", scene_id=f"s{min(i, 3)}", split=split,
                        sdr_path=str(tmp_path / f"sdr{i}.png"), hdr_path=str(tmp_path / f"hdr{i}.png"),
                        sdr_sha256=digest(tmp_path / f"sdr{i}.png"), hdr_sha256=digest(tmp_path / f"hdr{i}.png")))
    m = tmp_path / "m.jsonl"
    m.write_text("".join(json.dumps(r) + "\n" for r in out))
    return m


def test_prepare_takes_test_rows_only_and_is_immutable(tmp_path):
    m = rows(tmp_path)
    c = tk.prepare(m, tmp_path / "kit")
    assert c["frames"] == 3 and c["scenes"] == 2 and c["one_candidate_only"]
    with pytest.raises(ValueError, match="immutable"):
        tk.prepare(m, tmp_path / "kit")


def test_second_candidate_refused(tmp_path, monkeypatch):
    m = rows(tmp_path)
    kit = tmp_path / "kit"
    tk.prepare(m, kit)
    (kit / "used_by.json").write_text(json.dumps({"label": "x", "checkpoint_sha256": "ab" * 32}))
    import training.freeze_candidate as fc
    monkeypatch.setattr(fc, "verify", lambda p: {"ok": True, "problems": []})
    f = tmp_path / "freeze.json"
    f.write_text(json.dumps({"label": "y", "checkpoint": "c.pt", "checkpoint_sha256": "cd" * 32}))
    with pytest.raises(ValueError, match="already used"):
        tk.evaluate(kit, f, tmp_path / "out", run=lambda *a, **k: None)


def test_overlapping_test_scene_rejected(tmp_path):
    m = rows(tmp_path)
    data = [json.loads(l) for l in m.read_text().splitlines()]
    data[1]["scene_id"] = "s3"
    m.write_text("".join(json.dumps(r) + "\n" for r in data))
    with pytest.raises(ValueError, match="overlaps"):
        tk.prepare(m, tmp_path / "kit")
