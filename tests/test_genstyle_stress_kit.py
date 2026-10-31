import hashlib
import json

import cv2
import numpy as np
import pytest

from training.genstyle_stress_kit import build


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _manifest(tmp_path):
    rows = []
    for i, split in enumerate(["val", "val", "train"]):
        img = (np.random.default_rng(i).random((32, 40, 3)) * 65535).astype(np.uint16)
        p = tmp_path / f"s{i}.png"; cv2.imwrite(str(p), img)
        rows.append(dict(asset_id=f"a/{i}", scene_id=f"sc{i}", split=split, sdr_path=str(p),
                         sdr_sha256=_sha(p), hdr_path="h", hdr_sha256="x"))
    m = tmp_path / "m.jsonl"; m.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return m


def test_build_is_deterministic_and_val_only(tmp_path):
    m = _manifest(tmp_path)
    r1 = build(m, tmp_path / "k1"); r2 = build(m, tmp_path / "k2")
    a = [json.loads(l) for l in (tmp_path / "k1/manifest.jsonl").read_text().splitlines()]
    b = [json.loads(l) for l in (tmp_path / "k2/manifest.jsonl").read_text().splitlines()]
    assert r1["frames"] == 2 and [x["sdr_sha256"] for x in a] == [x["sdr_sha256"] for x in b]
    assert all(x["split"] == "val" and x["hdr_path"] == "h" for x in a)
    assert all(_sha(tmp_path / "k1/sdr" / x["sdr_path"].split("sdr")[-1].lstrip("\\/")) == x["sdr_sha256"] for x in a)
    with pytest.raises(ValueError):
        build(m, tmp_path / "k1")


def test_refuses_changed_source(tmp_path):
    m = _manifest(tmp_path)
    (tmp_path / "s0.png").write_bytes(b"changed")
    with pytest.raises(ValueError):
        build(m, tmp_path / "k")
