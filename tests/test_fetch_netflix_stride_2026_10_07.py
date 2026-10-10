"""A second pull of the Netflix pairs with --train-stride 2 --train-offset 1
selects every odd frame: disjoint from the stride-4 corpus of 1 Oct 2026, and
val/test untouched."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline.fetch_netflix_pairs as f  # noqa: E402


def _shots():
    names = [f"frame{i:08d}.j2k" for i in range(40)]
    idx = {n: {"offset": i} for i, n in enumerate(names)}
    return {("nocturne", "1-40"): {"HDR": "h", "SDR": "s"}, ("meridian", "1-40"): {"HDR": "h2", "SDR": "s2"}}, idx


def test_second_pull_is_disjoint(monkeypatch):
    shots, idx = _shots()
    monkeypatch.setattr(f, "zip_index", lambda url: idx)
    first = f.plan(shots)
    second = f.plan(shots, train_stride=2, train_offset=1)
    a = {j["asset_id"] for j in first if j["split"] == "train"}
    b = {j["asset_id"] for j in second if j["split"] == "train"}
    assert len(a) == 10 and len(b) == 20 and not (a & b)
    # Every frame is covered once between the two pulls, every odd one and every 4k+2.
    frames = sorted(int(x.rsplit("frame", 1)[1]) for x in a | b)
    assert frames == sorted([i for i in range(40) if i % 2 == 1] + [i for i in range(2, 40, 4)])
    # Test split: stride 10 either way, the same frames.
    t1 = {j["asset_id"] for j in first if j["split"] == "test"}
    t2 = {j["asset_id"] for j in second if j["split"] == "test"}
    assert t1 == t2 and len(t1) == 4
