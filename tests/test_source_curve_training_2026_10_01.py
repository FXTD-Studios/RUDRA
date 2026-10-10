"""Task 1.4 wiring: source-curve labels from manifest to model, honestly.

A label is only true while the SDR still has that curve, so degraded samples,
real SDR and unlabelled rows all train as unknown, and a fraction of the
clean labelled ones are dropped to unknown to keep the blind path trained.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from pipeline.hdr_io import HDRStorage, encode_hdr_u16  # noqa: E402
from rudra.sdr2hdr import SDR2HDRNet, source_curve_index  # noqa: E402
from training.infer_sdr2hdr import predict_image  # noqa: E402
from training.sdr2hdr_dataset import SDRHDRDataset, effective_source_curve  # noqa: E402

HABLE = source_curve_index("hable")


def test_label_rule():
    rec = {"source_curve": "hable", "sdr_kind": "rendered"}
    assert effective_source_curve(rec, degraded=False, augment=False, dropout=0.9) == HABLE
    assert effective_source_curve(rec, degraded=True, augment=False, dropout=0.0) == 0
    assert effective_source_curve({"source_curve": "unknown", "sdr_kind": "real"}, False, True, 0.0) == 0
    assert effective_source_curve({"sdr_kind": "real", "source_curve": "hable"}, False, False, 0.0) == 0
    assert effective_source_curve({}, False, True, 0.0) == 0          # pre-1.2 manifests
    rng = random.Random(0)
    kept = sum(effective_source_curve(rec, False, True, 0.3, rng) == HABLE for _ in range(4000))
    assert 0.66 < kept / 4000 < 0.74                                     # 30% dropout, train only


def _corpus(root: Path, n_train: int = 6, n_val: int = 2) -> Path:
    storage = HDRStorage(mode="log2_extended")
    (root / "sdr").mkdir(parents=True)
    (root / "hdr").mkdir()
    (root / "_ingest_config.json").write_text(json.dumps({"storage": storage.as_dict()}))
    rng = np.random.default_rng(0)
    curves = ["hable", "agx", "aces"]
    rows = []
    for i in range(n_train + n_val):
        lin = rng.uniform(0.0, 3.0, size=(40, 48, 3)).astype(np.float32)
        sdr = (np.clip(lin / (1.0 + lin), 0, 1) ** (1 / 2.2) * 65535).astype(np.uint16)
        hdr, _ = encode_hdr_u16(lin, storage)
        sp, hp = root / "sdr" / f"{i}.png", root / "hdr" / f"{i}.png"
        cv2.imwrite(str(sp), sdr)
        cv2.imwrite(str(hp), hdr)
        real = i == 1
        rows.append({"asset_id": f"a{i}", "scene_id": f"s{i}",
                     "split": "train" if i < n_train else "val",
                     "sdr_path": str(sp), "hdr_path": str(hp), "tonemap_ev": None if real else 0.0,
                     "sdr_kind": "real" if real else "rendered",
                     "source_curve": "unknown" if real else curves[i % 3]})
    manifest = root / "m.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifest


def test_dataset_emits_labels(tmp_path):
    m = _corpus(tmp_path)
    clean = SDRHDRDataset(m, split="train", crop_size=32, augment=False)
    assert [int(clean[i]["source_curve"]) for i in range(3)] == [source_curve_index("hable"), 0,
                                                                 source_curve_index("aces")]
    hard = SDRHDRDataset(m, split="train", crop_size=32, augment=False, deterministic_degradation=True)
    assert all(int(hard[i]["source_curve"]) == 0 for i in range(len(hard)))


def _train(argv, monkeypatch):
    from training import train_sdr2hdr
    monkeypatch.setattr(sys, "argv", ["train_sdr2hdr.py", *argv])
    return train_sdr2hdr.train(train_sdr2hdr.parse_args())


def test_end_to_end_warm_start_from_a_blind_curve_head(tmp_path, monkeypatch):
    m = _corpus(tmp_path / "c")
    common = ["--manifest", str(m), "--device", "cpu", "--workers", "0", "--batch-size", "2",
              "--crop-size", "32", "--base-channels", "8", "--curve-head", "--eval-every", "2",
              "--eval-batches", "1", "--log-every", "1", "--save-every", "2"]
    blind = _train(common + ["--steps", "2", "--output-dir", str(tmp_path / "blind")], monkeypatch)
    aware = _train(common + ["--steps", "2", "--source-curve", "--init-checkpoint", str(blind),
                             "--output-dir", str(tmp_path / "aware")], monkeypatch)
    from training.train_sdr2hdr import load_image_checkpoint
    model, checkpoint = load_image_checkpoint(aware, torch.device("cpu"))
    assert checkpoint["config"]["source_curve"] is True and model.source_curve
    assert checkpoint["config"]["source_curve_dropout"] == 0.3
    x = torch.rand(1, 3, 40, 48)
    a = predict_image(model, x, True, 0, 16, bf16=False, source_curve=HABLE)
    b = predict_image(model, x, True, 0, 16, bf16=False)
    assert a.shape == b.shape and torch.isfinite(a).all()
    # the blind checkpoint refuses a known curve rather than ignoring it
    blind_model, _ = load_image_checkpoint(blind, torch.device("cpu"))
    with pytest.raises(ValueError):
        predict_image(blind_model, x, True, 0, 16, bf16=False, source_curve=HABLE)


def test_source_curve_needs_labels_and_the_head(tmp_path, monkeypatch):
    m = _corpus(tmp_path / "c")
    base = ["--manifest", str(m), "--device", "cpu", "--workers", "0", "--steps", "1",
            "--output-dir", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="--curve-head"):
        _train(base + ["--source-curve"], monkeypatch)
    rows = [json.loads(l) for l in m.read_text().splitlines()]
    for r in rows:
        r["source_curve"] = "unknown"
    m.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="known source_curve"):
        _train(base + ["--source-curve", "--curve-head"], monkeypatch)
