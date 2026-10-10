"""rudra.inference is the one tiler (refactor, 9 Oct 2026).

`rudra video` carried its own copy of predict_image's tiling until 9 Oct
2026; the per-tile curve found in review was that copy drifting. These pin
the move: one implementation, the old import path still works, and nothing
the installed package ships imports training/ (which it does not ship).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def test_training_reexports_the_same_objects():
    import rudra.inference as inference
    import training.infer_sdr2hdr as legacy
    for name in ("predict_image", "predict_fields", "_tile_starts", "_tile_weight"):
        assert getattr(legacy, name) is getattr(inference, name), name


@pytest.mark.parametrize("path", ["rudra/video.py", "ui/server.py", "rudra/inference.py"])
def test_shipped_code_does_not_import_training(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("training"), (path, node.module)
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("training") for a in node.names), path


@pytest.fixture(scope="module")
def shadow_model():
    from training.infer_sdr2hdr import load_models
    torch.set_num_threads(4)
    return load_models(str(ROOT / "checkpoints" / "sdr2hdr_shadow_v1.pt"), None,
                       torch.device("cpu"))[0]


def _frame(h=300, w=520):
    rng = np.random.default_rng(1)
    img = rng.uniform(0.0, 1.0, size=(h, w, 3)).astype(np.float32)
    img[40:90, 100:300] = 1.0
    return torch.from_numpy(img).permute(2, 0, 1)[None]


@pytest.mark.parametrize("tile", [0, 256])
def test_default_shadow_weight_is_the_predicted_one(shadow_model, tile):
    """shadow_weight=None must keep the old behaviour exactly."""
    from rudra.inference import predict_image
    x = _frame()
    predicted = shadow_model.predict_shadow_weight(x)
    a = predict_image(shadow_model, x, True, tile, 32, bf16=False)
    b = predict_image(shadow_model, x, True, tile, 32, bf16=False, shadow_weight=predicted)
    assert torch.equal(a, b)


def test_an_explicit_shadow_weight_is_used(shadow_model):
    from rudra.inference import predict_image
    x = _frame()
    off = predict_image(shadow_model, x, True, 256, 32, bf16=False, shadow_weight=0.0)
    on = predict_image(shadow_model, x, True, 256, 32, bf16=False, shadow_weight=1.0)
    assert not torch.equal(off, on)


def test_video_predictor_is_predict_image(tmp_path):
    """`rudra video` frame = predict_image in fp32 with the smoothed weight."""
    from rudra.inference import predict_image
    from rudra.video import Predictor, ShadowSmoother
    p = Predictor(ROOT / "checkpoints" / "sdr2hdr_shadow_v1.pt", "cpu", 256, 32)
    rgb = _frame()[0].permute(1, 2, 0).numpy()
    hdr, weight, _ = p.predict(rgb, dict(transfer="srgb", primaries="rec2020"),
                               ShadowSmoother(0.0, 0.15))
    # Built exactly as Predictor builds it. The CPU conv picks its kernel from
    # the tensor's strides, so the same values in another layout differ in the
    # last bits (~3e-5) without anything being wrong.
    from rudra.sdr2hdr import canonicalize_sdr
    x = canonicalize_sdr(torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None], "srgb", "full")
    want = predict_image(p.model, x, True, 256, 32, bf16=False, shadow_weight=weight)
    np.testing.assert_array_equal(hdr, want[0].permute(1, 2, 0).numpy())
