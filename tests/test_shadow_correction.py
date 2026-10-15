"""BT.1886 shadows (14 Oct 2026): an opt-in Source picker entry, held to native."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rudra.source_baseline import (BASELINE_CURVES, DEFAULT_BASELINE, baseline_curve_index,  # noqa: E402
                                   baseline_knots, knots_table)


def test_appended_last_and_not_the_default():
    assert BASELINE_CURVES[:7] == ("unknown", "bt1886", "aces", "hable", "agx", "camera_log", "clip")
    assert baseline_curve_index("bt1886_shadows") == 7
    assert DEFAULT_BASELINE == "bt1886"
    assert knots_table().shape == (8, 256)


def test_bt1886_from_code_40_and_lifted_below():
    plain, lifted = baseline_knots("bt1886"), baseline_knots("bt1886_shadows")
    assert np.array_equal(plain[1 + 40:], lifted[1 + 40:])
    assert (lifted[1 + 1:1 + 40] > plain[1 + 1:1 + 40]).all()


def test_the_scored_path_is_the_models_path():
    """tools/shadow_correction_eval.py --app scored exactly what the picker baseline computes."""
    from rudra.sdr2hdr import SDR2HDRNet
    from tools.shadow_correction_eval import app_baseline, knots_for
    from rudra.source_baseline import _inverse
    from rudra.sdr2hdr import srgb_to_linear
    x = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(3)) * 0.25   # mostly shadows
    m = SDR2HDRNet(corpus_ev=0.0, baseline_source=True)
    got = m.picker_baseline(x, baseline_curve_index("bt1886_shadows"))[0].permute(1, 2, 0).double().numpy() * 10000.0
    display = srgb_to_linear(torch.arange(256, dtype=torch.float32) / 255.0).numpy()
    want = app_baseline(x[0].permute(1, 2, 0).double().numpy(), knots_for(_inverse("bt1886_shadows", display).astype(np.float64)))
    assert np.allclose(got, want, rtol=2e-3, atol=1e-5)
