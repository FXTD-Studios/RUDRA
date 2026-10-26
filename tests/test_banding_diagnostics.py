import numpy as np
from pipeline.run05_corpus import canonical, declared_inverse
from training.contract_checks import headroom
from training.banding_diagnostics import reference_gradient_errors


def ramp():
    src = np.broadcast_to(np.linspace(.18, .42, 512, dtype=np.float32)[None, :, None], (128, 512, 3)).copy()
    return src


def hdr(src):
    return declared_inverse(canonical(src), 'bt1886')


def test_noise_can_fool_flat_step_score_but_worsens_reference_errors():
    src = ramp()
    q = np.rint(src * 255) / 255
    noisy = q + np.random.default_rng(12).uniform(-.15, .15, q.shape).astype(np.float32) / 255
    truth, quantized, noise = hdr(src), hdr(q), hdr(noisy)
    assert headroom(quantized, truth)['banding_excess_worst'] > .8
    assert headroom(noise, truth)['banding_excess_worst'] < .05
    original = reference_gradient_errors(quantized, truth)
    perturbed = reference_gradient_errors(noise, truth)
    perfect = reference_gradient_errors(truth, truth)
    for a, b, c in zip(original['pushes'], perturbed['pushes'], perfect['pushes']):
        assert a['reference_gentle_pixels'] == b['reference_gentle_pixels']
        if a['code_rmse'] is not None:
            for metric in ('code_rmse', 'gradient_rmse', 'curvature_rmse'):
                assert b[metric] > a[metric]
                assert c[metric] == 0


def test_no_measurable_gradient_reports_missing_not_success():
    flat = np.ones((32, 32, 3), np.float32) * .002
    result = reference_gradient_errors(flat, flat)
    assert result['measured_pushes'] == 0
    assert all(r['code_rmse'] is None for r in result['pushes'])
