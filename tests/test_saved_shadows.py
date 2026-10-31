import numpy as np
from tools.audit_saved_shadows import shadow_metrics


def test_colour_blob_detected_and_target_identity_not_penalized():
    source = np.zeros((8, 9, 3), np.float32)
    target = source.copy()
    blob = source.copy()
    blob[..., 2] = .001
    bad = shadow_metrics(blob, target, source)['5']
    assert bad['pixels'] == 72
    assert bad['reference_abs_error_gt10_fraction'] == 1
    assert bad['signed_reference_rmse'] > 0
    good = shadow_metrics(blob, blob, source)['5']
    assert good['reference_abs_error_gt10_fraction'] == 0
    assert good['signed_reference_rmse'] == 0


def test_no_dark_pixels_are_missing_evidence():
    source = np.ones((8, 9, 3), np.float32)
    assert shadow_metrics(source, source, source)['5'] == {'pixels': 0}
