from tests.test_dequant_audit import report
from tools.compare_shadow_repair import compare_repair


def test_improving_bad_checkpoint_is_insufficient():
    result = compare_repair(report(3), report(2.5), report(2))
    assert result['paired_comparison']['quality_trial_pass']
    assert not result['numerical_trial_pass']
    assert not result['checks']['no_regression_vs_original_overall']


def test_small_nonzero_shadow_lift_still_fails_recovery_gate():
    treatment = report(1.8)
    treatment['per_pair'][0]['methods']['model']['contract']['s_visible_lift'] = .00001
    result = compare_repair(report(2), treatment, report(2))
    assert not result['checks']['zero_visible_shadow_lift']
    assert not result['numerical_trial_pass']


def test_numeric_pass_still_requires_visual_and_release_evidence():
    result = compare_repair(report(2), report(1.8), report(1.9))
    assert result['numerical_trial_pass']
    assert result['visual_review_required']
    assert not result['release_authorized']
