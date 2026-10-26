import pytest

from tools.prospective_checkpoint_gate import GRADE_LIMITS, require_checks, require_grade_numbers


def test_relaxed_zero_ev_cannot_pass_original_numeric_rule():
    summary = {'grades': {ev: {'constrained': limit} for ev, limit in GRADE_LIMITS.items()}}
    require_grade_numbers(summary)
    summary['grades']['0']['constrained'] = .0833
    with pytest.raises(ValueError, match='Original grade0'):
        require_grade_numbers(summary)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), None])
def test_nonfinite_or_missing_measurement_is_not_a_pass(bad):
    summary = {'grades': {ev: {'constrained': limit} for ev, limit in GRADE_LIMITS.items()}}
    summary['grades']['0']['constrained'] = bad
    with pytest.raises(ValueError):
        require_grade_numbers(summary)


@pytest.mark.parametrize('checks', [{}, {'preserved': False}, {'preserved': 1}])
def test_absent_failed_or_nonboolean_evidence_refused(checks):
    with pytest.raises(ValueError):
        require_checks(checks, {'preserved'}, 'preservation')
