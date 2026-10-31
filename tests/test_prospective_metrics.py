import numpy as np
import pytest
from tools.prospective_metrics import measure, aggregate


class LinearView:
    def forward(self, rgb):
        return rgb.copy()


def test_raw_reference_grade_is_not_replaced_by_analytic_target():
    truth = np.ones((16, 16, 3), np.float32)
    regions = {k: np.ones((16, 16), bool) for k in ('highlight', 'partial', 'fully_clipped')}
    values = measure({'candidate': truth, 'analytic': truth * .5}, truth, regions, LinearView())
    assert values['candidate']['highlight_asinh_mse'] == 0
    assert values['candidate']['grade_0_mse'] == 0
    assert values['analytic']['grade_0_mse'] == pytest.approx((255 * .5)**2)
    assert values['analytic']['grade_3_mse'] == pytest.approx((255 * 4)**2)


def test_empty_support_is_not_zero_error():
    truth = np.ones((16, 16, 3), np.float32)
    regions = {k: np.zeros((16, 16), bool) for k in ('highlight', 'partial', 'fully_clipped')}
    values = measure({'candidate': truth, 'analytic': truth}, truth, regions, LinearView())
    assert all(v is None for method in values.values() for v in method.values())
    rows = [{'case': {'scene_id': 'a', 'source_bits': 8}, 'metrics': values}]
    summaries = aggregate(rows, ['a'], [8])
    assert all(not v['no_worse_analytic'] and v['supporting_cases'] == 0 for v in summaries['a/bits8'].values())


def row(scene, bits, candidate, analytic=4):
    return {'case': {'scene_id': scene, 'source_bits': bits},
            'metrics': {'candidate': {'m': candidate}, 'analytic': {'m': analytic}}}


def test_per_scene_and_precision_do_not_hide_regression_in_pooled_score():
    rows = [row('a', 8, 0), row('b', 8, 6), row('b', 10, 9), row('a', 8, None, None)]
    groups = aggregate(rows, ['a', 'b'], [8, 10])
    assert groups['pooled/bits8']['m']['candidate'] == 3
    assert groups['pooled/bits8']['m']['supporting_cases'] == 2
    assert groups['pooled/bits8']['m']['total_cases'] == 3
    assert groups['pooled/bits8']['m']['no_worse_analytic']
    assert not groups['b/bits8']['m']['no_worse_analytic']
    assert groups['b/bits10']['m']['candidate'] == 9


@pytest.mark.parametrize('candidate,analytic', [(None, 1), (float('nan'), 1), (-1, 1)])
def test_invalid_or_unmatched_results_are_rejected(candidate, analytic):
    with pytest.raises(ValueError):
        aggregate([row('a', 8, candidate, analytic)], ['a'], [8])
