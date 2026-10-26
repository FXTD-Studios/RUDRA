from tools.evaluate_run05_dequant import summarize
from tools.compare_run05_dequant import compare
import copy
import pytest


def test_strata_equal_pair_weighting_and_missing_values():
    rows = []
    for scene, codec, score in [('a', 'none', 2), ('a', 'x264', 4), ('b', 'x264', None)]:
        rows.append({'scene_id': scene, 'codec': codec, 'methods': {'model': {
            'contract': {'r_within_1_code': 1, 'h_abs_log2': score},
            'reference': {'pushes': [dict(push=p, code_rmse=score,
                                        gradient_rmse=score, curvature_rmse=score)
                                     for p in range(-3, 4)]}}}})
    summary = summarize(rows)
    assert summary['all']['scenes'] == 2
    assert summary['all']['pairs'] == 3
    assert summary['all']['methods']['model']['pushes']['0']['gradient_rmse'] == {'mean': 3, 'pairs': 2}
    assert summary['none']['methods']['model']['pushes']['3']['code_rmse']['mean'] == 2
    assert summary['codec']['methods']['model']['contract']['h_abs_log2'] == {'mean': 4, 'pairs': 1}
    assert summary['none']['methods']['model']['contract']['s_visible_lift'] == {'mean': None, 'pairs': 0}


def report(score):
    rows = []
    for i in range(24):
        metrics = {'contract': {'r_within_1_code': 1, 'h_abs_log2': 1, 's_visible_lift': 0},
                   'reference': {'pushes': [dict(push=p, reference_gentle_pixels=1000,
                       code_rmse=score, gradient_rmse=score, curvature_rmse=score) for p in range(-3, 4)]}}
        rows.append({'asset_id': str(i), 'scene_id': str(i // 3), 'codec': 'none' if i % 2 else 'x264',
                     'methods': {'model': metrics, 'baseline_bt1886': {'fixed': True}}})
    # Baseline entries need full metrics for the shared summarizer.
    for row in rows:
        row['methods']['baseline_bt1886'] = copy.deepcopy(row['methods']['model'])
        for p in row['methods']['baseline_bt1886']['reference']['pushes']:
            p.update(code_rmse=3, gradient_rmse=3, curvature_rmse=3)
    return dict(manifest_sha256='same', view='bt1886', policy='observed_clip_v1', device='cpu',
                label='model', pairs=24, per_pair=rows, summary=summarize(rows))


def test_trial_requires_improvement_and_complete_matching_evidence():
    control, auxiliary = report(2), report(1.8)
    result = compare(control, auxiliary)
    assert result['quality_trial_pass'] and not result['release_authorized']
    assert not compare(control, control)['quality_trial_pass']
    auxiliary['per_pair'][0]['methods']['model']['contract']['r_within_1_code'] = .98
    assert not compare(control, auxiliary)['quality_trial_pass']
    auxiliary['per_pair'][1]['methods']['model']['reference']['pushes'][0]['reference_gentle_pixels'] = 999
    with pytest.raises(ValueError, match='mask mismatch'):
        compare(control, auxiliary)
