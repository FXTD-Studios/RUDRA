"""Bind a prospective evaluation to passing original DEV evidence, not relaxed limits."""
import hashlib
import json
import math
from pathlib import Path

GRADE_LIMITS = {'-3': 18.619855931827, '0': .060381588259977956, '3': .0025028764686429055}
PAIRED_KEYS = {'new_H_improvement_5pct', 'new_H_no_worse_analytic', 'old_coherent_H_preserved',
               'outside_highlight_exact', 'frozen_parent', 'partial_colour', 'no_new_negativeY',
               'aligned_residual_preserved', 'highlight_detail_no_worse_analytic',
               'fully_clipped_detail_no_worse_analytic'}
VIDEO_KEYS = {'complete', 'temporal', 'ordinary', 'colour', 'no_shadow_lift',
              'no_new_negativeY', 'outside_support', 'all_highlight_appearance'}
PRESERVATION_KEYS = {'frozen_weights', 'partial_weights_changed', 'nonempty_full_coverage',
                     'nonempty_outside_coverage', 'all_stages_preserved'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require_checks(checks, required, label):
    failed = sorted(k for k in required if checks.get(k) is not True)
    if failed:
        raise ValueError(f'{label}: missing/failed checks {failed}')


def require_grade_numbers(summary):
    for ev, limit in GRADE_LIMITS.items():
        value = summary['grades'][ev]['constrained']
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value > limit:
            raise ValueError(f'Original grade{ev} fails: {value} > {limit}')


def verify_candidate(root, steps):
    root = Path(root)
    def read(name):
        return json.loads((root / name).read_text())
    train = read('report.json'); paired = read('paired_development/acceptance.json')
    grade = read('old_detail_grade.json'); video = read('hunyuan48.json'); preserve = read('preservation.json')
    identity = read('duration_identity.json')
    prefix = 400 if steps == 1600 else 1600
    require_checks(identity, {f'history_first{prefix}_exact', f'schedule_first{prefix}_exact',
                             'crop_pool_exact', 'motion_pool_exact', f'updates{steps}', 'frozen_parent_exact'}, 'identity')
    checkpoint = root / f'step{steps}.pt'; digest = sha(checkpoint)
    if len(train['history']) != steps or train['checkpoint_sha256'] != digest:
        raise ValueError('Training checkpoint identity mismatch')
    if video['checkpoint_sha256'] != digest or preserve['candidate_sha256'] != digest:
        raise ValueError('Video/preservation checkpoint identity mismatch')
    for dataset in ('new_motion', 'old_coherent', 'legacy_natural'):
        prediction = read(f'paired_development/{dataset}/candidate_report.json')
        if prediction['checkpoint_sha256'] != digest:
            raise ValueError('Paired checkpoint identity mismatch')
    if grade['evaluation_sha256'] != sha(root / 'paired_development/old_coherent/candidate_report.json'):
        raise ValueError('Grade candidate input mismatch')
    if grade['parent_report_sha256'] != sha(root / 'paired_development/old_coherent/parent_report.json'):
        raise ValueError('Grade parent input mismatch')
    # The legacy outside_full_exact check assumes only full-branch updates.
    # Partial-route candidates require independent full/outside-highlight preservation instead.
    require_checks(paired['checks'], PAIRED_KEYS, 'paired')
    require_checks(preserve['checks'], PRESERVATION_KEYS, 'preservation')
    require_checks(video['checks'], VIDEO_KEYS, 'video')
    require_grade_numbers(grade['summary'])
    require_checks(grade['checks'], {'highlight_detail', 'fully_clipped_detail',
                                   'grade_-3', 'grade_0', 'grade_3'}, 'grading')
    bounds = [(paired['H_error_by_dataset']['new_motion']['candidate'], .292351081),
              (paired['H_error_by_dataset']['old_coherent']['candidate'], .3825248606569533),
              (paired['new_aligned_residual_mse']['candidate'], .010367255277136186),
              (video['max_transition_mean_ev'], .02)]
    if any(not math.isfinite(v) or v > limit for v, limit in bounds):
        raise ValueError('Original reconstruction/motion/video numerical limit failed')
    evidence = {name: sha(root / name) for name in ('report.json', 'duration_identity.json',
                'paired_development/acceptance.json', 'old_detail_grade.json', 'hunyuan48.json', 'preservation.json')}
    return {'checkpoint': checkpoint.as_posix(), 'checkpoint_sha256': digest,
            'evidence_sha256': evidence, 'original_dev_pass': True, 'release_authorized': False}
