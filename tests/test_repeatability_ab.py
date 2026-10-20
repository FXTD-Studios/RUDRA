import json
import pytest
import torch
from tests.test_preclamp_hold import preservation_recipe
from training.routing_ab import arm_args, arms, repeatability_comparison


def recipe_at(path):
    r = preservation_recipe(path)
    r['scope'] = 'internal-repeatability'
    r['trainer_args'] += ['--deterministic', '--log-every', '1', '--crop-size', '256', '--base-channels', '32']
    return r


def test_same_resolved_recipe(tmp_path):
    r = recipe_at(tmp_path)
    configs = [vars(arm_args(r, arm)) for arm in arms(r)]
    assert configs[0]['crushed_hold_domain'] == 'preclamp'
    assert configs[0]['region_focus'] == 0
    assert configs[0]['bench_push_pu']
    assert {k for k in configs[0] if configs[0][k] != configs[1][k]} == {'output_dir', 'preservation_ab_arm'}


@pytest.mark.parametrize('scope,seconds,valid', [
    ('internal-repeatability', 900, True), ('internal-repeatability', 901, False),
    ('internal-preservation-ab', 900, False), ('internal-region-focus-ab', 900, False),
    ('internal-routing-ab', 900, False)])
def test_extended_cap_is_repeatability_only(tmp_path, monkeypatch, scope, seconds, valid):
    from training import routing_ab as ab
    from pipeline.commercial_preflight import sha256
    monkeypatch.setattr(ab, 'REPO', tmp_path)
    monkeypatch.setattr(ab, 'CODE_FILES', ())
    manifest, evidence = tmp_path/'manifest', tmp_path/'evidence'
    manifest.write_text('synthetic')
    evidence.write_text(json.dumps({'scope':'internal-pilot','release_authorized':False,
        'reviewed_by':'test','reviewed_at':'test','sources':{'test':{'weight_distribution':False}}}))
    recipe={'scope':scope,'release_authorized':False,'arm_steps':200,'arm_seconds':seconds,
        'storage_stop_bytes':900000000,'output_root':str(tmp_path/'reports/new'),'code_sha256':{},
        'manifest':str(manifest),'manifest_sha256':sha256(manifest),
        'evidence':str(evidence),'evidence_sha256':sha256(evidence)}
    path, approval=tmp_path/'recipe.json',tmp_path/'approval.json'
    path.write_text(json.dumps(recipe))
    approval.write_text(json.dumps({'approved':True,'approved_by':'test','approved_at':'test',
        'release_authorized':False,'recipe_sha256':sha256(path)}))
    if valid:
        assert ab.approved_recipe(path,approval)['arm_seconds']==900
    else:
        with pytest.raises(ValueError,match='limits'): ab.approved_recipe(path,approval)


@pytest.mark.parametrize('change', ['--log-every', '--crop-size', '--base-channels'])
def test_confounds_rejected(tmp_path, change):
    r = recipe_at(tmp_path)
    r['trainer_args'] += [change, '3']
    with pytest.raises(ValueError, match='repeatability'):
        arm_args(r, 'control')


@pytest.mark.parametrize('change', ['none', 'weights', 'logs', 'inputs', 'runtime', 'missing_step'])
def test_exact_verdict(tmp_path, change):
    r = recipe_at(tmp_path)
    for arm in arms(r):
        out = tmp_path / arm
        out.mkdir()
        altered = arm == 'repeat'
        config = {'runtime': {'deterministic_algorithms': not (altered and change == 'runtime')}}
        weights = {'w': torch.tensor([2. if altered and change == 'weights' else 1.])}
        torch.save({'model': weights, 'config': config}, out / 'step_0000002.pt')
        torch.save({'model': {'w': torch.zeros(1)}}, out / 'step_0000000.pt')
        rows = [{'step': i, 'total': float(i)} for i in (1,2)]
        if altered and change == 'logs': rows[1]['total'] += .1
        if altered and change == 'missing_step': rows.pop()
        (out / 'train.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
        (out / 'bench_val.jsonl').write_text('{"step":2}')
        (tmp_path / f'{arm}_samples.jsonl').write_text('different' if altered and change == 'inputs' else 'same')
    if change in ('inputs', 'runtime', 'missing_step'):
        with pytest.raises(ValueError): repeatability_comparison(tmp_path, r)
    else:
        result = repeatability_comparison(tmp_path, r)
        assert result['repeatability_pass'] == (change == 'none')
        assert not result['production_pass']
