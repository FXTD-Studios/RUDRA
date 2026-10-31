import json

import pytest

from training.production_candidate_compare import compare_runs
from training.production_eval_kit import digest

SCENES = {'s1': 3, 's2': 2, 's3': 2}


def make_kit(tmp_path):
    kit = tmp_path / 'kit'
    kit.mkdir()
    (kit / 'validation.jsonl').write_text('{"asset_id": "x"}\n')
    contract = dict(validation_sha256=digest(kit / 'validation.jsonl'), frames=7, scenes=3,
                    conditions=['clean', 'hard'])
    (kit / 'contract.json').write_text(json.dumps(contract))
    return kit


def results(offset=0.0, per_scene=None, drop=False, nan=False):
    rows = []
    for scene, n in SCENES.items():
        for i in range(n):
            v = 30.0 + i + offset + (per_scene or {}).get(scene, 0.0)
            rows.append(dict(clip=scene, frame=f'{scene}_{i}', pu_psnr_db=v))
    if drop:
        rows.pop()
    if nan:
        rows[0]['pu_psnr_db'] = float('nan')
    return {'results': rows}


def make_run(tmp_path, kit, name, ckpt, offset, base=0.0, contract_hash=None, **kw):
    run = tmp_path / name
    for cond in ('clean', 'hard'):
        (run / cond).mkdir(parents=True)
        (run / cond / 'test.json').write_text(json.dumps(results(offset, **kw)))
        (run / cond / 'baseline_bt1886.json').write_text(json.dumps(results(base - 1.0)))
    (run / 'comparison.json').write_text(json.dumps(dict(
        checkpoint=ckpt, checkpoint_sha256=ckpt * 8, release_authorized=False,
        contract_sha256=contract_hash or digest(kit / 'contract.json'))))
    return run


def test_clear_win_and_per_scene_table(tmp_path):
    kit = make_kit(tmp_path)
    a = make_run(tmp_path, kit, 'a', 'aa', 0.5)
    b = make_run(tmp_path, kit, 'b', 'bb', 0.0)
    r = compare_runs(kit, a, b)
    assert r['release_authorized'] is False and r['promotion'] is False
    c = r['conditions']['hard']
    assert c['ci_verdict'] == 'a_better'
    assert c['a_minus_b']['scenes'] == 3 and c['a_minus_b']['frames'] == 7
    assert c['per_scene_worst_first'][0]['a_minus_b'] == pytest.approx(0.5)


def test_mixed_scenes_unresolved_and_worst_first(tmp_path):
    kit = make_kit(tmp_path)
    a = make_run(tmp_path, kit, 'a', 'aa', 0.0, per_scene={'s1': 1.0, 's2': -2.0, 's3': 0.5})
    b = make_run(tmp_path, kit, 'b', 'bb', 0.0)
    c = compare_runs(kit, a, b)['conditions']['clean']
    assert c['ci_verdict'] == 'unresolved'
    assert c['per_scene_worst_first'][0]['scene'] == 's2'
    assert c['scenes_a_worse'] == 1


@pytest.mark.parametrize('case', ['same_ckpt', 'other_kit', 'drop', 'nan', 'baseline_drift'])
def test_rejections(tmp_path, case):
    kit = make_kit(tmp_path)
    b = make_run(tmp_path, kit, 'b', 'bb', 0.0)
    kw = {}
    if case == 'same_ckpt':
        a = make_run(tmp_path, kit, 'a', 'bb', 0.5)
    elif case == 'other_kit':
        a = make_run(tmp_path, kit, 'a', 'aa', 0.5, contract_hash='0' * 64)
    elif case == 'baseline_drift':
        a = make_run(tmp_path, kit, 'a', 'aa', 0.5, base=0.01)
    else:
        kw[case] = True
        a = make_run(tmp_path, kit, 'a', 'aa', 0.5, **kw)
    with pytest.raises(ValueError):
        compare_runs(kit, a, b)


def test_changed_kit_rejected(tmp_path):
    kit = make_kit(tmp_path)
    a = make_run(tmp_path, kit, 'a', 'aa', 0.5)
    b = make_run(tmp_path, kit, 'b', 'bb', 0.0)
    (kit / 'validation.jsonl').write_text('{"asset_id": "changed"}\n')
    with pytest.raises(ValueError, match='changed'):
        compare_runs(kit, a, b)
