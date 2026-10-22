import pytest
from training.production_eval_kit import select_validation, strict_scores, evaluate


def rows():
    return [dict(asset_id='a', scene_id='one', split='val'),
            dict(asset_id='b', scene_id='two', split='val'),
            dict(asset_id='c', scene_id='three', split='test')]


def test_final_test_excluded():
    assert [r['asset_id'] for r in select_validation(rows())] == ['a', 'b']


def test_scene_leakage_rejected():
    r = rows(); r[-1]['scene_id'] = 'one'
    with pytest.raises(ValueError, match='overlaps'):
        select_validation(r)


def test_duplicate_ids_rejected():
    r = rows(); r[-1]['asset_id'] = 'a'
    with pytest.raises(ValueError, match='Duplicate'):
        select_validation(r)


def test_missing_scene_rejected():
    r = rows(); r[0]['scene_id'] = ''
    with pytest.raises(ValueError, match='Missing'):
        select_validation(r)


@pytest.mark.parametrize('values', [[], [float('nan')], [float('inf')], [None], [20, 21]])
def test_incomplete_nonfinite_duplicate_metrics_rejected(values):
    scores = [dict(clip='one', frame='a', pu_psnr_db=v) for v in values]
    with pytest.raises(ValueError):
        strict_scores(scores, {('one', 'a'): 'scene'})


def test_exact_coverage():
    assert strict_scores([dict(clip='one', frame='a', pu_psnr_db=20)],
                         {('one', 'a'): 'scene'}) == {('one', 'a'): {'pu_psnr_db': 20}}


def test_active_training_blocks_before_touching_files(monkeypatch, tmp_path):
    from training import run_next_hdr_commercial
    monkeypatch.setattr(run_next_hdr_commercial, 'active_trainers', lambda: [1234])
    with pytest.raises(RuntimeError, match='Training is active'):
        evaluate(tmp_path/'missing', tmp_path/'missing.pt', tmp_path/'out')
    assert not (tmp_path/'out').exists()
