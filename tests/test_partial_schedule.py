import json
from pathlib import Path
import pytest
from training.aces_partial_schedule import file_sha, load_registered_schedule


def setup_schedule(tmp_path):
    out = tmp_path/'out'; out.mkdir()
    for name in ('crop_pool', 'motion_pool'):
        (out/(name+'.json')).write_text(json.dumps([{}, {}, {}, {}]))
    (tmp_path/'control.json').write_text(json.dumps([[0, 1], [0, 1, 2, 3]]))
    (tmp_path/'schedule.json').write_text(json.dumps([[2, 3], [0, 1, 2, 3]]))
    manifest = dict(steps=2, batch=4, crop=192, border=64, paired_motion=True,
                    crop_pool_sha256=file_sha(out/'crop_pool.json'),
                    motion_pool_sha256=file_sha(out/'motion_pool.json'),
                    schedule_sha256=file_sha(tmp_path/'schedule.json'),
                    control_schedule_path=str(tmp_path/'control.json'),
                    control_schedule_sha256=file_sha(tmp_path/'control.json'), pilot_hashes={})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return out, manifest


def load(tmp_path, out):
    return load_registered_schedule(tmp_path, out, steps=2, batch=4, crop=192, border=64, paired_motion=True)


def test_valid_motion_change(tmp_path):
    out, _ = setup_schedule(tmp_path)
    assert load(tmp_path, out) == [[2, 3], [0, 1, 2, 3]]


@pytest.mark.parametrize('schedule', [[[2, 2], [0, 1, 2, 3]], [[2, 4], [0, 1, 2, 3]], [[2, 3], [1, 0, 2, 3]]])
def test_reject_invalid_or_changed_still(tmp_path, schedule):
    out, manifest = setup_schedule(tmp_path)
    (tmp_path/'schedule.json').write_text(json.dumps(schedule))
    manifest['schedule_sha256'] = file_sha(tmp_path/'schedule.json')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load(tmp_path, out)


def test_reject_changed_pool(tmp_path):
    out, _ = setup_schedule(tmp_path)
    (out/'motion_pool.json').write_text('[{}]')
    with pytest.raises(ValueError, match='pool identity'):
        load(tmp_path, out)


def test_reject_changed_motion_group(tmp_path):
    out, manifest = setup_schedule(tmp_path)
    (out/'motion_pool.json').write_text(json.dumps([{'group':g} for g in ('a','b','a','c')]))
    manifest['motion_pool_sha256'] = file_sha(out/'motion_pool.json')
    manifest['preserve_motion_groups'] = True
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='scene/exposure'):
        load(tmp_path, out)
