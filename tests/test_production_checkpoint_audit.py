import pytest
from training.audit_production_checkpoints import screen, read_evaluations


@pytest.mark.parametrize('clean,hard,expected', [(1, .2, True), (1, -.1, False),
    (0, 1, False), (float('nan'), 1, False), (1, float('inf'), False), (None, 1, False)])
def test_screen(clean, hard, expected):
    assert screen(dict(clean_pu_gain_db=clean, hard_pu_gain_db=hard)) is expected


def test_live_tail():
    assert read_evaluations(b'{"step":0,"eval":{}}\n{"ste') == [{'step': 0, 'eval': {}}]


def test_corruption_is_not_ignored():
    with pytest.raises(ValueError):
        read_evaluations(b'broken\n{"step":0,"eval":{}}\n')
