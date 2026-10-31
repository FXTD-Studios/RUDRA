from tools.prepare_run05_expansion import select


def row(scene, split='train', **kw):
    return dict(scene_id='ph:' + scene, asset_id=scene, source_id='polyhaven_moves',
                split=split, commercial_ok=True, licence_licence='CC0', **kw)


def test_related_heldout_names_and_unknown_rights_are_excluded():
    rows = [row('factory_01'), row('factory_02', 'val'), row('safe_night'), row('safe_day'),
            row('historical_sunrise'), row('unknown')]
    rows[-1]['commercial_ok'] = None
    history = [dict(scene_id='hl_ph_historical_sunset', split='test', source_id='polyhaven')]
    picked, report = select(rows, [row('pilot', 'val')], history, count=2, dark_count=1)
    assert [r['scene_id'] for r in picked] == ['ph:safe_night', 'ph:safe_day', 'ph:pilot']
    assert report['name_selected_dark_scenes'] == 1
