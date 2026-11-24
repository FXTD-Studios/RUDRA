"""AI fill v5 training (22 Nov 2026): the sampling file (tools/genfill_v5_weights.py) and how the trainer uses it.
Registration: configs/experiments/research_genfill_v5_20261122.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip('cv2')

from tools import genfill_v5_weights as gw  # noqa: E402

H, W, F = 32, 48, 5


def test_theme_factor_takes_the_largest_and_damps_sun_and_sky():
    assert gw.theme_factor(['sun', 'sky']) == 0.5
    assert gw.theme_factor(['sun', 'door']) == 2.0
    assert gw.theme_factor(['practical', 'sun', 'thin']) == 1.5
    assert gw.theme_factor([]) == 1.0


def test_thin_edge_counts_a_thin_dark_line_beside_the_region_not_a_wide_block():
    y = np.zeros((H, W), np.float32)
    mask = np.zeros((H, W), np.uint8)
    mask[:, :20] = 1                                  # the bright region
    y[:, :20] = 4.0
    y[:, 20:] = 3.0                                   # just outside: a bit darker, not 3 stops
    y[:, 22:24] = -2.0                                # a 2 px antenna 6 stops down, within 8 px of the region
    assert gw.thin_edge_px(y, mask) == 2 * H
    y2 = y.copy(); y2[:, 22:24] = 3.0; y2[:, 21:40] = -2.0      # a wide dark wall instead: not thin
    assert gw.thin_edge_px(y2, mask) < 2 * H
    y3 = y.copy(); y3[:, 22:24] = 3.0; y3[:, 40:42] = -2.0      # the antenna 20 px away: outside the band
    assert gw.thin_edge_px(y3, mask) == 0


def test_crossed_dev_leaves_out_whichever_side_is_dev():
    meta = [{'scene_id': 'blender:a', 'split': 'dev'}, {'scene_id': 'ph:sky1', 'split': 'train'},
            {'scene_id': 'blender:b', 'split': 'train'}, {'scene_id': 'acg:sky2', 'split': 'dev'},
            {'scene_id': 'blender:c', 'split': 'train'}, {'scene_id': 'ph:sky3', 'split': 'train'}]
    scenes = [{'scene_id': 'blender:a', 'sky_scene_id': 'ph:sky1'}, {'scene_id': 'blender:b', 'sky_scene_id': 'acg:sky2'},
              {'scene_id': 'blender:c', 'sky_scene_id': 'ph:sky3'}, {'scene_id': 'blender:d', 'sky_scene_id': 'ph:nowhere'}]
    assert gw.crossed_dev(meta, scenes) == {'blender:a', 'acg:sky2'}


def fake_store(tmp_path):
    root = tmp_path / 'clips'; root.mkdir()
    rows = [('train', 'blender_v5', 'blender:a2', ['door'], 'region', True),
            ('train', 'polyhaven_hdri_v5', 'ph:sun', ['sun', 'sky'], 'region', False),
            ('train', 'fxtd_lab', 'fxtd_x', [], 'highlight', False),
            ('dev', 'blender_v5', 'blender:a', ['window'], 'region', False),
            ('dev', 'polyhaven_hdri_v5', 'ph:other', ['thin'], 'shadow', False),
            ('train', 'polyhaven_hdri_v5', 'ph:sky1', ['sun'], 'region', False)]
    meta, ylog, mask = [], [], []
    for i, (split, src, scene, themes, kind, line) in enumerate(rows):
        y = np.full((H, W), 3.0, np.float32); m = np.zeros((F, H, W), np.uint8)
        m[:, :, :20] = 1; y[:, :20] = 4.0
        if line:
            y[:, 22:24] = -3.0
        ylog.append(y.astype(np.float16)); mask.append(np.packbits(m, axis=None))
        meta.append({'asset_id': f'a{i}', 'scene_id': scene, 'source': src, 'licence': 'CC0', 'kind': kind, 'split': split,
                     'themes': themes, 'structure': 0.05 if kind == 'region' else None, 'shard': 'all_0000', 'index': i})
    np.save(root / 'all_0000_ylog.npy', np.stack(ylog)); np.save(root / 'all_0000_mask.npy', np.stack(mask))
    (root / 'clips.jsonl').write_text(''.join(json.dumps(m) + '\n' for m in meta))
    (root / 'summary.json').write_text(json.dumps({'recipe': 'v5', 'frames': F, 'size': [H, W]}))
    sc = tmp_path / 'scenes.jsonl'
    sc.write_text(json.dumps({'scene_id': 'blender:a', 'sky_scene_id': 'ph:sky1'}) + '\n')
    return root, sc, meta


def test_the_tool_writes_weights_and_the_dev_rule_once(tmp_path):
    root, sc, meta = fake_store(tmp_path)
    gw.main(['--clips', str(root), '--scenes', str(sc)])
    d = json.loads((root / 'sampling_v5.json').read_text())
    w = {r['asset_id']: r for r in d['rows']}
    assert w['a0']['weight'] == 4.0 and w['a0']['thin_edge_px'] >= gw.EDGE_MIN_PX     # door 2 x edge 2
    assert w['a1']['weight'] == 0.5 and w['a2']['weight'] == 1.0
    assert w['a3']['dev_monitor'] is False and w['a4']['dev_monitor'] is True
    assert d['dev_left_out'] == {'scenes': ['blender:a'], 'clips': 1}
    assert d['clips_jsonl_sha256'] == gw.sha(root / 'clips.jsonl')
    with pytest.raises(SystemExit):                                                    # written once
        gw.main(['--clips', str(root), '--scenes', str(sc)])


def test_the_swirled_blender_scenes_are_left_out(tmp_path, monkeypatch):
    """22 Nov 2026: 9 motion-blurred Blender scenes whose camera rotation wrapped round: weight 0, not in the
    dev monitor, listed in the file."""
    root, sc, meta = fake_store(tmp_path)
    monkeypatch.setattr(gw, 'BROKEN_SCENES', {'blender:a2': [3], 'blender:a': [5]})
    gw.main(['--clips', str(root), '--scenes', str(sc)])
    d = json.loads((root / 'sampling_v5.json').read_text())
    w = {r['asset_id']: r for r in d['rows']}
    assert w['a0']['weight'] == 0.0 and w['a0']['dev_monitor'] is False
    assert w['a3']['dev_monitor'] is False and w['a1']['weight'] == 0.5
    assert d['left_out']['scenes'] == ['blender:a', 'blender:a2'] and d['left_out']['clips'] == 2


def test_the_trainer_multiplies_the_weights_and_drops_the_marked_dev_clips(tmp_path):
    pytest.importorskip('torch')
    from training.genfill_train import apply_clip_weights, load_clip_weights, pick_dev, sampling_weights
    root, sc, meta = fake_store(tmp_path)
    gw.main(['--clips', str(root), '--scenes', str(sc)])
    rows = load_clip_weights(root / 'sampling_v5.json', root)
    train = [m for m in meta if m['split'] == 'train']; dev = [m for m in meta if m['split'] == 'dev']
    mult, kept = apply_clip_weights(train, dev, rows)
    w = (sampling_weights(train, 3.0, 0.10) * mult).tolist()
    assert w == pytest.approx([2.5 * 4.0, 2.5 * 0.5, 1.0, 2.5 * 0.5])
    assert [m['asset_id'] for m in kept] == ['a4']
    (root / 'clips.jsonl').write_text((root / 'clips.jsonl').read_text() + '\n')      # another store
    with pytest.raises(SystemExit):
        load_clip_weights(root / 'sampling_v5.json', root)
    assert pick_dev(279, 32, 'first') == list(range(32))
    s = pick_dev(240, 32, 'spread')
    assert len(s) == 32 and s[0] == 0 and s[-1] == 239
    assert pick_dev(5, 32, 'spread') == [0, 1, 2, 3, 4]
