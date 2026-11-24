"""AI fill v5 training material (22 Nov 2026): the CC0 collector, the Blender scenes and the builder's v5
recipe (collected panoramas, rendered sequences, lens optics on the truth, quarter-stop exposure)."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import collect_genfill_v5 as col  # noqa: E402

PH_META = {
    'window_room': {'name': 'Window Room', 'categories': ['indoor', 'natural light', 'high contrast'],
                    'tags': ['window', 'couch'], 'attributes': {'environment': 'indoor', 'light_type': 'natural', 'time_of_day': 'afternoon'},
                    'authors': {'A': 'all'}},
    'lamp_bar': {'name': 'Lamp Bar', 'categories': ['indoor', 'artificial light'], 'tags': ['lamp', 'bar'],
                 'attributes': {'environment': 'indoor', 'light_type': 'artificial', 'time_of_day': 'night'}, 'authors': {'B': 'all'}},
    'power_field': {'name': 'Power Field', 'categories': ['outdoor', 'natural light', 'clear', 'skies'], 'tags': ['power line', 'field', 'sun'],
                    'attributes': {'environment': 'outdoor', 'sky_view': 'open', 'weather': 'clear', 'time_of_day': 'midday'}, 'authors': {'C': 'all'}},
    'hangar_door': {'name': 'Hangar', 'categories': ['outdoor', 'urban'], 'tags': ['hangar', 'concrete'],
                    'attributes': {'environment': 'outdoor', 'sky_view': 'obstructed', 'weather': 'overcast', 'time_of_day': 'morning'}, 'authors': {}},
    'plain_meadow': {'name': 'Meadow', 'categories': ['outdoor', 'nature', 'overcast'], 'tags': ['grass'],
                     'attributes': {'environment': 'outdoor', 'weather': 'overcast', 'time_of_day': 'midday'}, 'authors': {}},
    'night_street': {'name': 'Night', 'categories': ['outdoor', 'night', 'artificial light'], 'tags': ['street light', 'cables'],
                     'attributes': {'environment': 'outdoor', 'time_of_day': 'night', 'light_type': 'artificial'}, 'authors': {}},
    'held_sky': {'name': 'Held', 'categories': ['outdoor', 'pure skies'], 'tags': ['pure skies', 'sun'],
                 'attributes': {'environment': 'outdoor', 'weather': 'clear'}, 'authors': {}},
}


def acg_asset(aid, cat, tags, size=1000):
    return {'assetId': aid, 'displayName': aid, 'displayCategory': cat, 'tags': tags, 'shortLink': f'https://ambientcg.com/a/{aid}',
            'downloadFolders': {'default': {'downloadFiletypeCategories': {'zip': {'downloads': [
                {'fullDownloadPath': f'https://x/get?file={aid}_{r}.zip', 'downloadLink': f'https://x/get?file={aid}_{r}.zip',
                 'filetype': 'zip', 'attribute': r, 'size': size * (4 if r == '4K' else 1)} for r in ('2K', '4K')]}}}}}


ACG = [acg_asset('DaySkyHDRI001A', 'Day Sky HDRI', ['sky', 'sunny', 'clear']),
       acg_asset('DayEnvironmentHDRI050', 'Day Environment HDRI', ['forest', 'trees', 'park']),
       acg_asset('IndoorEnvironmentHDRI003', 'Indoor Environment HDRI', ['indoor', 'interior', 'lights']),
       acg_asset('NightSkyHDRI002', 'Night Sky HDRI', ['night', 'stars'])]


# ------------------------------------------------------------------------------- themes
def test_themes_follow_tags_categories_and_attributes():
    t = {k: col.themes_of(m) for k, m in PH_META.items()}
    assert t['window_room'] == {'window'}
    assert 'practical' in t['lamp_bar'] and 'window' not in t['lamp_bar']      # night: no daylight window
    assert t['power_field'] == {'thin', 'sun'}
    assert t['hangar_door'] == {'door'}
    assert t['plain_meadow'] == set()
    assert t['night_street'] == {'practical'}                                   # no sun, no thin at night
    assert {'sky', 'sun'} <= t['held_sky']
    a = {x['assetId']: col.themes_of(x) for x in ACG}
    assert a['DaySkyHDRI001A'] == {'sky', 'sun'}
    assert a['DayEnvironmentHDRI050'] == {'thin'}
    assert a['IndoorEnvironmentHDRI003'] == {'practical'}                     # no window without a sign of one
    assert a['NightSkyHDRI002'] == set()


def test_choose_takes_every_theme_in_turn_within_the_budget():
    cands = [{'scene_id': f'{t}{i}', 'themes': [t]} for t in col.THEME_ORDER for i in range(5)]
    cands.append({'scene_id': 'both', 'themes': ['thin', 'sun']})
    cands.append({'scene_id': 'none', 'themes': []})
    sizes = {c['scene_id']: 10 for c in cands}
    picks = col.choose(cands, sizes, budget_bytes=60, held={'thin0'})
    ids = [c['scene_id'] for c in picks]
    assert len(ids) == 6 and 'thin0' not in ids and 'none' not in ids
    assert ids[0] == 'both'                                   # the asset serving two themes first
    assert {t for c in picks for t in c['themes']} == set(col.THEME_ORDER)   # every theme got one
    capped = col.choose(cands, sizes, 10 ** 9, set(), per_theme=2)
    by = {t: sum(t in c['themes'] for c in capped) for t in col.THEME_ORDER}
    assert max(by.values()) <= 2 + 1                          # 'both' counts for thin and sun
    kept = col.choose(cands, sizes, 25, set(), have={'door3'})
    assert kept[0]['scene_id'] == 'door3' and len(kept) == 2   # what is here counts against the budget


def test_acg_zip_and_the_extracted_exr(tmp_path):
    assert col.acg_zip(ACG[0], '4k') == {'url': 'https://x/get?file=DaySkyHDRI001A_4K.zip', 'size': 4000}
    assert col.acg_zip(ACG[0], '8k') is None
    z = tmp_path / 'a.zip'
    with zipfile.ZipFile(z, 'w') as f:
        f.writestr('a_4K-HDR.exr', b'x' * 100)
        f.writestr('a_4K-HDR_small.exr', b'x' * 10)
        f.writestr('a_4K-TONEMAPPED.jpg', b'j')
    col.extract_exr(z, tmp_path / 'a.exr')
    assert (tmp_path / 'a.exr').stat().st_size == 100
    z2 = tmp_path / 'b.zip'
    with zipfile.ZipFile(z2, 'w') as f:
        f.writestr('b.jpg', b'j')
    with pytest.raises(ValueError):
        col.extract_exr(z2, tmp_path / 'b.exr')


def _fake_api(tmp_path, exr_bytes):
    md5 = hashlib.md5(exr_bytes).hexdigest()

    def get(url):
        if url.endswith('/assets?type=hdris'):
            return PH_META
        if '/files/' in url:
            aid = url.rsplit('/', 1)[1]
            return {'hdri': {'4k': {'exr': {'url': f'https://dl/{aid}_4k.exr', 'size': len(exr_bytes),
                                            'md5': 'bad' if aid == 'hangar_door' else md5}}}}
        if 'ambientcg' in url:
            off = int(url.split('offset=')[1].split('&')[0])
            return {'numberOfResults': len(ACG), 'foundAssets': ACG[off:off + 100]}
        raise AssertionError(url)

    def fetch_file(url, dst):
        if url.endswith('.zip'):
            with zipfile.ZipFile(dst, 'w') as z:
                z.writestr('x_4K-HDR.exr', exr_bytes)
        else:
            Path(dst).write_bytes(exr_bytes)
    return get, fetch_file


def _exr_bytes(tmp_path, sun=200.0, clipped=False):
    import OpenEXR
    img = np.full((32, 64, 3), 0.5, np.float32)
    img[4:7, 10:13] = sun
    if not clipped:
        img[5, 11] = sun * 3
    p = tmp_path / 'one.exr'
    OpenEXR.File({'compression': OpenEXR.ZIP_COMPRESSION, 'type': OpenEXR.scanlineimage},
                 {c: np.ascontiguousarray(img[..., i]) for i, c in enumerate('RGB')}).write(str(p))
    return p.read_bytes()


def test_the_collector_plans_fetches_and_keeps_receipts(tmp_path):
    pytest.importorskip('OpenEXR')
    data = _exr_bytes(tmp_path)
    get, fetch_file = _fake_api(tmp_path, data)
    reg = tmp_path / 'corpora' / 'run05' / 'sources.jsonl'
    reg.parent.mkdir(parents=True)
    reg.write_text(json.dumps({'scene_id': 'ph:held_sky', 'split': 'test'}) + '\n')
    dest = tmp_path / 'v5'
    plan = col.main(['--dest', str(dest), '--registry', str(reg), '--dry-run'], get=get, fetch_file=fetch_file)
    assert plan['picked'] == 8 and not dest.exists()        # 9 themed assets, the held sky left out
    plan = col.main(['--dest', str(dest), '--registry', str(reg), '--keep-free-gb', '0'], get=get, fetch_file=fetch_file)
    rows = [json.loads(l) for l in (dest / 'manifest.jsonl').read_text().splitlines()]
    assert plan['fetched'] == 7 and len(rows) == 7          # hangar_door: md5 differs, set aside
    assert (dest / 'polyhaven' / 'hangar_door_4k.exr.bad').exists()
    assert not any(r['scene_id'] == 'ph:held_sky' for r in rows)
    r = next(r for r in rows if r['scene_id'] == 'ph:power_field')
    assert r['licence'] == 'CC0' and r['themes'] == ['sun', 'thin'] and r['light']['plateau'] is False
    assert r['light']['peak'] == pytest.approx(600.0, rel=1e-3)
    assert (dest / 'receipts' / 'polyhaven_power_field.json').exists()
    acg = next(r for r in rows if r['library'] == 'ambientcg')
    assert Path(acg['path']).exists() and not list((dest / 'ambientcg').glob('*.zip'))
    again = col.main(['--dest', str(dest), '--registry', str(reg), '--keep-free-gb', '0'], get=get, fetch_file=fetch_file)
    assert again['already_here'] == 7 and len((dest / 'manifest.jsonl').read_text().splitlines()) == 7


def test_a_clipped_sun_is_a_plateau(tmp_path):
    pytest.importorskip('OpenEXR')
    p = tmp_path / 'c.exr'
    p.write_bytes(_exr_bytes(tmp_path, clipped=True))
    assert col.light_stats(p)['plateau'] is True
    p.write_bytes(_exr_bytes(tmp_path, clipped=False))
    assert col.light_stats(p)['plateau'] is False


# ------------------------------------------------------------------------------- builder: optics
def _edge(h=48, w=96, F=3):
    ref = np.full((F, h, w, 3), 0.2, np.float32)
    ref[:, :, w // 2:] = 50.0
    return ref


def test_v5_optics_soften_the_edge_and_keep_the_light(monkeypatch):
    import tools.build_genfill_clips as b
    ref = _edge()
    monkeypatch.setattr(b, 'V5_OPTICS', {'defocus': 1.0, 'motion': 0.0, 'glare': 0.0})
    out, did = b.v5_optics(ref, np.random.default_rng(0), themes=('thin',))
    assert set(did) == {'defocus_px'}
    row = out[0, 24]
    assert ((row > 0.3) & (row < 49)).sum() >= 1            # the step is no longer a step
    assert out.mean() == pytest.approx(ref.mean(), rel=0.01)
    monkeypatch.setattr(b, 'V5_OPTICS', {'defocus': 0.0, 'motion': 0.0, 'glare': 1.0})
    out, did = b.v5_optics(ref, np.random.default_rng(1), themes=())
    assert set(did) == {'glare_share'}
    assert out[0, 24, 44].mean() > ref[0, 24, 44].mean() + 1e-4   # light spread into the dark side
    assert out.mean() == pytest.approx(ref.mean(), rel=0.02)  # energy kept
    assert np.array_equal(out[0], out[1])                    # a lens is the same on every frame
    monkeypatch.setattr(b, 'V5_OPTICS', {'defocus': 0.0, 'motion': 0.0, 'glare': 0.0})
    out, did = b.v5_optics(ref, np.random.default_rng(2), themes=('thin',))
    assert did == {} and np.array_equal(out, ref)
    out, did = b.v5_optics(ref, np.random.default_rng(2), themes=('sun',))   # sun: glare on 60 % anyway
    monkeypatch.setattr(b, 'V5_OPTICS', {'defocus': 0.0, 'motion': 1.0, 'glare': 0.0})
    out, did = b.v5_optics(ref, np.random.default_rng(3))
    assert 'motion_px' in did and out.shape == ref.shape


def test_sequence_path_crops_one_window_for_the_whole_clip():
    from tools.build_genfill_clips import sequence_path
    seq = np.random.default_rng(0).uniform(0, 4, (12, 72, 128, 3)).astype(np.float32)
    out = sequence_path(seq, np.random.default_rng(1), 9, 36, 64)
    assert out.shape == (9, 36, 64, 3) and out.dtype == np.float32


# ------------------------------------------------------------------------------- builder: plan and clips
def _write_exr(p, img):
    import OpenEXR
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    OpenEXR.File({'compression': OpenEXR.ZIP_COMPRESSION, 'type': OpenEXR.scanlineimage},
                 {c: np.ascontiguousarray(img[..., i].astype(np.float32)) for i, c in enumerate('RGB')}).write(str(p))


def _pano(h=128, w=256, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w, 3), 0.05, np.float32)
    sky = yy < h // 2
    img[sky] = (1.5 + 0.8 * np.sin(xx[sky] / 5.0) * np.cos(yy[sky] / 3.0))[:, None] * np.array([0.8, 0.9, 1.0], np.float32)
    img[20:24, 60:64] = 3000.0                                 # the sun
    img[:, 100:102] = 0.02                                     # a mast across the sky
    img += rng.uniform(0, 0.01, img.shape).astype(np.float32)
    return img


def _v5_tree(tmp_path):
    g = tmp_path / 'G'
    corp = g / 'datasets/corpora'
    reg = corp / 'rudra_run05_sources_20261022'
    reg.mkdir(parents=True)
    rows = [{'source_id': 'polyhaven_moves', 'scene_id': 'ph:old_two_k', 'split': 'train', 'asset_id': 'a1',
             'licence_licence': 'CC0', 'commercial_ok': True},
            {'source_id': 'polyhaven_moves', 'scene_id': 'ph:val_sky', 'split': 'val', 'asset_id': 'a2',
             'licence_licence': 'CC0', 'commercial_ok': True}]
    (reg / 'sources.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    ev = corp / 'aces_holdout_test'
    ev.mkdir()
    (ev / 'cases.jsonl').write_text(json.dumps({'scene_id': 'acg:HeldOut', 'split': 'train'}) + '\n')
    fresh = g / 'rudra_training/research_release_candidate_20261026/fresh_fxtd_grade_v2_20261026'
    fresh.mkdir(parents=True)
    (fresh / 'report.json').write_text(json.dumps({'scenes': 0, 'rows': []}))
    ph = g / 'datasets/sources/polyhaven'
    _write_exr(ph / 'old_two_k_2k.exr', _pano(seed=1))
    v5 = g / 'datasets/sources/genfill_v5'
    man = []
    for sid, lib, aid, themes in (('ph:old_two_k', 'polyhaven', 'old_two_k', ['sun', 'thin']),
                                  ('acg:Sky1', 'ambientcg', 'Sky1', ['sky', 'sun']),
                                  ('ph:val_sky', 'polyhaven', 'val_sky', ['sky']),
                                  ('acg:HeldOut', 'ambientcg', 'HeldOut', ['sky'])):
        p = v5 / lib / f'{aid}_4k.exr'
        _write_exr(p, _pano(seed=hash(aid) % 100))
        man.append({'asset_id': f'{lib}4k:{aid}', 'scene_id': sid, 'library': lib, 'themes': themes, 'res': '4k',
                    'path': 'G:/' + str(p.relative_to(g)).replace('\\', '/'), 'licence': 'CC0'})
    (v5 / 'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in man) + '\n')
    bl = v5 / 'blender'
    srows = []
    for name, sky in (('thin_00000', 'acg:Sky1'), ('door_00001', 'ph:val_sky'), ('glare_00002', None)):
        for f in range(12):
            img = _pano(64, 112, seed=f)
            _write_exr(bl / name / f'f.{f + 1:04d}.exr', np.roll(img, f, axis=1))
        srows.append({'asset_id': f'blender_v5:{name}', 'scene_id': f'blender:{name}', 'kind': name.split('_')[0],
                      'folder': 'G:/' + str((bl / name).relative_to(g)).replace('\\', '/'), 'frames': 12, 'sky_scene_id': sky})
    (bl / 'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in srows) + '\n')
    return g


def _args(g, **kw):
    import argparse
    from tools.build_genfill_clips import POLYHAVEN, SOURCES, V5_HDRIS, V5_SCENES
    a = dict(root=str(g).replace('\\', '/') + '/', sources=SOURCES, polyhaven=POLYHAVEN, frames=9, height=48, width=80,
             seed=1, clips_per_hdri=2, clips_per_plate=1, clips_per_fxtd=1, recipe='v5', dev_fraction=0.0,
             v5_hdris=V5_HDRIS, v5_scenes=V5_SCENES, clips_per_v5_hdri=3, clips_per_scene=2)
    a.update(kw)
    return argparse.Namespace(**a)


def test_plan_v5_adds_the_collected_material_and_keeps_held_scenes_out(tmp_path):
    pytest.importorskip('OpenEXR')
    from tools.build_genfill_clips import plan
    g = _v5_tree(tmp_path)
    jobs, _ = plan(_args(g))
    by = {j['scene_id']: j for j in jobs}
    assert sorted(by) == ['acg:Sky1', 'blender:glare_00002', 'blender:thin_00000', 'ph:old_two_k']
    assert by['ph:old_two_k']['path'].endswith('genfill_v5/polyhaven/old_two_k_4k.exr')   # the 4k replaces the 2k
    assert by['ph:old_two_k']['themes'] == ['sun', 'thin'] and by['ph:old_two_k']['clips'] == 3
    assert by['blender:thin_00000']['kind'] == 'sequence' and by['blender:thin_00000']['licence'].startswith('CC0 sources and authored')
    assert by['blender:glare_00002']['themes'] == ['glare', 'thin']
    jobs4, _ = plan(_args(g, recipe='v4_2'))                    # v4_2 is as it was
    assert [j['scene_id'] for j in jobs4] == ['ph:old_two_k'] and jobs4[0]['path'].endswith('old_two_k_2k.exr')


def test_v5_clips_from_a_panorama_and_a_sequence(tmp_path):
    pytest.importorskip('OpenEXR')
    pytest.importorskip('PyOpenColorIO')
    from tools.build_genfill_clips import COMMERCIAL, make_clip, plan
    g = _v5_tree(tmp_path)
    jobs, _ = plan(_args(g))
    seen = set()
    for j in jobs:
        if j['scene_id'] not in ('ph:old_two_k', 'blender:thin_00000'):
            continue
        clips = make_clip(j)
        assert clips, j['scene_id']
        for c in clips:
            m = c['meta']
            seen.add(m['kind'])
            assert c['sdr'].shape == (9, 48, 80, 3) and c['truth'].shape == (9, 48, 80, 3) and c['mask'].shape == (9, 48, 80)
            assert m['licence'] in COMMERCIAL and 'optics' in m and m['themes'] == j['themes']
            assert abs(m['offset'] * 4 - round(m['offset'] * 4)) < 1e-6        # quarter stops
            assert np.isfinite(c['truth'].astype(np.float32)).all()
    assert seen


def test_the_blender_scenes_render_every_kind(tmp_path):
    """Runs only where bpy imports (Blender's Python, or the bpy wheel)."""
    pytest.importorskip('bpy')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'blender'))
    import genfill_v5_scenes as gs
    out = tmp_path / 'scenes'
    gs.main(['--out', str(out), '--count', '5', '--frames', '2', '--width', '64', '--height', '36',
             '--samples', '2', '--device', 'CPU', '--no-denoise'])
    rows = [json.loads(l) for l in (out / 'manifest.jsonl').read_text().splitlines()]
    assert sorted(r['kind'] for r in rows) == sorted(gs.KINDS)
    for r in rows:
        assert len(list(Path(r['folder']).glob('f.*.exr'))) == 2 and r['space'] == 'rec709'
    gs.main(['--out', str(out), '--count', '5', '--frames', '2', '--width', '64', '--height', '36',
             '--samples', '2', '--device', 'CPU', '--no-denoise'])
    assert len((out / 'manifest.jsonl').read_text().splitlines()) == 5          # resumed: nothing twice


def test_the_blender_camera_never_turns_round_between_frames():
    """22 Nov 2026: a camera panning across +-180 degrees of yaw must not jump 360 degrees between two
    keyframes (Cycles' motion blur swept the turn into the frame: 9 swirled scenes in the first build)."""
    pytest.importorskip('bpy')
    import math
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'blender'))
    import genfill_v5_scenes as gs

    class Most:                                     # every draw at its top: the fastest pan, no still camera
        def uniform(self, a, b):
            return b

        def random(self):
            return 0.99
    for aim in ((-1, 0.02, 0.4), (-1, -0.02, 0.6), (0.02, -1, 0.3)):
        scene = gs.reset()
        cam = gs.camera(scene, Most(), (0, 0, 0), aim, 33)[0]
        vals = []
        for f in range(1, 34):
            scene.frame_set(f)
            vals.append(tuple(cam.rotation_euler))
        for a, b in zip(vals, vals[1:]):
            assert max(abs(x - y) for x, y in zip(a, b)) < math.radians(10), (aim, a, b)
