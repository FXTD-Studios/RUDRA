"""Layer 3 end to end on a tiny random Wan VACE: a master folder in, a filled folder out.
Checks the contract, not the picture (the tiny model is noise): every 8-bit code of the SDR
view kept, untouched pixels bit-identical, the masks written, windows joined, AP0 and half."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip('diffusers')
pytest.importorskip('OpenEXR')
pytest.importorskip('PyOpenColorIO')

from rudra.genfill.masks import MaskSettings
from rudra.genfill.model import FillSettings
from rudra.genfill.run import fill_master, pad_len, plan_windows, read_frame, working_size

from rudra.genfill.synthetic import write_master  # noqa: E402


def test_planning():
    assert plan_windows(5, 33, 8) == [(0, 0, 5)]
    p = plan_windows(70, 33, 8)
    assert p == [(0, 0, 33), (25, 33, 58), (50, 58, 70)]
    written = [i for s, n, e in p for i in range(n, e)]
    assert written == list(range(70))
    assert [pad_len(k) for k in (1, 2, 5, 6, 33)] == [1, 5, 5, 9, 33]
    assert working_size(1080, 1920, 832 * 480) == (464, 832) and working_size(480, 832, 832 * 480) == (480, 832)
    assert working_size(100, 100, 832 * 480) == (96, 96) and working_size(8, 8, 999) == (16, 16)
    with pytest.raises(ValueError):
        plan_windows(10, 32, 8)


@pytest.mark.parametrize('ap0,half', [(False, False), (True, True)])
def test_fill_master_keeps_the_contract(tmp_path, ap0, half):
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    view = ACES2View()
    master = tmp_path / 'master'
    write_master(master, ap0=ap0, half=half)
    out = tmp_path / 'filled'
    events = []
    rep = fill_master(master, out, tiny_fill(), view, FillSettings(steps=2, seed=3),
                      MaskSettings(dilate=2, feather=3), window=5, overlap=2, max_area=48 * 64,
                      emit=lambda **o: events.append(o))
    names = sorted(p.name for p in master.glob('*.exr'))
    assert sorted(p.name for p in out.glob('*.exr')) == names
    assert rep['stats']['windows'] == 2 and rep['stats']['windows_generated'] == 2
    assert rep['stats']['generated_pixels'] > 0
    assert json.loads((out / 'genfill.json').read_text())['release_authorized'] is False
    assert (out / 'master_provenance.json').exists()
    assert [e['index'] for e in events if e['event'] == 'frame'] == list(range(1, 8))
    for name in names:
        a, b = read_frame(master / name), read_frame(out / name)
        assert b['gamut'] == ('ap0' if ap0 else 'ap1') and b['dtype'] == a['dtype']
        gen = b['channels']['rudra.generated'].astype(np.float32)
        touched = gen > 0
        for c in 'RGB':                                   # untouched: bit-identical
            assert np.array_equal(a['channels'][c][~touched].view(np.uint8), b['channels'][c][~touched].view(np.uint8))
        codes = lambda f: np.rint(np.clip(view.forward(f['ap1']), 0, 1) * 255)
        diff = np.abs(codes(a) - codes(b))
        # float32: every code kept; half: the file's own rounding may move a code by one at most
        assert diff.max() <= (1 if half else 0)
        assert (b['channels']['rudra.reconstructed'] >= a['channels']['rudra.reconstructed']).all()
        assert (gen <= 1).all() and (gen >= 0).all()
        if touched.any():
            assert not np.array_equal(a['ap1'][touched], b['ap1'][touched])


def test_v1_1_blends_only_where_the_source_is_fully_clipped(tmp_path):
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    view = ACES2View()
    master = tmp_path / 'master'
    write_master(master, n=5)
    out = tmp_path / 'filled'
    fill_master(master, out, tiny_fill(), view, FillSettings(steps=2, seed=1), MaskSettings(dilate=2),
                window=5, overlap=2, max_area=48 * 64)
    for name in sorted(p.name for p in master.glob('*.exr')):
        a, b = read_frame(master / name), read_frame(out / name)
        codes = np.rint(np.clip(view.forward(a['ap1']), 0, 1) * 255).max(-1)
        gen = b['channels']['rudra.generated'] > 0
        assert gen.any()
        assert (codes[gen] >= 255).all()                  # nothing blended into codes 250-254 ...
        assert not gen[codes <= 4].any()                  # ... nor into the shadows (off by default)


def test_v2_caps_the_fill_at_three_stops_over_the_master(tmp_path):
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    view = ACES2View()
    master = tmp_path / 'master'
    write_master(master, n=5)
    lumf = lambda x: np.maximum(x @ np.array([.2722287168, .6740817658, .0536895174], np.float32), 1e-6)
    for cap in (3.0, 0.5):
        out = tmp_path / f'filled_{cap}'
        fill_master(master, out, tiny_fill(), view, FillSettings(steps=2, seed=1), MaskSettings(dilate=2, max_gain_stops=cap),
                    window=5, overlap=2, max_area=48 * 64)
        for name in sorted(p.name for p in master.glob('*.exr')):
            a, b = read_frame(master / name), read_frame(out / name)
            gen = b['channels']['rudra.generated'] > 0
            ratio = lumf(b['ap1'][gen]) / lumf(a['ap1'][gen])
            assert ratio.max() <= 2.0 ** cap * 1.001


def test_fill_master_refuses_an_existing_folder(tmp_path):
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    master = tmp_path / 'm'
    write_master(master, n=1)
    (tmp_path / 'o').mkdir()
    with pytest.raises(FileExistsError):
        fill_master(master, tmp_path / 'o', tiny_fill(), ACES2View(), FillSettings(steps=1), MaskSettings())


def test_a_single_still(tmp_path):
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    master = tmp_path / 'm'
    write_master(master, n=1)
    f = next(master.glob('*.exr'))
    rep = fill_master(f, tmp_path / 'o', tiny_fill(), ACES2View(), FillSettings(steps=2), MaskSettings(dilate=1),
                      max_area=48 * 64)
    assert rep['stats']['frames'] == 1 and (tmp_path / 'o' / f.name).exists()


def test_a_stop_mid_window_leaves_only_whole_frames(tmp_path):
    """Review 28 Oct: the app's 'cancel' stops the fill at its next step; frames are
    written as .part and renamed, so nothing short is left under a finished name."""
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    master = tmp_path / 'master'
    write_master(master)
    out = tmp_path / 'filled'
    steps = []
    def emit(**o):
        if o.get('event') == 'progress':
            steps.append(o)
    # Stops in the second window: the first window's frames are whole, none after.
    stop = lambda: any(s['window'] == 2 for s in steps)
    rep = fill_master(master, out, tiny_fill(), ACES2View(), FillSettings(steps=2, seed=3),
                      MaskSettings(dilate=2, feather=3), window=5, overlap=2, max_area=48 * 64,
                      emit=emit, cancel=stop)
    assert rep['stats'].get('cancelled') is True
    written = sorted(p.name for p in out.glob('*.exr'))
    assert 0 < len(written) < len(list(master.glob('*.exr')))
    assert not list(out.glob('*.part'))
    for name in written:
        read_frame(out / name)


def test_the_fill_leaves_a_review_for_the_app(tmp_path):
    """Review 28 Oct: A/B pictures and the mask per frame, and the flicker the fill adds."""
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    master = tmp_path / 'master'
    write_master(master)
    out = tmp_path / 'filled'
    rep = fill_master(master, out, tiny_fill(), ACES2View(), FillSettings(steps=2, seed=3),
                      MaskSettings(dilate=2, feather=3), window=5, overlap=2, max_area=48 * 64)
    names = sorted(p.name for p in master.glob('*.exr'))
    assert [f['name'] for f in rep['frames']] == names
    assert rep['frames'][0]['temporal_excess_stops'] is None
    for f in rep['frames'][1:]:
        assert f['temporal_excess_stops'] is None or f['temporal_excess_stops'] >= 0
    for n in names:
        stem = Path(n).stem
        for suffix in ('a', 'b', 'mask'):
            assert (out / 'review' / f'{stem}.{suffix}.png').exists()
    report = json.loads((out / 'genfill.json').read_text())
    assert report['review']['folder'] == 'review' and len(report['frames']) == len(names)


# ---- RUDRA 1.1 priority 2 (5 Nov 2026): the reconstruction region and prompt strength ----

def test_a_region_that_excludes_the_highlights_leaves_the_master_bit_for_bit(tmp_path):
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    from rudra.delivery.region import ReconstructionRegion
    view = ACES2View()
    master = tmp_path / 'master'
    write_master(master, n=5)
    out = tmp_path / 'filled'
    shadows_only = ReconstructionRegion(enabled=True, band=True, band_low=0.0, band_high=0.3, band_softness=0.0)
    rep = fill_master(master, out, tiny_fill(), view, FillSettings(steps=2, seed=1), MaskSettings(dilate=2),
                      window=5, overlap=2, max_area=48 * 64, region=shadows_only)
    assert rep['stats']['generated_pixels'] == 0
    assert rep['stats']['windows_generated'] == 0          # nothing to fill: the model never ran
    assert rep['region']['band_high'] == 0.3
    for name in sorted(p.name for p in master.glob('*.exr')):
        a, b = read_frame(master / name), read_frame(out / name)
        for c in 'RGB':
            assert np.array_equal(a['channels'][c], b['channels'][c])


def test_the_fill_acts_only_inside_the_region(tmp_path):
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    from rudra.delivery.region import ReconstructionRegion, fill_region_weight
    view = ACES2View()
    master = tmp_path / 'master'
    write_master(master, n=5)
    region = ReconstructionRegion(enabled=True, highlight_from=0.97, highlight_softness=0.02, shadow_strength=0.0)
    out = tmp_path / 'filled'
    fill_master(master, out, tiny_fill(), view, FillSettings(steps=2, seed=1), MaskSettings(dilate=2),
                window=5, overlap=2, max_area=48 * 64, region=region)
    for name in sorted(p.name for p in master.glob('*.exr')):
        a, b = read_frame(master / name), read_frame(out / name)
        codes = np.rint(np.clip(view.forward(a['ap1']), 0, 1) * 255).astype(np.uint8)
        w = fill_region_weight(codes, region)
        gen = b['channels']['rudra.generated'].astype(np.float32)
        assert not (gen[w == 0] > 0).any()
        assert (gen <= w + 1e-6).all()                       # the blend never exceeds the region's weight
        for c in 'RGB':
            assert np.array_equal(a['channels'][c][w == 0], b['channels'][c][w == 0])


def test_prompt_strength_one_is_v3_and_zero_ignores_the_prompt():
    import torch
    from genfill_tiny import tiny_fill
    m = tiny_fill()
    ctx = torch.randn(1, 96, 2, 8, 8, generator=torch.Generator().manual_seed(0))
    reg = {'highlight': torch.ones(1, 1, 5, 16, 16)}
    run = lambda **kw: m.denoise(ctx, FillSettings(steps=2, seed=4, **kw), reg)   # noqa: E731
    assert torch.equal(run(prompt_highlight='a bright sun'),
                       run(prompt_highlight='a bright sun', prompt_strength=1.0))
    assert torch.equal(run(prompt_highlight='a bright sun', prompt_strength=0.0), run())
    assert not torch.equal(run(prompt_highlight='a bright sun', prompt_strength=1.5),
                           run(prompt_highlight='a bright sun'))


def test_generated_is_split_between_highlights_and_shadows_exactly(tmp_path):
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    from rudra.genfill.run import GENERATED_PARTS
    master = tmp_path / 'master'
    write_master(master, n=5)
    out = tmp_path / 'filled'
    fill_master(master, out, tiny_fill(), ACES2View(), FillSettings(steps=2, seed=1),
                MaskSettings(dilate=2, tau_low=8, blend_low=4), window=5, overlap=2, max_area=48 * 64)
    seen = 0
    for name in sorted(p.name for p in master.glob('*.exr')):
        ch = read_frame(out / name)['channels']
        g, hi, lo = ch['rudra.generated'], ch[GENERATED_PARTS[0]], ch[GENERATED_PARTS[1]]
        assert np.array_equal(hi + lo, g)                   # exactly rudra.generated
        assert not ((hi > 0) & (lo > 0)).any()              # a pixel belongs to one region
        seen += int((hi > 0).sum())
    assert seen > 0


# ---- RUDRA 1.1 priority 5 (5 Nov 2026): the fallback when the fill flickers ----

def test_spike_fallback_writes_the_masters_frame_and_keeps_onset_and_end(tmp_path, monkeypatch):
    import rudra.genfill.run as R
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    master = tmp_path / 'master'
    write_master(master, n=5)
    monkeypatch.setattr(R, 'TEMPORAL_FLAG', -1.0)          # every measured frame counts as a jump
    out = tmp_path / 'filled'
    rep = fill_master(master, out, tiny_fill(), ACES2View(), FillSettings(steps=2, seed=1), MaskSettings(dilate=2),
                      window=5, overlap=2, max_area=48 * 64, fallback='spikes')
    frames = rep['frames']
    assert rep['fallback'] == 'spikes'
    assert frames[0]['fell_back'] is None and frames[-1]['fell_back'] is None   # nothing before / after them
    middle = [f for f in frames[1:-1] if f['temporal_excess_stops'] is not None]
    assert middle and all(f['fell_back'] == 'flicker' for f in middle)
    assert rep['stats']['fallback_frames'] == len(middle)
    for f in middle:
        a, b = read_frame(master / f['name']), read_frame(out / f['name'])
        assert not (b['channels']['rudra.generated'] > 0).any()
        for c in 'RGB':
            assert np.array_equal(a['channels'][c], b['channels'][c])


def test_no_fallback_is_v3(tmp_path):
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    master = tmp_path / 'master'
    write_master(master, n=5)
    a, b = tmp_path / 'a', tmp_path / 'b'
    args = dict(window=5, overlap=2, max_area=48 * 64)
    fill_master(master, a, tiny_fill(), ACES2View(), FillSettings(steps=2, seed=1), MaskSettings(dilate=2), **args)
    rb = fill_master(master, b, tiny_fill(), ACES2View(), FillSettings(steps=2, seed=1), MaskSettings(dilate=2),
                     fallback='none', **args)
    assert rb['stats']['fallback_frames'] == 0
    for name in sorted(p.name for p in master.glob('*.exr')):
        x, y = read_frame(a / name), read_frame(b / name)
        for c in 'RGB':
            assert np.array_equal(x['channels'][c], y['channels'][c])
    with pytest.raises(ValueError):
        fill_master(master, tmp_path / 'c', tiny_fill(), ACES2View(), FillSettings(steps=2), MaskSettings(),
                    fallback='sometimes', **args)
