"""1.2.9 (22 Nov 2026): window continuation for the v4 independent windows, and the window-join measure.

  model.denoise(known=, pin=)   a window's first latent frames held at the previous window's picture
  run.follow_frames/latents     which frames the two windows share, encoded as whole latent frames
  Rerender.follow / --rerender-follow   off by default: 1.2.8's windows exactly
  tools.window_joins            the step at each handover against the shot's own change
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.window_joins import frame_measures, join_report, join_steps, plan_joins  # noqa: E402


# ------------------------------------------------------------------------------ window joins
def test_the_joins_are_the_windows_handovers():
    assert plan_joins(97, 33, 8) == [(25, 33), (50, 58), (75, 83)]
    assert plan_joins(97, 33, 8, full_last=True) == [(25, 33), (50, 58), (64, 83)]
    assert plan_joins(30, 33, 8) == []


def test_full_last_makes_the_last_window_whole_and_changes_nothing_else():
    from rudra.genfill.run import plan_windows
    assert plan_windows(97, 33, 8, full_last=True) == [(0, 0, 33), (25, 33, 58), (50, 58, 83), (64, 83, 97)]
    assert plan_windows(97, 33, 8)[-1] == (75, 83, 97)                   # off: as 1.2.8
    assert plan_windows(20, 33, 8, full_last=True) == [(0, 0, 20)]       # shorter than a window: one window
    for n in range(1, 140):
        for ov in (0, 8, 16):
            p = plan_windows(n, 33, ov, full_last=True)
            written = [i for s, nw, e in p for i in range(nw, e)]
            assert written == list(range(n))                              # every frame written once, in order
            assert all(e - s <= 33 for s, nw, e in p)
            assert all(e - s == 33 for s, nw, e in p) or len(p) == 1     # every window whole


def test_a_step_at_a_join_reads_as_a_large_ratio_and_a_smooth_shot_as_about_one():
    n = 97
    smooth = np.linspace(0.0, 1.0, n)                  # the shot's own drift
    stepped = smooth.copy()
    stepped[33:] += 0.4                                # window 2 drew a brighter sky
    joins = plan_joins(n)
    calm = join_steps(smooth, joins)
    jump = join_steps(stepped, joins)
    assert all(abs(j['ratio'] - 1.0) < 0.01 for j in calm)
    assert jump[0]['ratio'] > 4 and all(abs(j['ratio'] - 1.0) < 0.01 for j in jump[1:])
    rep = join_report({'level': stepped}, n)
    assert rep['level']['max_ratio'] == jump[0]['ratio']


def test_frames_without_a_region_are_left_out():
    v = np.linspace(0, 1, 97)
    v[24] = np.nan
    s = join_steps(v, plan_joins(97))
    assert s[0]['ratio'] is None and s[1]['ratio'] is not None


def test_frame_measures_reads_the_region_and_the_boxes():
    img = np.full((50, 30, 3), 0.5, np.float32)
    img[2:6, 2:6] = 8.0                                # the lamp, far from the region (blur reach 9 px)
    gen = np.zeros((50, 30), np.float32)
    gen[25:, :] = 1
    m = frame_measures(img, gen, {'lamp': (2, 2, 6, 6)})
    assert m['level'] == pytest.approx(-1.0, abs=1e-4)
    assert m['texture'] == pytest.approx(0.0, abs=1e-4)
    assert m['lamp50'] == pytest.approx(8.0, rel=1e-4) and m['lamp99'] == pytest.approx(8.0, rel=1e-4)


# --------------------------------------------------------------------------- follow: the frames
def test_follow_counts_whole_latent_frames_of_the_shared_frames():
    from rudra.genfill.run import follow_frames
    def carry(frames, lane=0):
        return {i: {lane: None} for i in frames}
    assert follow_frames(carry(range(25, 33)), 25, 33, 0) == 5       # overlap 8: latent frames 0-1
    assert follow_frames(carry(range(17, 33)), 17, 33, 0) == 13      # overlap 16: latent frames 0-3
    assert follow_frames(carry(range(25, 34)), 25, 34, 0) == 9       # overlap 9: 0-2
    assert follow_frames(carry([25]), 25, 33, 0) == 1
    assert follow_frames(carry(range(25, 33), lane=1), 25, 33, 0) == 0   # another lane's
    assert follow_frames(carry([26, 27]), 25, 33, 0) == 0            # nothing from the start
    assert follow_frames({}, 25, 33, 0) == 0


def test_the_cli_and_the_settings_keep_follow_off_by_default():
    from rudra.genfill.run import Rerender
    assert Rerender('x').follow == 0.0


# ------------------------------------------------------------------------- follow: the model
def _needs(*mods):
    for m in mods:
        pytest.importorskip(m)                         # the optional genfill extra; CI runs without it


def _window(fill, frames=9, h=32, w=32, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    video = torch.rand(1, 3, frames, h, w, generator=g) * 2 - 1
    mask = torch.zeros(1, 1, frames, h, w)
    mask[..., 8:24, 8:24] = 1
    return fill.context(video, mask)


def test_known_none_or_pin_zero_is_the_window_as_before():
    _needs('torch', 'diffusers')
    import torch
    from genfill_tiny import tiny_fill
    from rudra.genfill.model import FillSettings
    fill = tiny_fill()
    ctx = _window(fill)
    s = FillSettings(steps=4, seed=2)
    a = fill.denoise(ctx, s)
    known = torch.randn(1, 16, 2, *ctx.shape[3:])
    assert torch.equal(a, fill.denoise(ctx, s, known=None, pin=0.7))
    assert torch.equal(a, fill.denoise(ctx, s, known=known, pin=0.0))


def test_pin_one_ends_on_the_known_frames_and_the_rest_follows_them():
    _needs('torch', 'diffusers')
    import torch
    from genfill_tiny import tiny_fill
    from rudra.genfill.model import FillSettings
    fill = tiny_fill()
    ctx = _window(fill)
    s = FillSettings(steps=4, seed=2)
    free = fill.denoise(ctx, s)
    known = torch.randn(1, 16, 2, *ctx.shape[3:])
    held = fill.denoise(ctx, s, known=known, pin=1.0)
    assert torch.allclose(held[:, :, :2], known, atol=1e-5)          # sigma 0 at the last step: x0 itself
    assert not torch.allclose(held[:, :, 2:], free[:, :, 2:])         # the rest of the window responds
    half = fill.denoise(ctx, s, known=known, pin=0.5)
    d = lambda z: float((z[:, :, :2] - known).abs().mean())
    assert d(half) < d(free)                                         # held half the way: nearer than free


def test_known_latents_that_do_not_fit_are_refused():
    _needs('torch', 'diffusers')
    import torch
    from genfill_tiny import tiny_fill
    from rudra.genfill.model import FillSettings
    fill = tiny_fill()
    ctx = _window(fill)
    with pytest.raises(ValueError):
        fill.denoise(ctx, FillSettings(steps=2), known=torch.zeros(1, 16, 9, 2, 2), pin=1.0)


def test_follow_latents_encodes_the_carried_frames():
    _needs('torch', 'diffusers')
    from genfill_tiny import tiny_fill
    from rudra.genfill.run import follow_latents
    fill = tiny_fill()
    rng = np.random.default_rng(0)
    carry = {i: {0: rng.uniform(-1, 1, (32, 32, 3)).astype(np.float32)} for i in range(25, 33)}
    z = follow_latents(fill, carry, 25, 33, 0)
    assert tuple(z.shape) == (1, 16, 2, 4, 4)
    assert follow_latents(fill, carry, 25, 33, 1) is None


# ----------------------------------------------------------------------- follow: the whole fill
def test_follow_writes_every_frame_once_and_leaves_the_first_window_alone(tmp_path):
    _needs('torch', 'diffusers', 'OpenEXR', 'PyOpenColorIO')
    import cv2
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    from rudra.genfill.masks import MaskSettings
    from rudra.genfill.model import FillSettings
    from rudra.genfill.run import Rerender, fill_master, read_frame
    from rudra.genfill.synthetic import write_master
    view = ACES2View()
    master = tmp_path / 'master'
    write_master(master, n=13)
    names = sorted(p.name for p in master.glob('*.exr'))
    src = tmp_path / 'source'
    src.mkdir()
    for k, name in enumerate(names):
        c = np.rint(np.clip(view.forward(read_frame(master / name)['ap1']), 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(src / f'f_{k:04d}.png'), c[..., ::-1])
    reps = {}
    for follow in (0.0, 0.6):
        out = tmp_path / f'filled_{follow}'
        events = []
        reps[follow] = fill_master(master, out, tiny_fill(), view, FillSettings(steps=3, seed=3),
                                   MaskSettings(dilate=2, feather=3, tau_high=240, tau_low=-1), window=9, overlap=5,
                                   max_area=48 * 64, emit=lambda **o: events.append(o),
                                   rerender=Rerender(str(src), 'bt1886', independent=True, follow=follow))
        assert [e['index'] for e in events if e['event'] == 'frame'] == list(range(1, len(names) + 1))
        assert sorted(p.name for p in out.glob('*.exr')) == names
        for name in names:
            a, b = read_frame(master / name), read_frame(out / name)
            untouched = b['channels']['rudra.generated'] <= 0
            for c in 'RGB':
                assert np.array_equal(a['channels'][c][untouched].view(np.uint8), b['channels'][c][untouched].view(np.uint8))
    assert 'window_follow' not in reps[0.0]['stats']
    wf = reps[0.6]['stats']['window_follow']
    assert wf and all(x['latent_frames'] == 2 and x['pixel_frames'] == 5 and x['pin'] == 0.6 for x in wf)
    assert reps[0.6]['rerender']['follow'] == 0.6
    a = read_frame(tmp_path / 'filled_0.0' / names[0])['ap1']           # window 1 has nothing to follow
    b = read_frame(tmp_path / 'filled_0.6' / names[0])['ap1']
    assert np.array_equal(a, b)
    a = read_frame(tmp_path / 'filled_0.0' / names[-1])['ap1']          # the later windows do
    b = read_frame(tmp_path / 'filled_0.6' / names[-1])['ap1']
    assert not np.array_equal(a, b)


def test_full_last_writes_every_frame_once_and_follow_holds_more(tmp_path):
    """1.2.9 --rerender-full-last: 13 frames, window 9, overlap 2 -> windows 0-8 and 4-12 (not 7-12 padded):
    every frame written once in order, the master kept outside the fill, and with follow the 5 shared
    frames (4-8) held, not 1."""
    _needs('torch', 'diffusers', 'OpenEXR', 'PyOpenColorIO')
    import cv2
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    from rudra.genfill.masks import MaskSettings
    from rudra.genfill.model import FillSettings
    from rudra.genfill.run import Rerender, fill_master, read_frame
    from rudra.genfill.synthetic import write_master
    view = ACES2View()
    master = tmp_path / 'master'
    write_master(master, n=13)
    names = sorted(p.name for p in master.glob('*.exr'))
    src = tmp_path / 'source'
    src.mkdir()
    for k, name in enumerate(names):
        c = np.rint(np.clip(view.forward(read_frame(master / name)['ap1']), 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(src / f'f_{k:04d}.png'), c[..., ::-1])
    reps = {}
    for tag, kw in (('short', {}), ('full', {'full_last': True}), ('full_follow', {'full_last': True, 'follow': 0.6})):
        out = tmp_path / tag
        events = []
        reps[tag] = fill_master(master, out, tiny_fill(), view, FillSettings(steps=3, seed=3),
                                MaskSettings(dilate=2, feather=3, tau_high=240, tau_low=-1), window=9, overlap=2,
                                max_area=48 * 64, emit=lambda **o: events.append(o),
                                rerender=Rerender(str(src), 'bt1886', independent=True, **kw))
        assert [e['index'] for e in events if e['event'] == 'frame'] == list(range(1, len(names) + 1))
        for name in names:
            a, b = read_frame(master / name), read_frame(out / name)
            untouched = b['channels']['rudra.generated'] <= 0
            for c in 'RGB':
                assert np.array_equal(a['channels'][c][untouched].view(np.uint8), b['channels'][c][untouched].view(np.uint8))
    assert reps['full']['rerender']['full_last'] is True and reps['short']['rerender']['full_last'] is False
    assert [x['pixel_frames'] for x in reps['full_follow']['stats']['window_follow']] == [5]
    for k in range(4):                                     # frames only window 1 writes: the same either way
        a = read_frame(tmp_path / 'short' / names[k])['ap1']
        b = read_frame(tmp_path / 'full' / names[k])['ap1']
        assert np.array_equal(a, b)


def test_follow_outside_zero_to_one_is_refused(tmp_path):
    _needs('torch', 'diffusers', 'OpenEXR', 'PyOpenColorIO')
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    from rudra.genfill.masks import MaskSettings
    from rudra.genfill.model import FillSettings
    from rudra.genfill.run import Rerender, fill_master
    from rudra.genfill.synthetic import write_master
    master = tmp_path / 'master'
    write_master(master, n=3)
    with pytest.raises(ValueError, match='follow'):
        fill_master(master, tmp_path / 'o', tiny_fill(), ACES2View(), FillSettings(steps=1), MaskSettings(),
                    rerender=Rerender(str(master), 'bt1886', independent=True, follow=1.5))
