"""--focus region (7 Nov 2026): the model draws only the box around what the shot fills, one box
for the whole shot, so the same working size gives the fill more detail; outside the box every
pixel is the master's. The box itself is torch-free; the fill runs on the tiny random Wan VACE."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rudra.genfill.run import FOCUS_MAX_SHARE, FOCUS_MIN_MARGIN, FOCUS_MIN_SIDE, focus_box


def test_the_box_holds_the_masks_with_context_and_stays_in_the_frame():
    """A door a third of the frame wide: the box holds it with context, inside the frame."""
    u = np.zeros((1080, 1920), bool)
    u[100:600, 300:900] = True                       # a door a third of the frame wide
    y0, y1, x0, x1 = focus_box(u)
    assert y0 <= 100 - FOCUS_MIN_MARGIN and y1 >= 600 + FOCUS_MIN_MARGIN
    assert x0 <= 300 - 90 and x1 >= 900 + 90          # 15 % of 600 px each side
    assert 0 <= y0 < y1 <= 1080 and 0 <= x0 < x1 <= 1920
    assert u[y0:y1, x0:x1].sum() == u.sum()          # every masked pixel inside


def test_a_speck_gets_a_minimum_box_and_an_edge_box_is_shifted_in():
    """A glint in a corner gets at least the minimum box, moved inside the frame."""
    u = np.zeros((1080, 1920), bool)
    u[0:3, 1915:1920] = True                          # a glint in the corner
    y0, y1, x0, x1 = focus_box(u)
    assert (y1 - y0) >= FOCUS_MIN_SIDE and (x1 - x0) >= FOCUS_MIN_SIDE
    assert y0 == 0 and x1 == 1920


def test_nothing_or_most_of_the_frame_means_the_whole_frame():
    """No mask, or a box of most of the frame: no box (the whole frame)."""
    assert focus_box(np.zeros((100, 200), bool)) is None
    u = np.zeros((1080, 1920), bool)
    u[50:1000, 50:1800] = True
    assert focus_box(u) is None
    v = np.zeros((1080, 1920), bool)
    v[0:5, 0:5] = v[1075:, 1915:] = True               # two far corners: the box is the frame
    assert focus_box(v) is None and FOCUS_MAX_SHARE < 1


def test_specks_far_away_do_not_stretch_the_box():
    """Specks far from the main area stay out of the box; covering them all means the whole frame."""
    u = np.zeros((1080, 1920), bool)
    u[200:700, 300:900] = True                        # the blown door
    u[50:55, 1800:1805] = u[1000:1004, 20:24] = True   # a lamp and a glint in far corners
    y0, y1, x0, x1 = focus_box(u)
    assert (y1 - y0) * (x1 - x0) < 0.3 * u.size and u[y0:y1, x0:x1][100:500].any()
    assert focus_box(u, cover=1.0) is None             # everything in it: the whole frame


def test_the_feather_is_inside_the_box():
    """The blend's reach past the mask (the feather) is inside the box."""
    u = np.zeros((1000, 1000), bool)
    u[500:520, 500:520] = True
    for extra in (0, 40, 150):                        # the blend reaches `extra` px past the mask
        y0, y1, x0, x1 = focus_box(u, extra=extra)
        assert min(500 - y0, y1 - 520, 500 - x0, x1 - 520) >= extra + FOCUS_MIN_MARGIN


def test_focus_region_keeps_the_contract_and_draws_only_in_the_box(tmp_path):
    """On the tiny Wan VACE: more detail, nothing drawn outside the box, every SDR code kept."""
    pytest.importorskip('diffusers')
    pytest.importorskip('OpenEXR')
    pytest.importorskip('PyOpenColorIO')
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    from rudra.genfill.masks import MaskSettings
    from rudra.genfill.model import FillSettings
    from rudra.genfill.run import fill_master, read_frame
    from rudra.genfill.synthetic import write_master
    view = ACES2View()
    master = tmp_path / 'master'
    write_master(master, n=5, h=360, w=640)           # a small sun in a wide frame
    reps = {}
    for focus in ('frame', 'region'):
        reps[focus] = fill_master(master, tmp_path / focus, tiny_fill(), view, FillSettings(steps=2, seed=1),
                                  MaskSettings(dilate=2), window=5, overlap=2, max_area=48 * 64, focus=focus)
    f, r = reps['frame']['stats']['focus'], reps['region']['stats']['focus']
    assert f == {'mode': 'frame', 'box': None, 'mask_outside_box': 0.0, 'detail_gain': 1.0}
    y0, y1, x0, x1 = r['box']
    assert r['detail_gain'] > 1.5                      # the same working size, spent on the box
    assert reps['region']['stats']['generated_pixels'] > 0
    for name in sorted(p.name for p in master.glob('*.exr')):
        a, b = read_frame(master / name), read_frame(tmp_path / 'region' / name)
        gen = b['channels']['rudra.generated'] > 0
        outside = np.ones(gen.shape, bool)
        outside[y0:y1, x0:x1] = False
        assert not gen[outside].any()
        for c in 'RGB':                                # untouched: bit-identical
            assert np.array_equal(a['channels'][c][~gen].view(np.uint8), b['channels'][c][~gen].view(np.uint8))
        codes = lambda fr: np.rint(np.clip(view.forward(fr['ap1']), 0, 1) * 255)
        assert np.abs(codes(a) - codes(b)).max() == 0  # float32 master: every SDR code kept


def test_focus_is_checked(tmp_path):
    """An unknown focus is refused."""
    pytest.importorskip('diffusers')
    from genfill_tiny import tiny_fill
    from rudra.genfill.masks import MaskSettings
    from rudra.genfill.model import FillSettings
    from rudra.genfill.run import fill_master
    with pytest.raises(ValueError, match='focus'):
        fill_master(tmp_path, tmp_path / 'o', tiny_fill(), None, FillSettings(), MaskSettings(), focus='crop')


# ---------------------------------------------------------------- --focus areas (8 Nov 2026)
from rudra.genfill.run import FOCUS_MAX_BOXES, focus_boxes  # noqa: E402


def _overlapping(boxes):
    return any(a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]
               for i, a in enumerate(boxes) for b in boxes[i + 1:])


def test_two_lamps_at_opposite_edges_of_4k_get_a_box_each():
    """Far-apart highlights in a 4K frame: one small box each (each drawn at full resolution), where
    focus_box would need one box nearly as big as the frame."""
    u = np.zeros((2160, 3840), bool)
    u[300:420, 200:360] = True                         # a lamp at the left
    u[1700:1820, 3500:3660] = True                     # a lamp at the right
    assert focus_box(u, cover=1.0) is None             # one box: most of the frame, so none
    boxes = focus_boxes(u)
    assert len(boxes) == 2 and not _overlapping(boxes)
    for y0, y1, x0, x1 in boxes:
        assert (y1 - y0) * (x1 - x0) <= 832 * 480      # each fits the working size: drawn 1:1
    inside = np.zeros_like(u)
    for y0, y1, x0, x1 in boxes:
        inside[y0:y1, x0:x1] = True
    assert u[inside].sum() == u.sum()                  # every masked pixel in a box


def test_close_areas_merge_and_boxes_never_overlap():
    """Areas whose boxes would overlap become one box; the result never overlaps."""
    u = np.zeros((1080, 1920), bool)
    u[400:440, 400:440] = u[400:440, 470:510] = True   # two lights 30 px apart
    u[900:920, 1700:1720] = True                       # one far away
    boxes = focus_boxes(u)
    assert len(boxes) == 2 and not _overlapping(boxes)


def test_many_areas_are_capped_by_merging_the_closest():
    """A starfield of small highlights: at most FOCUS_MAX_BOXES boxes, none overlapping."""
    u = np.zeros((2160, 3840), bool)
    rng = np.random.default_rng(3)
    for y, x in zip(rng.integers(0, 2100, 40), rng.integers(0, 3780, 40)):
        u[y:y + 6, x:x + 6] = True
    boxes = focus_boxes(u, cover=1.0)
    if boxes is not None:                              # or the frame, when they cover most of it
        assert len(boxes) <= FOCUS_MAX_BOXES and not _overlapping(boxes)


def test_areas_is_none_when_there_is_nothing_or_most_of_the_frame():
    """No mask, or boxes over most of the frame: the whole frame."""
    assert focus_boxes(np.zeros((100, 200), bool)) is None
    u = np.zeros((1080, 1920), bool)
    u[50:1000, 50:1800] = True
    assert focus_boxes(u) is None


def test_one_area_is_the_same_box_as_region():
    """A single area: --focus areas and --focus region draw the same box."""
    u = np.zeros((1080, 1920), bool)
    u[100:600, 300:900] = True
    assert focus_boxes(u) == [focus_box(u)]


def test_focus_areas_keeps_the_contract_and_draws_only_in_its_boxes(tmp_path):
    """On the tiny Wan VACE with two suns far apart: a box each, nothing drawn outside them, every
    SDR code kept, untouched pixels bit-identical."""
    pytest.importorskip('diffusers')
    pytest.importorskip('OpenEXR')
    pytest.importorskip('PyOpenColorIO')
    from genfill_tiny import tiny_fill
    from rudra.aces_view import ACES2View
    from rudra.genfill.masks import MaskSettings
    from rudra.genfill.model import FillSettings
    import OpenEXR
    from rudra.genfill.run import fill_master, read_frame
    from rudra.genfill.synthetic import AP1
    view = ACES2View()
    master = tmp_path / 'master'
    master.mkdir()
    h, w = 360, 640
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for i in range(5):                                 # two suns in opposite corners, moving a little
        base = 0.05 + 0.5 * (xx / w)[..., None] * np.array([1.0, 0.9, 0.8], np.float32)
        sun = lambda cx, cy: 500.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / 60.0))[..., None]
        img = (base + sun(30 + 2 * i, 30) + sun(600 - 2 * i, 320)).astype(np.float32)
        ch = {c: np.ascontiguousarray(img[..., k]) for k, c in enumerate('RGB')}   # OpenEXR reads the raw buffer
        ch['rudra.reconstructed'] = np.zeros((h, w), np.float32)
        hdr = {'compression': OpenEXR.NO_COMPRESSION, 'type': OpenEXR.scanlineimage, 'chromaticities': AP1}
        OpenEXR.File(hdr, ch).write(str(master / f'shot.{1001 + i:06d}.exr'))
    rep = fill_master(master, tmp_path / 'areas', tiny_fill(), view, FillSettings(steps=2, seed=1),
                      MaskSettings(dilate=2), window=5, overlap=2, max_area=48 * 64, focus='areas')
    f = rep['stats']['focus']
    assert f['mode'] == 'areas' and len(f['boxes']) >= 2
    assert rep['stats']['generated_pixels'] > 0
    inside = np.zeros((360, 640), bool)
    for b in f['boxes']:
        y0, y1, x0, x1 = b['box']
        inside[y0:y1, x0:x1] = True
    for name in sorted(p.name for p in master.glob('*.exr')):
        a, b = read_frame(master / name), read_frame(tmp_path / 'areas' / name)
        gen = b['channels']['rudra.generated'] > 0
        assert not gen[~inside].any()
        for c in 'RGB':
            assert np.array_equal(a['channels'][c][~gen].view(np.uint8), b['channels'][c][~gen].view(np.uint8))
        codes = lambda fr: np.rint(np.clip(view.forward(fr['ap1']), 0, 1) * 255)
        assert np.abs(codes(a) - codes(b)).max() == 0
