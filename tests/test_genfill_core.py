"""Layer 3 (generative fill) core: encoding, masks, the round-trip-safe composite, the VACE
conditioning against diffusers' own pipeline, seeds and the LoRA file."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip('diffusers')
pytest.importorskip('peft')

from rudra.genfill import encoding
from rudra.genfill.composite import r_safe_blend
from rudra.genfill.masks import MaskSettings, TemporalMask, feather, frame_masks
from rudra.genfill.model import FillSettings, add_lora, load_lora, save_lora


def test_encoding_round_trip_and_range():
    x = np.concatenate([[0.0, 1e-6, 0.0078125, 0.18, 1.0, 16.0, 1000.0, 2.0 ** 14],
                        np.geomspace(1e-5, 2 ** 14, 200)]).astype(np.float64)
    y = encoding.encode(x)
    assert y.min() >= -1 and y.max() <= 1 and np.all(np.diff(y[8:]) > 0)
    assert y[0] == pytest.approx(-1) and y[7] == pytest.approx(1)
    assert np.allclose(encoding.decode(y), x, rtol=1e-9, atol=1e-12)
    assert encoding.encode(np.array([0.18]))[0] == pytest.approx(-0.468, abs=2e-3)
    t = torch.tensor(x)
    assert torch.allclose(encoding.decode(encoding.encode(t)), t, rtol=1e-9, atol=1e-12)
    assert encoding.encode(np.array([1e9, -3.0])).tolist() == pytest.approx([1.0, -1.0])


def test_masks_thresholds_dilation_and_time():
    codes = np.full((40, 60, 3), 120, np.uint8)
    codes[10:14, 10:14] = 255
    codes[30:33, 40:44] = (2, 3, 1)
    hi, lo = frame_masks(codes, MaskSettings(dilate=0, tau_low=4))
    assert hi.sum() == 16 and lo.sum() == 12 and not (hi & lo).any()
    hi2, _ = frame_masks(codes, MaskSettings(dilate=2))
    assert hi2.sum() > 16 and hi2[10:14, 10:14].all()
    off, _ = frame_masks(codes, MaskSettings(tau_high=256, tau_low=-1))
    assert not off.any()
    tm = TemporalMask(MaskSettings(dilate=0, ema=0.7, tau_low=4))
    tm(codes)
    flicker = codes.copy(); flicker[10:14, 10:14] = 120       # gone for one frame: kept (EMA 0.7)
    assert tm(flicker)[0][10:14, 10:14].all()
    w = feather(hi, 4)
    assert w[hi].min() == 1 and 0 < w[9, 12] < 1 and w[30, 5] == 0
    with pytest.raises(ValueError):
        MaskSettings(tau_high=4, tau_low=4).check()
    # v1.1: the blend core is only the fully clipped (and, with shadows on, fully crushed) codes
    from rudra.genfill.masks import blend_core
    c = np.full((4, 4, 3), 120, np.uint8); c[0, 0] = 255; c[0, 1] = (254, 254, 254); c[1, 0] = (255, 10, 10)
    c[3, 3] = 0; c[3, 2] = 2
    core = blend_core(c, MaskSettings())
    assert core[0, 0] and core[1, 0] and not core[0, 1] and not core[3, 3]       # shadows off by default
    core = blend_core(c, MaskSettings(tau_low=4))
    assert core[3, 3] and not core[3, 2]
    assert MaskSettings().tau_low == -1 and MaskSettings().feather == 0


class _View:
    """A stand-in view: Reinhard-like display curve, clipped, per channel (monotone)."""
    def forward(self, x):
        x = np.maximum(x, 0)
        return np.clip(1.25 * x / (1 + x), 0, 1)


def test_r_safe_blend_keeps_every_code():
    rng = np.random.default_rng(0)
    master = rng.uniform(0, 6, (50, 70, 3)).astype(np.float32)
    master[:20] = 8.0                       # clipped (view = 1 above 4)
    master[40:] = 0.0005                    # crushed (code 0)
    fill = (master * rng.uniform(0.3, 30, master.shape)).astype(np.float32)
    weight = np.ones((50, 70), np.float32)
    v = _View()
    out, alpha = r_safe_blend(master, fill, weight, v)
    codes = lambda a: np.rint(np.clip(v.forward(a), 0, 1) * 255)
    assert (codes(out) == codes(master)).all()
    assert alpha[:20].mean() > 0.99                   # free above the clip
    assert alpha[45:].mean() > 0.0                     # shadows may move below the code edge
    assert alpha[20:40].mean() < 0.6                   # mid-tones held near the master
    out0, a0 = r_safe_blend(master, fill, np.zeros((50, 70), np.float32), v)
    assert (out0 == master).all() and not a0.any()


def test_plain_blend_moves_codes_inside_and_nothing_outside():
    """v4 (9 Nov 2026): the region re-render's composite keeps nothing inside the mask and everything outside."""
    from rudra.genfill.composite import plain_blend
    rng = np.random.default_rng(1)
    master = rng.uniform(0.5, 1.2, (30, 40, 3)).astype(np.float32)
    fill = (master * 3.0).astype(np.float32)
    weight = np.zeros((30, 40), np.float32)
    weight[5:20, 10:30] = 1.0
    weight[20, 10:30] = 0.5
    out, alpha = plain_blend(master, fill, weight)
    assert (out[weight == 0] == master[weight == 0]).all()
    assert np.allclose(out[5:20, 10:30], fill[5:20, 10:30])
    assert np.allclose(out[20, 10:30], 2.0 * master[20, 10:30])
    assert (alpha == weight).all()
    v = _View()
    codes = lambda a: np.rint(np.clip(v.forward(a), 0, 1) * 255)
    assert (codes(out)[5:20, 10:30] != codes(master)[5:20, 10:30]).any()   # the codes inside are not kept


def test_condition_of_bt1886_is_the_display_light_in_ap1():
    from rudra.genfill.run import condition_of
    codes = np.zeros((2, 3, 3), np.uint8)
    codes[0, 0] = 255                       # white: 1.0 = diffuse white, neutral in AP1
    codes[0, 1] = 128
    cond = condition_of(codes, 'bt1886', view=None)
    assert cond.shape == codes.shape and cond.dtype == np.float32
    assert np.allclose(cond[0, 0], 1.0, atol=2e-3)
    assert np.allclose(cond[0, 1], (128 / 255) ** 2.4, atol=2e-3)
    assert (cond[1] == 0).all()


def test_finish_rerender_is_off_by_default_and_sets_hue_level_and_time():
    from rudra.genfill.composite import LUMA_AP1, finish_rerender, white_scale
    rng = np.random.default_rng(3)
    h, w = 96, 128
    codes = np.full((h, w, 3), 120, np.uint8)
    codes[20:80, 30:110] = 254
    codes[20:80, 30:50] = 246                                                   # a darker band inside the region
    grey = np.power(codes.astype(np.float32) / 255.0, 2.4)
    source = grey * np.array([1.0, 1.05, 1.05], np.float32)
    master = grey * 0.9
    fill = (rng.uniform(1.0, 2.0, (h, w, 1)) * np.array([0.6, 1.0, 2.0])).astype(np.float32)   # blue, about 1.4
    weight = np.zeros((h, w), np.float32)
    weight[20:80, 30:110] = 1.0
    inside = weight > 0
    lum = lambda x: x @ LUMA_AP1                                                # noqa: E731
    assert finish_rerender(fill, master, source, codes, weight) is fill         # every step off: untouched
    white = white_scale(master, codes)
    assert abs(white - 0.9) < 1e-3
    out = finish_rerender(fill, master, source, codes, weight, hue='source', level=3.0, white=white)
    assert np.array_equal(out[~inside], fill[~inside])                          # outside the weight: as given
    assert abs(float(np.median(lum(out[30:70, 70:100]))) / white - 3.0) < 0.15  # clipped: the level, in SDR whites
    assert float(np.median(lum(out[30:70, 34:42]))) < 0.8 * float(np.median(lum(out[30:70, 70:100])))   # darker source, darker fill
    from rudra.genfill.composite import RERENDER_CEILING_STOPS
    bright = finish_rerender(fill * 64.0, master, source, codes, weight, level=3.0, white=white)   # the model far brighter
    top = 3.0 * 2.0 ** RERENDER_CEILING_STOPS                                   # its own level only to the ceiling (0: not at all)
    assert abs(float(np.median(lum(bright[30:70, 70:100]))) / white - top) < 0.05 * top
    edge = finish_rerender(fill, master, source, np.full_like(codes, 240), weight, hue='source', level=3.0, white=white)
    assert np.allclose(edge[inside], master[inside], rtol=1e-3, atol=1e-5)      # at tau_high the fill is the master: no step
    core = out[30:70, 70:100].reshape(-1, 3).mean(0)
    assert abs(core[1] / core[0] - 1.05) < 0.01 and abs(core[2] / core[0] - 1.05) < 0.01   # the source's colour
    mixed = finish_rerender(fill, master, source, codes, weight, hue='source', colour=0.5)[30:70, 70:100].reshape(-1, 3).mean(0)
    assert 1.05 < mixed[2] / mixed[0] < 2.0 / 0.6                               # between the source's and the model's
    assert np.allclose(lum(finish_rerender(fill, master, source, codes, weight, hue='source')[inside]), lum(fill[inside]), rtol=1e-4)
    state = {}
    a = finish_rerender(fill, master, source, codes, weight, smooth=0.8, state=state)
    assert np.allclose(a[inside], fill[inside], rtol=1e-4)                      # first frame: nothing to follow
    b = finish_rerender(fill * 2.0, master, source, codes, weight, smooth=0.8, state=state)
    assert np.allclose(np.log2(lum(b[inside]) / lum(a[inside])), 0.2, atol=1e-3)   # a one-stop jump arrives as 0.2
    with pytest.raises(ValueError):
        finish_rerender(fill, master, source, codes, weight, hue='other')


def test_finish_rerender_holds_the_fill_where_the_source_holds_still():
    from rudra.genfill.composite import LUMA_AP1, finish_rerender
    h, w = 32, 48
    codes = np.full((h, w, 3), 252, np.uint8)
    source = np.power(codes.astype(np.float32) / 255.0, 2.4)
    fill = np.full((h, w, 3), 2.0, np.float32)
    weight = np.ones((h, w), np.float32)
    lum = lambda x: x @ LUMA_AP1                                                # noqa: E731
    state = {}
    a = finish_rerender(fill, source, source, codes, weight, smooth=0.5, still=0.9, state=state)
    moved = codes.copy()
    moved[:, : w // 2] = 240                                                    # the left half of the source changes
    b = finish_rerender(fill * 2.0, source, source, moved, weight, smooth=0.5, still=0.9, state=state)
    step = np.log2(lum(b) / lum(a))
    assert np.allclose(step[:, w // 2:], 0.1, atol=1e-3)                       # held: 0.9 kept, a stop arrives as 0.1
    assert np.allclose(step[:, : w // 2], 0.5, atol=1e-3)                      # moved: smooth 0.5 stands
    with pytest.raises(ValueError):
        finish_rerender(fill, source, source, codes, weight, still=1.0)


def test_neutral_weight_leaves_a_saturated_highlight_to_the_master():
    from rudra.genfill.composite import neutral_weight
    codes = np.array([[[254, 254, 254], [245, 254, 254], [255, 190, 120], [255, 210, 200], [0, 0, 0]]], np.uint8)
    w = neutral_weight(codes)[0]
    assert w[0] == 1.0 and w[1] == 1.0          # white, and a door a little cool: whole
    assert w[2] == 0.0                          # a sunset: red clipped, blue far below
    assert 0.0 < w[3] < 1.0                     # between: a soft change, not a step
    assert w[4] == 0.0                          # black: nothing to say (and no division by zero)


def test_r_safe_blend_with_the_aces_view():
    pytest.importorskip('PyOpenColorIO')
    from rudra.aces_view import ACES2View
    v = ACES2View()
    rng = np.random.default_rng(1)
    master = rng.uniform(0, 2, (24, 24, 3)).astype(np.float32)
    master[:8] = 400.0
    fill = master * 3
    out, alpha = r_safe_blend(master, fill, np.ones((24, 24), np.float32), v)
    c = lambda a: np.rint(np.clip(v.forward(a), 0, 1) * 255)
    assert (c(out) == c(master)).all() and alpha[:8].min() == 1.0


def test_context_matches_the_diffusers_pipeline():
    from diffusers import WanVACEPipeline
    from genfill_tiny import tiny_fill
    f = tiny_fill()
    pipe = WanVACEPipeline(tokenizer=None, text_encoder=None, vae=f.vae, scheduler=f.scheduler, transformer=f.transformer)
    g = torch.Generator().manual_seed(3)
    video = torch.rand(1, 3, 9, 32, 48, generator=g) * 2 - 1
    mask = (torch.rand(1, 1, 9, 32, 48, generator=g) > 0.6).float()
    ours = f.context(video, mask)
    m3 = mask.expand(1, 3, 9, 32, 48)
    ref = pipe.prepare_video_latents(video, m3, [[]], None, torch.device('cpu'))
    ref = torch.cat([ref, pipe.prepare_masks(m3, [[]], None)], 1)
    assert ours.shape == ref.shape == (1, 96, 3, 4, 6)
    assert torch.allclose(ours, ref.float(), atol=1e-5)


def test_denoise_is_seeded_and_region_prompts_act_locally():
    from genfill_tiny import tiny_fill
    f = tiny_fill()
    v = torch.rand(1, 3, 5, 32, 32) * 2 - 1
    m = torch.zeros(1, 1, 5, 32, 32); m[..., :16, :] = 1
    ctx = f.context(v, m)
    a = f.denoise(ctx, FillSettings(steps=3, seed=7))
    b = f.denoise(ctx, FillSettings(steps=3, seed=7))
    c = f.denoise(ctx, FillSettings(steps=3, seed=8))
    assert torch.equal(a, b) and not torch.equal(a, c)
    r = f.denoise(ctx, FillSettings(steps=3, seed=7, prompt_highlight='a bright sun'), {'highlight': m})
    assert not torch.allclose(r[..., :2, :], a[..., :2, :])           # inside the region
    # the far rows see the region only through attention, so differ less than inside
    assert (r[..., 3:, :] - a[..., 3:, :]).abs().mean() < (r[..., :2, :] - a[..., :2, :]).abs().mean()


def test_lora_round_trip(tmp_path):
    from genfill_tiny import tiny_components
    _, tr, _ = tiny_components()
    params = add_lora(tr, rank=4)
    assert params and all(p.requires_grad for p in params)
    assert not any(p.requires_grad for n, p in tr.named_parameters() if 'lora_' not in n)
    with torch.no_grad():
        for p in params:
            p.add_(torch.randn_like(p) * 0.05)
    save_lora(tr, tmp_path / 'l.safetensors', {'run': 'test'})
    _, tr2, _ = tiny_components()
    meta = load_lora(tr2, tmp_path / 'l.safetensors')
    assert meta['run'] == 'test' and meta['rank'] == '4'
    x = torch.randn(1, 16, 2, 4, 4); ctx = torch.randn(1, 96, 2, 4, 4); e = torch.randn(1, 8, 32)
    kw = dict(timestep=torch.tensor([500]), encoder_hidden_states=e, control_hidden_states=ctx,
              control_hidden_states_scale=torch.ones(2), return_dict=False)
    with torch.no_grad():
        assert torch.allclose(tr(hidden_states=x, **kw)[0], tr2(hidden_states=x, **kw)[0], atol=1e-6)
    _, tr3, _ = tiny_components()
    with torch.no_grad():
        assert not torch.allclose(tr(hidden_states=x, **kw)[0], tr3(hidden_states=x, **kw)[0], atol=1e-4)
    bad = tmp_path / 'bad.safetensors'
    from safetensors.torch import save_file
    save_file({'x': torch.zeros(1)}, str(bad))
    with pytest.raises(ValueError):
        load_lora(tiny_components()[1], bad)
