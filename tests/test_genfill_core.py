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
