"""A tiny, random Wan2.1-VACE (the shapes of the real one, a few thousand parameters) for the
genfill tests: the VAE, transformer and scheduler are diffusers' own classes, so the tests run
the real conditioning, denoising, LoRA and training code paths in seconds on a CPU."""
import torch

TEXT_DIM = 32


def tiny_components(seed=0):
    import pytest
    pytest.importorskip('diffusers')   # the optional genfill extra; CI runs without it
    from diffusers import AutoencoderKLWan, WanVACETransformer3DModel
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    torch.manual_seed(seed)
    vae = AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True]).eval()
    tr = WanVACETransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
                                   in_channels=16, out_channels=16, text_dim=TEXT_DIM, freq_dim=256, ffn_dim=32,
                                   num_layers=3, cross_attn_norm=True, qk_norm='rms_norm_across_heads',
                                   rope_max_seq_len=32, vace_layers=[0, 2], vace_in_channels=96).eval()
    sch = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, flow_shift=3.0)
    return vae, tr, sch


def tiny_fill(seed=0):
    from rudra.genfill.model import BASE_PROMPT, PromptBank, WanFill
    vae, tr, sch = tiny_components(seed)
    pb = PromptBank(None, None)
    g = torch.Generator().manual_seed(1)
    for text in (BASE_PROMPT, 'a bright sun', 'deep blue shadow', ''):
        pb.put(text, torch.randn(1, 512, TEXT_DIM, generator=g))
    return WanFill(vae, tr, sch, pb, device='cpu', dtype=torch.float32)
