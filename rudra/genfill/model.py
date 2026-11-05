"""The Wan2.1-VACE components for the fill: VAE (float32), transformer (bf16) + RUDRA LoRA,
flow-matching scheduler, prompt embeddings (umT5, loaded only to embed, then freed, cached).

Mirrors diffusers' WanVACEPipeline (0.35+) step for step, on tensors instead of PIL images, so
the HDR encoding is never squeezed through 8 bits:

    context  = [ VAE(video * (1 - m)), VAE(video * m) ]  (16 + 16 channels, normalised latents)
             + the mask folded 8x8 into 64 channels at the latent grid           -> 96 channels
    denoise  = flow matching from noise, the transformer's control branch reading the context

Region prompts: one prediction per prompt each step, blended by the masks at the latent grid
(the base prompt everywhere else). With no region prompt, one prediction a step.

The VAE runs in float32 on purpose: at bf16 the decoder bands smooth dark gradients.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

BASE_PROMPT = ('scene-referred high dynamic range footage, natural light, physically plausible '
               'highlights and shadow detail')
MAX_SEQUENCE_LENGTH = 512
LORA_TARGETS = ['to_q', 'to_k', 'to_v', 'to_out.0', 'ffn.net.0.proj', 'ffn.net.2']


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------- prompts
class PromptBank:
    """umT5 prompt embeddings, cached on disk by text (so the 11 GB encoder loads at most once)."""

    def __init__(self, base: str | Path | None, cache: str | Path | None, device='cpu', dtype=torch.bfloat16):
        self.base, self.device, self.dtype = base, device, dtype
        self.cache = Path(cache) if cache else None
        self._mem: dict[str, torch.Tensor] = {}
        self._encoder = None

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.strip().encode()).hexdigest()[:24]

    def put(self, text: str, embeds: torch.Tensor):
        self._mem[self.key(text)] = embeds

    def _load_encoder(self):
        from transformers import AutoTokenizer, UMT5EncoderModel
        tok = AutoTokenizer.from_pretrained(self.base, subfolder='tokenizer')
        enc = UMT5EncoderModel.from_pretrained(self.base, subfolder='text_encoder', torch_dtype=self.dtype)
        self._encoder = (tok, enc.to(self.device).eval())

    def release(self):
        self._encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @torch.no_grad()
    def get(self, text: str) -> torch.Tensor:
        """(1, 512, text_dim) embeddings, zero-padded past the prompt's own length (as the pipeline)."""
        k = self.key(text)
        if k in self._mem:
            return self._mem[k]
        f = self.cache / f'{k}.pt' if self.cache else None
        if f and f.exists():
            e = torch.load(f, map_location='cpu', weights_only=True)
            self._mem[k] = e
            return e
        if self.base is None:
            raise KeyError(f'no embeddings for prompt {text!r} and no text encoder to make them')
        if self._encoder is None:
            self._load_encoder()
        tok, enc = self._encoder
        t = tok([' '.join(text.split())], padding='max_length', max_length=MAX_SEQUENCE_LENGTH, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
        n = int(t.attention_mask.gt(0).sum())
        e = enc(t.input_ids.to(self.device), t.attention_mask.to(self.device)).last_hidden_state.float().cpu()
        e[:, n:] = 0
        if f:
            f.parent.mkdir(parents=True, exist_ok=True)
            torch.save(e, f)
            (f.with_suffix('.json')).write_text(json.dumps({'text': text, 'tokens': n}))
        self._mem[k] = e
        return e


# ------------------------------------------------------------------------------------ model
@dataclass
class FillSettings:
    steps: int = 30
    seed: int = 0
    conditioning_scale: float = 1.0
    guidance: float = 1.0                    # > 1: classifier-free guidance with `negative` (2x compute)
    negative: str = ''
    prompt: str = BASE_PROMPT
    prompt_highlight: str = ''               # '' = the base prompt there too
    prompt_shadow: str = ''
    extra: dict = field(default_factory=dict)


class WanFill:
    def __init__(self, vae, transformer, scheduler, prompts: PromptBank, device='cpu', dtype=torch.bfloat16):
        self.vae, self.transformer, self.scheduler, self.prompts = vae, transformer, scheduler, prompts
        self.device, self.dtype = torch.device(device), dtype
        self.t_scale = 2 ** sum(vae.config.temperal_downsample)          # 4
        self.s_scale = 2 ** len(vae.config.temperal_downsample)          # 8
        self.patch = transformer.config.patch_size[1]                    # 2
        z = vae.config.z_dim
        self._mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, z, 1, 1, 1)
        self._inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, z, 1, 1, 1)
        self.identity: dict = {}

    @property
    def multiple(self) -> int:
        return self.s_scale * self.patch                                  # 16: frame sizes must divide by it

    @classmethod
    def load(cls, base: str | Path, lora: str | Path | None = None, device='cuda', dtype=torch.bfloat16,
             flow_shift: float = 3.0, prompt_cache: str | Path | None = None, text_device: str | None = None):
        from diffusers import AutoencoderKLWan, WanVACETransformer3DModel
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        vae = AutoencoderKLWan.from_pretrained(base, subfolder='vae', torch_dtype=torch.float32).to(device).eval()
        tr = WanVACETransformer3DModel.from_pretrained(base, subfolder='transformer', torch_dtype=dtype)
        if lora:
            load_lora(tr, lora)
        tr = tr.to(device).eval()
        sch = UniPCMultistepScheduler.from_pretrained(base, subfolder='scheduler')
        sch = UniPCMultistepScheduler.from_config(sch.config, flow_shift=flow_shift)
        pb = PromptBank(base, prompt_cache, device=text_device or device, dtype=dtype)
        self = cls(vae, tr, sch, pb, device=device, dtype=dtype)
        self.identity = {'base': str(base), 'lora': str(lora) if lora else None,
                         'lora_sha256': sha256_file(lora) if lora else None, 'flow_shift': flow_shift,
                         'dtype': str(dtype).replace('torch.', ''), 'vae_dtype': 'float32'}
        return self

    # ------------------------------------------------------------------------------ VAE
    @torch.no_grad()
    def encode_pixels(self, video: torch.Tensor) -> torch.Tensor:
        """(1, 3, F, H, W) in [-1, 1] -> normalised latents (1, 16, 1 + (F-1)/4, H/8, W/8), float32."""
        z = self.vae.encode(video.to(self.device, torch.float32)).latent_dist.mode()
        return (z.float() - self._mean.to(z.device)) * self._inv_std.to(z.device)

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.device, torch.float32) / self._inv_std.to(self.device) + self._mean.to(self.device)
        return self.vae.decode(z, return_dict=False)[0].float().clamp(-1, 1)

    def mask_latent(self, mask: torch.Tensor) -> torch.Tensor:
        """(1, 1, F, H, W) in {0, 1} -> (1, 64, f, H/8, W/8): the pipeline's prepare_masks."""
        _, _, n, h, w = mask.shape
        s = self.s_scale
        nf = (n + self.t_scale - 1) // self.t_scale
        m = mask[0, 0].float().view(n, h // s, s, w // s, s).permute(2, 4, 0, 1, 3).flatten(0, 1)
        m = F.interpolate(m.unsqueeze(0), size=(nf, h // s, w // s), mode='nearest-exact')
        return m

    @torch.no_grad()
    def context(self, video: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """video (1, 3, F, H, W) in [-1, 1], mask (1, 1, F, H, W) in {0, 1} -> (1, 96, f, H/8, W/8)."""
        _, _, n, h, w = video.shape
        if (n - 1) % self.t_scale or h % self.multiple or w % self.multiple:
            raise ValueError(f'frames must be 4k+1 and sizes multiples of {self.multiple} (got {n}x{h}x{w})')
        m = (mask > 0.5).float().to(self.device)
        v = video.to(self.device, torch.float32)
        inactive = self.encode_pixels(v * (1 - m))
        reactive = self.encode_pixels(v * m)
        return torch.cat([inactive, reactive, self.mask_latent(m).to(inactive)], dim=1)

    # ------------------------------------------------------------------------- denoising
    def _region_weights(self, masks: dict, shape) -> dict:
        """Pixel masks (1, 1, F, H, W) -> weights at the latent grid (1, 1, f, h, w), max-pooled."""
        out = {}
        f, h, w = shape[2:]
        for k, m in masks.items():
            if m is None:
                continue
            m = m.float()
            n = m.shape[2]
            # frame 0 alone, then groups of 4 (the VAE's causal temporal grouping)
            groups = [m[:, :, :1]] + [m[:, :, 1 + 4 * i:1 + 4 * (i + 1)] for i in range((n - 1) // 4)]
            g = torch.cat([x.amax(2, keepdim=True) for x in groups], 2)
            out[k] = F.adaptive_max_pool3d(g, (f, h, w)).to(self.device)
        return out

    def _predict(self, x, t, embeds, ctx, scale):
        return self.transformer(hidden_states=x.to(self.dtype), timestep=t.expand(x.shape[0]),
                                encoder_hidden_states=embeds.to(self.device, self.dtype),
                                control_hidden_states=ctx.to(self.dtype), control_hidden_states_scale=scale,
                                return_dict=False)[0].float()

    @torch.no_grad()
    def denoise(self, ctx: torch.Tensor, s: FillSettings, region_masks: dict | None = None,
                progress=None) -> torch.Tensor:
        """ctx (1, 96, f, h, w) -> normalised latents (1, 16, f, h, w)."""
        n_layers = len(self.transformer.config.vace_layers)
        scale = torch.full((n_layers,), float(s.conditioning_scale), device=self.device, dtype=self.dtype)
        g = torch.Generator('cpu').manual_seed(int(s.seed))
        shape = (1, self.transformer.config.in_channels) + tuple(ctx.shape[2:])
        x = torch.randn(shape, generator=g, dtype=torch.float32).to(self.device)
        base = self.prompts.get(s.prompt)
        regions = []
        weights = self._region_weights(region_masks or {}, shape)
        for key, text in (('highlight', s.prompt_highlight), ('shadow', s.prompt_shadow)):
            if text.strip() and key in weights and weights[key].any():
                regions.append((weights[key], self.prompts.get(text)))
        neg = self.prompts.get(s.negative) if s.guidance > 1 else None
        self.scheduler.set_timesteps(s.steps, device=self.device)
        for i, t in enumerate(self.scheduler.timesteps):
            eps = self._predict(x, t, base, ctx, scale)
            for w, e in regions:
                eps = eps * (1 - w) + self._predict(x, t, e, ctx, scale) * w
            if neg is not None:
                un = self._predict(x, t, neg, ctx, scale)
                eps = un + s.guidance * (eps - un)
            x = self.scheduler.step(eps, t, x, return_dict=False)[0]
            if progress:
                progress(i + 1, len(self.scheduler.timesteps))
        return x


# -------------------------------------------------------------------------------------- LoRA
def add_lora(transformer, rank: int = 32, alpha: float | None = None, targets=LORA_TARGETS):
    from peft import LoraConfig
    cfg = LoraConfig(r=rank, lora_alpha=alpha or rank, target_modules=targets, init_lora_weights=True)
    transformer.add_adapter(cfg, adapter_name='rudra_genfill')
    for n, p in transformer.named_parameters():
        p.requires_grad_('lora_' in n)
    return [p for p in transformer.parameters() if p.requires_grad]


def lora_state(transformer) -> dict:
    from peft.utils import get_peft_model_state_dict
    return {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(transformer, adapter_name='rudra_genfill').items()}


def save_lora(transformer, path: str | Path, meta: dict):
    from safetensors.torch import save_file
    cfg = transformer.peft_config['rudra_genfill']
    meta = {**{k: str(v) for k, v in meta.items()}, 'rank': str(cfg.r), 'alpha': str(cfg.lora_alpha),
            'targets': json.dumps(sorted(cfg.target_modules)), 'format': 'rudra_genfill_lora_v1'}
    save_file(lora_state(transformer), str(path), metadata=meta)


def load_lora(transformer, path: str | Path):
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), 'pt') as f:
        meta = f.metadata() or {}
    if meta.get('format') != 'rudra_genfill_lora_v1':
        raise ValueError(f'{path}: not a RUDRA genfill LoRA')
    cfg = LoraConfig(r=int(meta['rank']), lora_alpha=float(meta['alpha']),
                     target_modules=json.loads(meta['targets']), init_lora_weights=True)
    transformer.add_adapter(cfg, adapter_name='rudra_genfill')
    res = set_peft_model_state_dict(transformer, load_file(str(path)), adapter_name='rudra_genfill')
    missing = [k for k in (res.missing_keys if res else []) if 'lora_' in k]
    if missing:
        raise ValueError(f'{path}: LoRA weights missing for {len(missing)} modules (first {missing[0]})')
    for p in transformer.parameters():
        p.requires_grad_(False)
    return meta
