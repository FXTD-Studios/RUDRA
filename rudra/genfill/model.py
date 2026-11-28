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

Memory (16 Nov 2026, RUDRA 1.2.5): set_memory_mode() lets the fill run on GPUs under 16 GB.
  'swap'  the transformer goes to the CPU while the VAE encodes and decodes, and back for the
          denoising: the two never hold GPU memory at once. Bit-identical to 'off'; it costs two
          copies of the transformer (about 3 GB) per window.
  'tile'  the VAE encodes and decodes in overlapping 256-pixel tiles (diffusers' own tiling). Its
          peak falls a lot; the result can differ slightly at tile seams, so it is measured
          (tools/bench_genfill_memory.py) before any card size is offered it.
  'on'    both. 'off' is the fill as it was gated.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

PROMPT_PASSES = ('blend', 'single')
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
# 1.3.2 (26 Nov 2026): a new prompt is read faster. The encoder reads the prompt's own tokens, not all 512 padded
# ones: the padded positions are masked out of every attention, so the prompt's tokens come out as before (measured on
# the CPU: identical, cosine 1.0; diffusers' Wan pipeline keeps only those and zero-pads, as get() does), at a
# fraction of the work: a 21-token prompt took 1.2 s on the CPU instead of 516.6 s. With device 'auto' the encoder
# runs on the GPU when the GPU has its size free (make_room, set by WanFill, may first move the fill's transformer to
# the CPU), else on the CPU; measured slower than the CPU now (moving 11 GB costs more than the reading), so RUDRA's
# fills and the plug-in read on the CPU.
TEXT_ENCODER_MARGIN = 768 * 2 ** 20      # beyond the weights: activations and the CUDA allocator's slack


def text_encoder_parameters(base: str | Path | None) -> int:
    """The text encoder's parameter count, from its safetensors headers (nothing loaded); 0 when unknown."""
    if base is None:
        return 0
    n = 0
    for f in sorted((Path(base) / 'text_encoder').glob('*.safetensors')):
        try:
            with open(f, 'rb') as fh:
                size = int.from_bytes(fh.read(8), 'little')
                head = json.loads(fh.read(size))
        except (OSError, ValueError):
            return 0
        for k, v in head.items():
            if k != '__metadata__':
                c = 1
                for d in v.get('shape', []):
                    c *= int(d)
                n += c
    return n


def gpu_free_bytes() -> int:
    """GPU memory free for this process now (CUDA; 0 elsewhere), within RUDRA_GENFILL_MEMORY_CAP_GB when set."""
    if not torch.cuda.is_available():
        return 0
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - a driver that cannot say: as if nothing were free
        return 0
    cap = os.environ.get('RUDRA_GENFILL_MEMORY_CAP_GB', '').strip()
    if cap:
        try:
            free = min(free, int(float(cap) * 2 ** 30) - torch.cuda.memory_reserved())
        except ValueError:
            pass
    return max(0, int(free))


def encode_tokens(encoder, input_ids: torch.Tensor, attention_mask: torch.Tensor, device) -> torch.Tensor:
    """(1, L, dim) float32 on the CPU, L = input_ids' length: the encoder on the prompt's own tokens (the mask's
    leading ones, as the tokenizer pads on the right), zeros after them. A mask that is not leading ones (a
    tokenizer padding on the left) is encoded whole, masked, as before 1.3.2."""
    n = int(attention_mask.gt(0).sum())
    if n > 0 and bool(attention_mask[0, :n].gt(0).all()):
        h = encoder(input_ids[:, :n].to(device), attention_mask[:, :n].to(device)).last_hidden_state.float().cpu()
        e = torch.zeros((1, input_ids.shape[1], h.shape[-1]), dtype=torch.float32)
        e[:, :n] = h
        return e
    e = encoder(input_ids.to(device), attention_mask.to(device)).last_hidden_state.float().cpu()
    e[:, n:] = 0
    return e


class PromptBank:
    """umT5 prompt embeddings, cached on disk by text (so the 11 GB encoder loads at most once)."""

    def __init__(self, base: str | Path | None, cache: str | Path | None, device='cpu', dtype=torch.bfloat16):
        """device: where the encoder runs; 'auto' (1.3.2): the GPU when it has the room, else the CPU."""
        self.base, self.device, self.dtype = base, device, dtype
        self.cache = Path(cache) if cache else None
        self._mem: dict[str, torch.Tensor] = {}
        self._encoder = None
        self.make_room = None            # WanFill.load: frees the GPU for the encoder (its transformer to the CPU)
        self.encoder_device = None       # where the encoder ran last, for the log
        self.seconds = 0.0               # time spent loading the encoder and embedding, for the log

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.strip().encode()).hexdigest()[:24]

    def has(self, text: str) -> bool:
        """Whether get(text) is instant: in memory or in the disk cache (no umT5 load)."""
        k = self.key(text)
        return k in self._mem or bool(self.cache and (self.cache / f'{k}.pt').exists())

    def put(self, text: str, embeds: torch.Tensor):
        self._mem[self.key(text)] = embeds

    def _encoder_room(self) -> str:
        """'cuda' when the GPU has the encoder's weights and TEXT_ENCODER_MARGIN free (after make_room if
        need be), else 'cpu'. Only for device 'auto'."""
        if not torch.cuda.is_available():
            return 'cpu'
        params = text_encoder_parameters(self.base)
        if params <= 0:
            return 'cpu'
        need = params * torch.tensor([], dtype=self.dtype).element_size() + TEXT_ENCODER_MARGIN
        if gpu_free_bytes() >= need:
            return 'cuda'
        if self.make_room is not None:
            self.make_room()
            torch.cuda.empty_cache()
            if gpu_free_bytes() >= need:
                return 'cuda'
        return 'cpu'

    def _load_encoder(self):
        from transformers import AutoTokenizer, UMT5EncoderModel
        tok = AutoTokenizer.from_pretrained(self.base, subfolder='tokenizer')
        enc = UMT5EncoderModel.from_pretrained(self.base, subfolder='text_encoder', torch_dtype=self.dtype).eval()
        device = self._encoder_room() if self.device == 'auto' else self.device
        if device != 'cpu':
            oom = (getattr(torch.cuda, 'OutOfMemoryError', RuntimeError), RuntimeError)
            try:
                enc = enc.to(device)
            except oom:                       # the room went to someone else meanwhile: the CPU, as before 1.3.2
                enc = enc.to('cpu')
                device = 'cpu'
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self._encoder = (tok, enc, device)
        self.encoder_device = device

    def release(self):
        self._encoder = None
        gc.collect()                          # the 11 GB go back now, not at the next collection
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
        t0 = time.time()
        if self._encoder is None:
            self._load_encoder()
        tok, enc, device = self._encoder
        t = tok([' '.join(text.split())], padding='max_length', max_length=MAX_SEQUENCE_LENGTH, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
        n = int(t.attention_mask.gt(0).sum())
        e = encode_tokens(enc, t.input_ids, t.attention_mask, device)
        self.seconds += time.time() - t0
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
    # RUDRA 1.1 (5 Nov 2026): how strongly a region prompt steers its region. 1 = the region's
    # prediction replaces the base one there (as before, bit for bit); 0 = the prompt is ignored;
    # above 1 extrapolates away from the base prompt (stronger adherence, experimental: the v3
    # gates were passed at 1).
    prompt_strength: float = 1.0
    # 1.3.2 (26 Nov 2026): how a single region prompt at strength 1 is applied. 'blend': a prediction with the
    # base prompt everywhere and one with the region prompt, blended by the region's mask (two transformer
    # passes a step, as gated). 'single': the region prompt drives the whole window in one pass: inside the
    # region that is the prediction 'blend' takes there, outside it the window is the master's own context
    # (VACE keeps it), so the fill comes out nearly the same (measured: run.py PROMPT_PASS), faster. Two region
    # prompts, or a strength other than 1, always blend. The default here stays 'blend'; run.py runs 'single'.
    prompt_pass: str = 'blend'
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
        self.memory_mode = 'off'
        self._swap = False
        self._transformer_on_device = True

    MEMORY_MODES = ('off', 'swap', 'tile', 'on')

    def set_memory_mode(self, mode: str) -> str:
        """'off', 'swap', 'tile' or 'on' (the module docstring). Recorded in identity, so the fill's
        record says how it ran. Swapping is only for CUDA: elsewhere the memory is the system's."""
        if mode not in self.MEMORY_MODES:
            raise ValueError(f"memory mode must be one of {', '.join(self.MEMORY_MODES)} (got {mode!r})")
        tile = mode in ('tile', 'on')
        if tile and hasattr(self.vae, 'enable_tiling'):
            self.vae.enable_tiling()
        elif hasattr(self.vae, 'use_tiling'):
            self.vae.use_tiling = False
        self._swap = mode in ('swap', 'on') and self.device.type == 'cuda'
        if not self._swap:
            self._transformer_to_device(True)
        self.memory_mode = mode
        self.identity['memory_mode'] = mode
        return mode

    def _transformer_to_device(self, on_device: bool):
        """With 'swap': the transformer on the GPU only while it denoises."""
        if on_device == self._transformer_on_device:
            return
        if on_device:
            self.transformer.to(self.device)
        else:
            self.transformer.to('cpu')
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        self._transformer_on_device = on_device

    @property
    def multiple(self) -> int:
        return self.s_scale * self.patch                                  # 16: frame sizes must divide by it

    @classmethod
    def load(cls, base: str | Path, lora: str | Path | None = None, device='cuda', dtype=torch.bfloat16,
             flow_shift: float = 3.0, prompt_cache: str | Path | None = None, text_device: str | None = None,
             prompts: PromptBank | None = None):
        """prompts (1.3.2): a PromptBank made before the model, whose prompts were read while the GPU was free;
        else one is made here (text_device, else the model's device)."""
        from diffusers import AutoencoderKLWan, WanVACETransformer3DModel
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        vae = AutoencoderKLWan.from_pretrained(base, subfolder='vae', torch_dtype=torch.float32).to(device).eval()
        tr = WanVACETransformer3DModel.from_pretrained(base, subfolder='transformer', torch_dtype=dtype)
        if lora:
            load_lora(tr, lora)
        tr = tr.to(device).eval()
        sch = UniPCMultistepScheduler.from_pretrained(base, subfolder='scheduler')
        sch = UniPCMultistepScheduler.from_config(sch.config, flow_shift=flow_shift)
        pb = prompts if prompts is not None else PromptBank(base, prompt_cache, device=text_device or device, dtype=dtype)
        self = cls(vae, tr, sch, pb, device=device, dtype=dtype)
        if self.device.type == 'cuda':
            pb.make_room = lambda: self._transformer_to_device(False)   # denoise() brings it back
        self.identity = {'base': str(base), 'lora': str(lora) if lora else None,
                         'lora_sha256': sha256_file(lora) if lora else None, 'flow_shift': flow_shift,
                         'dtype': str(dtype).replace('torch.', ''), 'vae_dtype': 'float32'}
        return self

    # ------------------------------------------------------------------------------ VAE
    @torch.no_grad()
    def encode_pixels(self, video: torch.Tensor) -> torch.Tensor:
        """(1, 3, F, H, W) in [-1, 1] -> normalised latents (1, 16, 1 + (F-1)/4, H/8, W/8), float32."""
        if self._swap:
            self._transformer_to_device(False)
        z = self.vae.encode(video.to(self.device, torch.float32)).latent_dist.mode()
        return (z.float() - self._mean.to(z.device)) * self._inv_std.to(z.device)

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        if self._swap:
            self._transformer_to_device(False)
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
                progress=None, known: torch.Tensor | None = None, pin: float = 0.0) -> torch.Tensor:
        """ctx (1, 96, f, h, w) -> normalised latents (1, 16, f, h, w).

        known (1.2.9, window continuation): normalised latents (1, 16, k, h, w) for the window's first k
        latent frames, the previous window's picture of the frames the two share. For the first `pin` share
        of the steps those frames are not the model's: after every step they are set to `known` noised to
        the step's level (x_t = (1 - sigma) x0 + sigma noise, the flow schedule, the window's own noise),
        so the rest of the window is generated to continue them; after that the model finishes them itself
        (the seam is crossfaded as before). known None or pin 0: exactly the window as before."""
        # The prompts first (1.3.2): one not read yet loads the text encoder, which may move the transformer off
        # the GPU to make room; it comes back just below.
        shape = (1, self.transformer.config.in_channels) + tuple(ctx.shape[2:])
        base = self.prompts.get(s.prompt)
        regions = []
        weights = self._region_weights(region_masks or {}, shape)
        for key, text in (('highlight', s.prompt_highlight), ('shadow', s.prompt_shadow)):
            if text.strip() and key in weights and weights[key].any():
                regions.append((weights[key], self.prompts.get(text)))
        neg = self.prompts.get(s.negative) if s.guidance > 1 else None
        self._transformer_to_device(True)
        n_layers = len(self.transformer.config.vace_layers)
        scale = torch.full((n_layers,), float(s.conditioning_scale), device=self.device, dtype=self.dtype)
        g = torch.Generator('cpu').manual_seed(int(s.seed))
        x = torch.randn(shape, generator=g, dtype=torch.float32).to(self.device)
        if s.prompt_pass not in PROMPT_PASSES:
            raise ValueError(f"prompt_pass must be one of {', '.join(PROMPT_PASSES)} (got {s.prompt_pass!r})")
        drive = base
        if s.prompt_pass == 'single' and len(regions) == 1 and s.prompt_strength == 1.0:
            drive, regions = regions[0][1], []     # one pass a step: the region prompt drives the window
        self.scheduler.set_timesteps(s.steps, device=self.device)
        steps = len(self.scheduler.timesteps)
        held = 0 if known is None or pin <= 0 else max(1, min(steps, int(round(float(pin) * steps))))
        if held:
            k = int(known.shape[2])
            if k > shape[2] or tuple(known.shape[3:]) != tuple(shape[3:]):
                raise ValueError(f'known latents {tuple(known.shape)} do not fit the window {shape}')
            x0 = known.to(self.device, torch.float32)
            noise = x.clone()
            sig = self.scheduler.sigmas.to(self.device, torch.float32)

            def hold(level):
                x[:, :, :k] = (1 - level) * x0 + level * noise[:, :, :k]
            hold(sig[0])
        for i, t in enumerate(self.scheduler.timesteps):
            eps = self._predict(x, t, drive, ctx, scale)
            for w, e in regions:
                if s.prompt_strength == 1.0:
                    eps = eps * (1 - w) + self._predict(x, t, e, ctx, scale) * w
                elif s.prompt_strength != 0.0:
                    eps = eps + (self._predict(x, t, e, ctx, scale) - eps) * (w * float(s.prompt_strength))
            if neg is not None:
                un = self._predict(x, t, neg, ctx, scale)
                eps = un + s.guidance * (eps - un)
            x = self.scheduler.step(eps, t, x, return_dict=False)[0]
            if i < held:
                hold(sig[i + 1])
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
