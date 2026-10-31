"""Generator-VAE round trip for run05 inputs (22 Oct 2026).

Every latent video generator decodes its frames through a VAE. Passing our SDR
through the same kind of VAE reproduces the generator's texture smoothing,
colour drift and fine-detail loss while we still know the true HDR.

Default: the Wan 2.1 / 2.2 video VAE (Apache-2.0, Alibaba). It is NOT downloaded
here. Point RUDRA_GEN_VAE at a local copy, e.g. ComfyUI's
models/vae/wan_2.1_vae.safetensors (single file) or a diffusers folder that has
a `vae` subfolder. Needs `pip install diffusers` in the venv; CUDA is used when
available. No LTX VAE is supported here, by owner decision.

    python pipeline/run05_corpus.py build ... --vae pipeline.generator_vae:wan_vae --vae-p 0.3
"""
from __future__ import annotations

import os
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np

LICENCES = {"wan": "Apache-2.0 (Wan-AI/Wan2.1, Wan2.2 VAE)"}


def wan_vae_identity(path: str | None = None):
    """Pin local weight/config contents without loading weights or downloading."""
    value = path or os.environ.get('RUDRA_GEN_VAE')
    if not value or not Path(value).exists():
        raise FileNotFoundError('set RUDRA_GEN_VAE to verified local Wan VAE weights')
    root = Path(value).resolve()
    if 'ltx' in root.name.lower():
        raise ValueError('LTX weights are not used in RUDRA')
    files = [root] if root.is_file() else sorted(
        p for p in root.rglob('*') if p.is_file() and '.cache' not in p.parts
        and p.suffix in ('.safetensors', '.json'))
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError('Wan VAE identity requires local safetensors weights')
    hashes = {}
    for p in files + [Path(__file__).resolve()]:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        hashes[str(p)] = h.hexdigest()
    return dict(path=str(root), sha256=hashes, diffusers=importlib.metadata.version('diffusers'))


def wan_vae(path: str | None = None, device: str | None = None, multiple: int = 16):
    import torch
    from diffusers import AutoencoderKLWan
    path = path or os.environ.get("RUDRA_GEN_VAE")
    if not path or not Path(path).exists():
        raise FileNotFoundError("set RUDRA_GEN_VAE to a local Wan VAE (.safetensors file or diffusers folder)")
    if "ltx" in Path(path).name.lower():
        raise ValueError("LTX weights are not used in RUDRA (owner decision, 22 Oct 2026)")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    p = Path(path)
    if p.is_file():
        vae = AutoencoderKLWan.from_single_file(str(p), torch_dtype=torch.float32)
    else:
        sub = "vae" if (p / "vae").is_dir() else None
        vae = AutoencoderKLWan.from_pretrained(str(p), subfolder=sub, torch_dtype=torch.float32)
    vae = vae.to(dev).eval()
    return make_roundtrip(vae, dev, multiple)


def make_roundtrip(vae, device, multiple: int = 16):
    """Wrap any diffusers-style video VAE (encode(x).latent_dist.mode(), decode(z).sample)."""
    import torch

    @torch.inference_mode()
    def roundtrip(rgb8: np.ndarray) -> np.ndarray:
        h, w = rgb8.shape[:2]
        ph, pw = (-h) % multiple, (-w) % multiple
        img = np.pad(rgb8, ((0, ph), (0, pw), (0, 0)), mode="reflect") if (ph or pw) else rgb8
        x = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None, :, None].to(device)
        z = vae.encode(x).latent_dist.mode()
        y = vae.decode(z).sample[0, :, 0].clamp(-1, 1)
        out = ((y.permute(1, 2, 0).float().cpu().numpy() + 1.0) * 127.5).round().astype(np.uint8)
        return out[:h, :w]

    return roundtrip
