"""Train RUDRA's generative-fill LoRA on Wan2.1-VACE (layer 3), 27 Oct 2026.

    python training/genfill_train.py --clips <build_genfill_clips output> --base <Wan2.1-VACE-1.3B-diffusers>
                                     --out <new dir> [--steps 8000] [--max-hours 9.5] [--rank 32] [--lr 1e-4]

Only the LoRA (rank 32 on every attention and feed-forward projection of the transformer and
its VACE control blocks) trains; the transformer stays bf16 and frozen, the VAE and the text
encoder are not loaded (latents and the prompt embedding come precomputed from the builder).

Objective: rectified flow matching, as Wan was trained. For a clip's latents z0 and noise e,
sigma from a logit-normal shifted by `--shift` (3, Wan's 480p value), x = (1 - sigma) z0 + sigma e,
target v = e - z0; loss = mean(w * (v_pred - v)^2) with w = 1 + `--region-weight` inside the fill
mask (at the latent grid), because the region is all the fill is used for.

Every --save-every updates: the LoRA (lora_<step>.safetensors) and the dev loss (fixed noise
and sigmas on the held-out dev clips, a monitor only: the last LoRA is the result, nothing is
selected on it). Stops at --steps or at the first save after --max-hours.
"""
import argparse, hashlib, json, math, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class ClipStore:
    """The builder's shards, memory-mapped; one split ('train' or 'dev')."""

    def __init__(self, root: Path, split: str):
        self.root = Path(root)
        meta = [json.loads(l) for l in (self.root / 'clips.jsonl').read_text().splitlines() if l.strip()]
        self.meta = [m for m in meta if m['split'] == split]
        self._cache = {}
        s = json.loads((self.root / 'summary.json').read_text())
        self.frames, (self.h, self.w) = s['frames'], s['size']

    def __len__(self):
        return len(self.meta)

    def _arr(self, shard, kind):
        k = (shard, kind)
        if k not in self._cache:
            self._cache[k] = np.load(self.root / f'{shard}_{kind}.npy', mmap_mode='r')
        return self._cache[k]

    def get(self, i):
        m = self.meta[i]
        z0 = torch.from_numpy(np.array(self._arr(m['shard'], 'z0')[m['index']], np.float32))
        ctx = torch.from_numpy(np.array(self._arr(m['shard'], 'ctx')[m['index']], np.float32))
        bits = np.array(self._arr(m['shard'], 'mask')[m['index']])
        mask = np.unpackbits(bits)[:self.frames * self.h * self.w].reshape(self.frames, self.h, self.w)
        return z0, ctx, torch.from_numpy(mask.astype(np.float32))

    def has_core(self) -> bool:
        return bool(self.meta) and (self.root / f"{self.meta[0]['shard']}_core.npy").exists()

    def get_core(self, i):
        """v2 stores: (core (F, H, W) {0,1}: max code 255 or 0; ylog0 (H, W): frame 0's true log2 luminance)."""
        m = self.meta[i]
        bits = np.array(self._arr(m['shard'], 'core')[m['index']])
        core = np.unpackbits(bits)[:self.frames * self.h * self.w].reshape(self.frames, self.h, self.w)
        ylog = np.array(self._arr(m['shard'], 'ylog')[m['index']], np.float32)
        return torch.from_numpy(core.astype(np.float32)), torch.from_numpy(ylog)


def mask_latent64(mask, s_scale=8, t_scale=4):
    """(F, H, W) {0,1} -> (64, f, H/8, W/8): the VACE pipeline's mask folding."""
    n, h, w = mask.shape
    nf = (n + t_scale - 1) // t_scale
    m = mask.view(n, h // s_scale, s_scale, w // s_scale, s_scale).permute(2, 4, 0, 1, 3).flatten(0, 1)
    return F.interpolate(m.unsqueeze(0), size=(nf, h // s_scale, w // s_scale), mode='nearest-exact')[0]


def region_weight_latent(mask, shape):
    """(F, H, W) -> (1, f, h, w) max over each latent cell (frame 0 alone, then groups of 4)."""
    n = mask.shape[0]
    groups = [mask[:1]] + [mask[1 + 4 * i:1 + 4 * (i + 1)] for i in range((n - 1) // 4)]
    g = torch.stack([x.amax(0) for x in groups])[None, None]
    return F.adaptive_max_pool3d(g, shape)[0]


def sample_sigma(gen, shift, n=1):
    u = torch.sigmoid(torch.randn(n, generator=gen))
    return shift * u / (1 + (shift - 1) * u)


def flow_loss(transformer, z0, ctx, mask, embeds, sigma, noise, region_weight, scale, dtype,
              weight_mask=None, return_x0=False):
    """One clip: the weighted flow-matching loss (scalar). weight_mask (F, H, W) picks where the loss is
    weighted up (v2: the code-255/0 core; v1: the model's context mask). return_x0: also the predicted clean
    latents x0 = x_t - sigma * v (for the calibration term)."""
    ctx96 = torch.cat([ctx, mask_latent64(mask).to(ctx)], 0)[None]
    x = (1 - sigma) * z0[None] + sigma * noise
    v = noise - z0[None]
    pred = transformer(hidden_states=x.to(dtype), timestep=(sigma * 1000).view(1).to(x.device),
                       encoder_hidden_states=embeds.to(dtype), control_hidden_states=ctx96.to(dtype),
                       control_hidden_states_scale=scale, return_dict=False)[0].float()
    wm = mask if weight_mask is None else weight_mask
    w = 1 + region_weight * region_weight_latent(wm, tuple(z0.shape[1:])).to(pred)
    loss = (w * (pred - v) ** 2).mean()
    if return_x0:
        return loss, x - sigma * pred
    return loss


AP1_Y = torch.tensor([.2722287168, .6740817658, .0536895174])


def calibration_loss(vae, x0, core0, ylog0, over_weight, latents_mean, latents_inv_std):
    """v2: how bright the predicted fill is, against the truth, on frame 0's fully clipped / crushed
    pixels. The predicted clean latents of the first latent frame (frame 0 alone, the VAE's causal
    start) are decoded by the frozen float32 VAE, our encoding is undone, and the log2 luminance is
    compared: overshoot (more light than there was: the v1.1 failure) costs `over_weight` times more
    than undershoot."""
    from rudra.genfill import encoding
    if core0.sum() < 1:
        return x0.new_zeros(())
    z = x0[:, :, :1].float() / latents_inv_std + latents_mean
    img = vae.decode(z, return_dict=False)[0][:, :, 0].float().clamp(-1, 1)        # (1, 3, H, W)
    lin = encoding.decode(img)[0]
    y = (lin * AP1_Y.to(lin).view(3, 1, 1)).sum(0)
    d = (torch.log2(y.clamp_min(1e-6)) - ylog0)[core0 > 0]
    return (over_weight * torch.relu(d) ** 2 + torch.relu(-d) ** 2).mean()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clips', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--max-hours', type=float, default=0)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--accumulate', type=int, default=4)
    ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--shift', type=float, default=3.0)
    ap.add_argument('--region-weight', type=float, default=4.0)
    ap.add_argument('--dev-clips', type=int, default=32)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--registration', default='')
    ap.add_argument('--init-lora', default='', help='start from this RUDRA genfill LoRA (v2: the v1 LoRA)')
    ap.add_argument('--weight-on', choices=('auto', 'core', 'mask'), default='auto',
                    help='where the flow loss is weighted up: core (v2, code 255/0) or the context mask (v1)')
    ap.add_argument('--calib-weight', type=float, default=0.0, help='v2: weight of the brightness calibration term (0 = off)')
    ap.add_argument('--calib-sigma', type=float, default=0.6, help='calibrate only at noise levels <= this')
    ap.add_argument('--over-weight', type=float, default=4.0, help='overshoot costs this many times undershoot')
    ap.add_argument('--no-checkpointing', action='store_true')
    ap.add_argument('--tiny', action='store_true', help=argparse.SUPPRESS)   # tests: a random tiny Wan
    a = ap.parse_args(argv)
    from rudra.genfill.model import BASE_PROMPT, PromptBank, add_lora, save_lora
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    clips = Path(a.clips)
    summary = json.loads((clips / 'summary.json').read_text())
    if not summary.get('commercial_clean') or not summary.get('encoded'):
        sys.exit('the clip store must be encoded and commercial-clean')
    train, dev = ClipStore(clips, 'train'), ClipStore(clips, 'dev')
    if not len(train):
        sys.exit('no training clips')
    if a.tiny:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
        from genfill_tiny import tiny_components
        _, tr, _ = tiny_components(a.seed)
        embeds = torch.randn(1, 512, tr.config.text_dim, generator=torch.Generator().manual_seed(0))
    else:
        from diffusers import WanVACETransformer3DModel
        tr = WanVACETransformer3DModel.from_pretrained(a.base, subfolder='transformer', torch_dtype=dtype)
        embeds = PromptBank(None, clips / 'prompt_cache').get(BASE_PROMPT)
    # from_pretrained(torch_dtype=bf16) keeps Wan's float32 modules (norms, time embedder); a blanket
    # .to(dtype) would cast them too (the v1 run's warning)
    tr = tr.to(device, dtype) if a.tiny else tr.to(device)
    use_core = train.has_core() if a.weight_on == 'auto' else a.weight_on == 'core'
    if use_core and not train.has_core():
        sys.exit('--weight-on core needs a v2 clip store (core masks)')
    vae = lat_mean = lat_inv_std = None
    if a.calib_weight > 0:
        if not train.has_core():
            sys.exit('--calib-weight needs a v2 clip store (core masks and frame-0 luminance)')
        if a.tiny:
            vae = tiny_components(0)[0]
        else:
            from diffusers import AutoencoderKLWan
            vae = AutoencoderKLWan.from_pretrained(a.base, subfolder='vae', torch_dtype=torch.float32)
        vae = vae.to(device).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        zc = vae.config.z_dim
        lat_mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, zc, 1, 1, 1).to(device)
        lat_inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32)).view(1, zc, 1, 1, 1).to(device)
    calib_oom = 0
    if a.init_lora:
        from rudra.genfill.model import load_lora
        init_meta = load_lora(tr, a.init_lora)
        params = [p for n, p in tr.named_parameters() if 'lora_' in n]
        for p in params:
            p.requires_grad_(True)
    else:
        params = add_lora(tr, rank=a.rank)
    for p in params:
        p.data = p.data.float()                   # LoRA weights train in float32
    if not a.no_checkpointing and hasattr(tr, 'enable_gradient_checkpointing'):
        tr.enable_gradient_checkpointing()
    tr.train()
    n_lora = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup)))
    scale = torch.ones(len(tr.config.vace_layers), device=device, dtype=dtype)
    embeds = embeds.to(device)
    gen = torch.Generator().manual_seed(a.seed)
    # the dev monitor: fixed clips, sigmas and noise
    dgen = torch.Generator().manual_seed(a.seed + 1)
    dev_idx = list(range(min(a.dev_clips, len(dev))))
    dev_fixed = [(sample_sigma(dgen, a.shift)[0].item(), torch.randn((16,) + tuple(dev.get(i)[0].shape[1:]), generator=dgen))
                 for i in dev_idx]

    def dev_loss():
        if not dev_idx:
            return None
        tr.eval()
        tot = 0.0
        with torch.no_grad(), torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu', dtype=dtype, enabled=device == 'cuda'):
            for i, (sg, nz) in zip(dev_idx, dev_fixed):
                z0, ctx, m = dev.get(i)
                wm = dev.get_core(i)[0].to(device) if use_core else None
                tot += float(flow_loss(tr, z0.to(device), ctx.to(device), m.to(device), embeds, torch.tensor(sg, device=device),
                                       nz[None].to(device), a.region_weight, scale, dtype, weight_mask=wm))
        tr.train()
        return tot / len(dev_idx)

    log = (out / 'log.jsonl').open('w')
    t0 = time.time(); step = 0; order = []
    report = {'clips': str(clips), 'clips_summary_sha256': sha(clips / 'summary.json'), 'base': a.base,
              'train_clips': len(train), 'dev_clips': len(dev), 'lora_params': n_lora, 'args': vars(a),
              'registration_sha256': sha(a.registration) if a.registration else None, 'device': device,
              'dtype': str(dtype), 'release_authorized': False, 'saves': [], 'weight_on': 'core' if use_core else 'mask',
              'init_lora': a.init_lora or None, 'init_lora_sha256': sha(a.init_lora) if a.init_lora else None}
    d0 = dev_loss()
    log.write(json.dumps({'step': 0, 'dev_loss': d0}) + '\n'); log.flush()
    print(f'LoRA {n_lora / 1e6:.2f} M params; {len(train)} train / {len(dev)} dev clips; dev loss {d0}', flush=True)
    stop = False
    while not stop:
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        cal_acc = []
        for _ in range(a.accumulate):
            if not order:
                order = torch.randperm(len(train), generator=gen).tolist()
            idx = order.pop()
            z0, ctx, m = train.get(idx)
            core = ylog = None
            if use_core or vae is not None:
                core, ylog = train.get_core(idx)
            sigma = sample_sigma(gen, a.shift)[0].to(device)
            noise = torch.randn((1,) + tuple(z0.shape), generator=gen).to(device)
            calib_now = vae is not None and float(sigma) <= a.calib_sigma and calib_oom < 3
            with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu', dtype=dtype, enabled=device == 'cuda'):
                res = flow_loss(tr, z0.to(device), ctx.to(device), m.to(device), embeds, sigma, noise,
                                a.region_weight, scale, dtype, weight_mask=core.to(device) if use_core else None,
                                return_x0=calib_now)
            loss, x0 = res if calib_now else (res, None)
            cal = None
            if calib_now:
                try:
                    cal = calibration_loss(vae, x0, core[0].to(device), ylog.to(device), a.over_weight, lat_mean, lat_inv_std)
                    loss = loss + a.calib_weight * cal
                except torch.cuda.OutOfMemoryError:
                    calib_oom += 1
                    torch.cuda.empty_cache()
                    print(f'calibration skipped (GPU memory), {calib_oom}/3', flush=True)
            (loss / a.accumulate).backward()
            acc += float(loss.detach()) / a.accumulate
            if cal is not None:
                cal_acc.append(float(cal.detach()))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); step += 1
        if step % 20 == 0 or step == 1:
            cm = float(np.mean(cal_acc)) if cal_acc else None
            log.write(json.dumps({'step': step, 'loss': acc, 'calib': cm, 'lr': sched.get_last_lr()[0],
                                  'hours': (time.time() - t0) / 3600}) + '\n'); log.flush()
            print(f'step {step} loss {acc:.4f}' + (f' calib {cm:.4f}' if cm is not None else '') + f' {time.time() - t0:.0f} s', flush=True)
        over = a.max_hours and (time.time() - t0) / 3600 >= a.max_hours
        if step % a.save_every == 0 or step >= a.steps or (over and step % a.save_every == 0):
            path = out / f'lora_{step:06d}.safetensors'
            save_lora(tr, path, {'step': step, 'base': a.base, 'clips_summary_sha256': report['clips_summary_sha256'],
                                 'seed': a.seed})
            dl = dev_loss()
            report['saves'].append({'step': step, 'file': path.name, 'sha256': sha(path), 'dev_loss': dl,
                                    'hours': round((time.time() - t0) / 3600, 3)})
            log.write(json.dumps({'step': step, 'dev_loss': dl, 'saved': path.name}) + '\n'); log.flush()
            print(f'saved {path.name}; dev loss {dl}', flush=True)
            if over:
                stop = True
        if step >= a.steps:
            stop = True
    last = out / report['saves'][-1]['file'] if report['saves'] else None
    if last is None:
        last = out / f'lora_{step:06d}.safetensors'
        save_lora(tr, last, {'step': step, 'base': a.base, 'seed': a.seed})
    final = out / 'rudra_genfill.safetensors'
    shutil.copyfile(last, final)
    if (clips / 'prompt_cache').exists():
        shutil.copytree(clips / 'prompt_cache', out / 'prompt_cache')
    report.update({'steps_done': step, 'hours': round((time.time() - t0) / 3600, 3), 'result': final.name,
                   'result_sha256': sha(final), 'dev_loss_start': d0, 'calib_oom_skips': calib_oom,
                   'calibration_disabled_by_oom': calib_oom >= 3})
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('steps_done', 'hours', 'result', 'result_sha256')}), flush=True)


if __name__ == '__main__':
    main()
