"""The generative-fill clip builder: exclusions, Poly Haven camera moves and plate windows,
masks and exposure, the render-only command and the VAE encode stage (tiny random VAE)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip('OpenEXR')
pytest.importorskip('PyOpenColorIO')
UP = Path('/mnt/user-data/uploads/datasets')
PH = ['abandoned_bakery', 'abandoned_church']
FXTD = 'fxtd_manchale_untitled_100_f00001232.png'


def _tree(tmp: Path):
    """A fake G:/ with the registry, one eval corpus, the fresh report, two small HDRIs, one FXTD plate."""
    import OpenEXR
    import cv2
    if not all((UP / 'sources/polyhaven' / f'{n}_2k.exr').exists() for n in PH):
        pytest.skip('staged Poly Haven HDRIs not present')
    g = tmp / 'G'
    ph = g / 'datasets/sources/polyhaven'; ph.mkdir(parents=True)
    from tools.build_genfill_clips import read_exr_rgb
    for n in PH:
        img = cv2.resize(read_exr_rgb(UP / 'sources/polyhaven' / f'{n}_2k.exr'), (512, 256), interpolation=cv2.INTER_AREA)
        OpenEXR.File({'compression': OpenEXR.ZIP_COMPRESSION, 'type': OpenEXR.scanlineimage},
                     {c: np.ascontiguousarray(img[..., i]) for i, c in enumerate('RGB')}).write(str(ph / f'{n}_2k.exr'))
    corp = g / 'datasets/corpora'
    reg = corp / 'rudra_run05_sources_20261022'; reg.mkdir(parents=True)
    rows = [
        {'source_id': 'polyhaven_moves', 'scene_id': 'ph:abandoned_bakery', 'split': 'train', 'asset_id': 'a1',
         'hdr_path': 'G:/x.png', 'plate_primaries': 'rec709', 'licence_source': 'Poly Haven', 'licence_licence': 'CC0', 'commercial_ok': True},
        {'source_id': 'polyhaven_moves', 'scene_id': 'ph:abandoned_church', 'split': 'val', 'asset_id': 'a2',
         'hdr_path': 'G:/y.png', 'plate_primaries': 'rec709', 'licence_source': 'Poly Haven', 'licence_licence': 'CC0', 'commercial_ok': True},
        {'source_id': 'fxtd_manchale', 'scene_id': 'fxtd_manchale_untitled_100', 'split': 'train', 'asset_id': 'f1',
         'hdr_path': f'G:\\datasets\\corpora\\rudra_fxtd_camera_20261016\\hdr\\{FXTD}', 'plate_primaries': 'rec2020',
         'licence_source': 'fxtd', 'licence_licence': 'FXTD Studios own footage', 'commercial_ok': True},
        {'source_id': 'fxtd_lab', 'scene_id': 'fxtd_lab_held', 'split': 'train', 'asset_id': 'f2',
         'hdr_path': f'G:\\datasets\\corpora\\rudra_fxtd_camera_20261016\\hdr\\{FXTD}', 'plate_primaries': 'rec2020',
         'licence_source': 'fxtd', 'licence_licence': 'FXTD Studios own footage', 'commercial_ok': True},
    ]
    (reg / 'sources.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    ev = corp / 'aces_holdout_test'; ev.mkdir()
    (ev / 'cases.jsonl').write_text(json.dumps({'scene_id': 'fxtd_lab_held', 'split': 'train'}) + '\n')
    cam = corp / 'rudra_fxtd_camera_20261016'; (cam / 'hdr').mkdir(parents=True)
    src = UP / 'corpora/rudra_fxtd_camera_20261016'
    (cam / 'hdr' / FXTD).write_bytes((src / 'hdr' / FXTD).read_bytes())
    (cam / '_ingest_config.json').write_bytes((src / '_ingest_config.json').read_bytes())
    fresh = g / 'rudra_training/claude_release_candidate_20261026/fresh_fxtd_grade_v2_20261026'; fresh.mkdir(parents=True)
    (fresh / 'report.json').write_text(json.dumps({'scenes': 1, 'rows': [{'scene_id': 'fxtd_elmatareed_x', 'kept': True}]}))
    reg_file = tmp / 'reg.json'; reg_file.write_text('{}')
    return g, reg_file


def _args(g, **kw):
    import argparse
    from tools.build_genfill_clips import POLYHAVEN, SOURCES
    a = dict(root=str(g).replace('\\', '/') + '/', sources=SOURCES, polyhaven=POLYHAVEN, frames=9, height=96, width=160,
             seed=1, clips_per_hdri=2, clips_per_plate=1, clips_per_fxtd=1, recipe='v2', dev_fraction=0.0)
    a.update(kw)
    return argparse.Namespace(**a)


def test_plan_excludes_evaluation_and_validation_scenes(tmp_path):
    from tools.build_genfill_clips import plan
    g, _ = _tree(tmp_path)
    jobs, excl = plan(_args(g))
    scenes = sorted(j['scene_id'] for j in jobs)
    assert scenes == ['fxtd_manchale_untitled_100', 'ph:abandoned_bakery']
    assert 'fxtd_lab_held' in excl['excluded_scenes']
    assert {j['kind'] for j in jobs} == {'equirect', 'plate'}


def test_clips_are_consistent(tmp_path):
    from rudra.aces_view import ACES2View
    from tools.build_genfill_clips import make_clip, plan
    g, _ = _tree(tmp_path)
    jobs, _ = plan(_args(g))
    view = ACES2View()
    n = 0
    for job in jobs:
        for c in make_clip(job):
            n += 1
            assert c['sdr'].shape == (9, 96, 160, 3) and c['sdr'].dtype == np.uint8
            assert c['truth'].shape == c['cond'].shape == (9, 96, 160, 3)
            assert c['mask'].shape == (9, 96, 160) and 0.002 <= c['mask'].mean() <= 0.6
            assert np.isfinite(c['truth'].astype(np.float32)).all() and c['truth'].min() >= 0
            if not c['meta']['noise']:          # the codes are the view of the truth
                v = np.rint(np.clip(view.forward(c['truth'].astype(np.float32).reshape(-1, 3)), 0, 1) * 255)
                assert np.abs(v - c['sdr'].reshape(-1, 3)).max() <= 1
            # the condition views back to the codes (exact inverse)
            back = np.rint(np.clip(view.forward(c['cond'].astype(np.float32).reshape(-1, 3)), 0, 1) * 255)
            assert np.mean(np.abs(back - c['sdr'].reshape(-1, 3)) <= 1) > 0.999
            assert c['meta']['licence'] in ('CC0', 'FXTD Studios own footage')
    assert n >= 2


def test_render_only_command_and_encode_stage(tmp_path):
    g, reg = _tree(tmp_path)
    out = tmp_path / 'clips'
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, str(ROOT / 'tools/build_genfill_clips.py'), '--out', str(out), '--registration', str(reg),
                        '--root', str(g).replace('\\', '/') + '/', '--frames', '5', '--height', '64', '--width', '96',
                        '--workers', '2', '--render-only', '--clips-per-hdri', '1'],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((out / 'summary.json').read_text())
    assert s['commercial_clean'] and s['clips'] >= 1 and s['release_authorized'] is False
    clip = np.load(next(out.glob('clip_*.npz')))
    from genfill_tiny import tiny_components
    from tools.build_genfill_clips import VaeEncoder
    vae, _, _ = tiny_components()
    enc = VaeEncoder.from_vae(vae)
    z0, ctx = enc.encode_clip({k: clip[k] for k in ('truth', 'cond', 'mask')})
    assert z0.shape == (16, 2, 8, 12) and ctx.shape == (32, 2, 8, 12) and z0.dtype == np.float16
    # same as the inference model's own context (inactive / reactive halves)
    import torch
    from genfill_tiny import tiny_fill
    from rudra.genfill import encoding
    f = tiny_fill()
    f.vae = vae
    v = torch.from_numpy(encoding.encode(clip['cond'].astype(np.float32))).permute(3, 0, 1, 2)[None]
    m = torch.from_numpy(clip['mask'].astype(np.float32))[None, None]
    ref = f.context(v, m)[0, :32].numpy()
    assert np.allclose(ctx.astype(np.float32), ref, atol=2e-2)


def test_encoded_store_trains_a_lora_that_fills(tmp_path):
    """Builder (encode, tiny VAE) -> trainer (tiny Wan, CPU) -> the LoRA loads and fills a master."""
    g, reg = _tree(tmp_path)
    clips = tmp_path / 'enc'
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, str(ROOT / 'tools/build_genfill_clips.py'), '--out', str(clips), '--registration', str(reg),
                        '--root', str(g).replace('\\', '/') + '/', '--frames', '5', '--height', '64', '--width', '96',
                        '--workers', '2', '--tiny', '--clips-per-hdri', '2', '--dev-fraction', '0.5', '--device', 'cpu'],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((clips / 'summary.json').read_text())
    assert s['encoded'] and s['clips'] >= 2
    assert any(clips.glob('*_z0.npy')) and any(clips.glob('preview_*.npz'))
    from training.genfill_train import ClipStore
    for split in ('train', 'dev'):
        st = ClipStore(clips, split)
        for i in range(len(st)):
            z0, ctx, m = st.get(i)
            assert z0.shape == (16, 2, 8, 12) and ctx.shape == (32, 2, 8, 12) and m.shape == (5, 64, 96)
    if not len(ClipStore(clips, 'train')):
        pytest.skip('every scene hashed to dev at this fraction')
    out = tmp_path / 'run'
    r = subprocess.run([sys.executable, str(ROOT / 'training/genfill_train.py'), '--clips', str(clips), '--base', 'tiny',
                        '--out', str(out), '--steps', '30', '--save-every', '10', '--accumulate', '1', '--rank', '4',
                        '--lr', '3e-3', '--warmup', '1', '--device', 'cpu', '--tiny', '--no-checkpointing'],
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads((out / 'report.json').read_text())
    assert rep['steps_done'] == 30 and [x['step'] for x in rep['saves']] == [10, 20, 30]
    assert rep['result'] == 'rudra_genfill.safetensors' and rep['release_authorized'] is False
    # it learns: the training clips' loss at fixed sigmas and noise, LoRA vs none
    import torch
    from genfill_tiny import tiny_components
    from rudra.genfill.model import load_lora as _load
    from training.genfill_train import flow_loss
    st = ClipStore(clips, 'train')
    emb = torch.randn(1, 512, 32, generator=torch.Generator().manual_seed(0))

    def total(tr):
        g = torch.Generator().manual_seed(5); t = 0.0
        with torch.no_grad():
            for i in range(len(st)):
                z0, ctx, m = st.get(i)
                for sg in (0.3, 0.6, 0.9):
                    t += float(flow_loss(tr, z0, ctx, m, emb, torch.tensor(sg), torch.randn((1,) + tuple(z0.shape), generator=g),
                                         4.0, torch.ones(2), torch.float32))
        return t
    plain = tiny_components(20261028)[1]
    tuned = tiny_components(20261028)[1]
    _load(tuned, out / 'rudra_genfill.safetensors')
    assert total(tuned) < 0.9 * total(plain)
    from genfill_tiny import tiny_fill
    from rudra.genfill.model import load_lora
    f = tiny_fill(seed=20261028)
    meta = load_lora(f.transformer, out / 'rudra_genfill.safetensors')
    assert meta['step'] == '30'


def test_evaluator_runs_and_reports_gates(tmp_path):
    g, _ = _tree(tmp_path)
    out = tmp_path / 'ev'
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_genfill.py'), '--out', str(out), '--tiny',
                        '--root', str(g).replace('\\', '/') + '/', '--frames', '9', '--height', '64', '--width', '96',
                        '--window', '5', '--overlap', '2', '--steps', '2', '--ph-offset', '0', '--fxtd-offset', '0', '--fxtd-pool', 'fresh',
                        '--registration', str(ROOT / 'configs/experiments/claude_genfill_v1_1_20261029.json')],
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads((out / 'report.json').read_text())
    assert rep['clips'] == 2                                   # the val scene (church): highlight + shadow
    assert {x['scene_id'] for x in rep['rows']} == {'ph:abandoned_church'}
    assert rep['gates']['R']['pass'] is True                    # codes kept, even by a random model
    assert rep['verdict'] in ('PASS', 'FAIL') and rep['release_authorized'] is False
    assert any('h_log_master' in x for x in rep['rows'])
    assert any('v1region_h_log_master' in x for x in rep['rows'])     # v1's region, reported alongside


def test_builder_resumes_an_interrupted_build_without_repeating_sources(tmp_path):
    g, reg = _tree(tmp_path)
    out = tmp_path / 'enc'
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    base = [sys.executable, str(ROOT / 'tools/build_genfill_clips.py'), '--out', str(out), '--registration', str(reg),
            '--root', str(g).replace('\\', '/') + '/', '--frames', '5', '--height', '64', '--width', '96',
            '--workers', '2', '--in-flight', '1', '--tiny', '--clips-per-hdri', '2', '--device', 'cpu']
    r = subprocess.run(base + ['--limit', '1'], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    first = [json.loads(l) for l in (out / 'clips.jsonl').read_text().splitlines()]
    (out / 'summary.json').unlink()                       # as if it had stopped before the end
    r = subprocess.run(base, capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode != 0                              # never into an existing folder without --resume
    r = subprocess.run(base + ['--resume'], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert '"resume": true' in r.stdout
    meta = [json.loads(l) for l in (out / 'clips.jsonl').read_text().splitlines()]
    assert meta[:len(first)] == first
    assert len({m['asset_id'] for m in meta}) == 2        # both sources, the first not repeated
    assert sum(m['asset_id'] == first[0]['asset_id'] for m in meta) == len([m for m in first if m['asset_id'] == first[0]['asset_id']])
    s = json.loads((out / 'summary.json').read_text())
    assert s['clips'] == len(meta)
    from training.genfill_train import ClipStore
    n = sum(len(ClipStore(out, sp)) for sp in ('train', 'dev'))
    assert n == len(meta)
    for sp in ('train', 'dev'):
        st = ClipStore(out, sp)
        for i in range(len(st)):
            st.get(i)                                     # every row points at a real shard entry
    r = subprocess.run(base + ['--resume'], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode != 0 and 'complete' in (r.stdout + r.stderr)


def test_v2_store_trains_with_core_weighting_calibration_and_v1_init(tmp_path):
    """v2: the builder keeps code-255/0 cores and frame-0 truth luminance; the trainer weights the loss on
    the core, adds the brightness calibration through the frozen VAE, and starts from a v1 LoRA."""
    g, reg = _tree(tmp_path)
    clips = tmp_path / 'v2'
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, str(ROOT / 'tools/build_genfill_clips.py'), '--out', str(clips), '--registration', str(reg),
                        '--root', str(g).replace('\\', '/') + '/', '--frames', '5', '--height', '64', '--width', '96',
                        '--workers', '2', '--tiny', '--clips-per-hdri', '4', '--clips-per-fxtd', '2', '--dev-fraction', '0',
                        '--device', 'cpu'], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((clips / 'summary.json').read_text())
    assert s['recipe'] == 'v2' and s['has_core']
    assert any(clips.glob('*_core.npy')) and any(clips.glob('*_ylog.npy'))
    meta = [json.loads(l) for l in (clips / 'clips.jsonl').read_text().splitlines()]
    assert sum(m['source'] == 'fxtd_manchale' for m in meta) == 2          # FXTD x2
    assert all('core_fraction' in m for m in meta)
    from training.genfill_train import ClipStore
    st = ClipStore(clips, 'train')
    assert st.has_core()
    core, ylog = st.get_core(0)
    assert core.shape == (5, 64, 96) and ylog.shape == (64, 96) and set(core.unique().tolist()) <= {0.0, 1.0}
    # a v1 LoRA to start from (rank 4, tiny)
    import torch
    from genfill_tiny import tiny_components
    from rudra.genfill.model import add_lora, save_lora
    _, tr0, _ = tiny_components(20261028)
    add_lora(tr0, rank=4)
    save_lora(tr0, tmp_path / 'v1.safetensors', {'step': 1})
    out = tmp_path / 'run'
    r = subprocess.run([sys.executable, str(ROOT / 'training/genfill_train.py'), '--clips', str(clips), '--base', 'tiny',
                        '--out', str(out), '--steps', '6', '--save-every', '3', '--accumulate', '1', '--lr', '1e-3',
                        '--warmup', '1', '--device', 'cpu', '--tiny', '--no-checkpointing', '--init-lora', str(tmp_path / 'v1.safetensors'),
                        '--calib-weight', '0.5', '--calib-sigma', '1.0', '--over-weight', '4'],
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads((out / 'report.json').read_text())
    assert rep['weight_on'] == 'core' and rep['init_lora_sha256'] and rep['steps_done'] == 6
    log = [json.loads(l) for l in (out / 'log.jsonl').read_text().splitlines()]
    assert any(x.get('calib') is not None for x in log)                  # the calibration term ran


def test_calibration_loss_punishes_overshoot_more():
    import torch
    from genfill_tiny import tiny_components
    from rudra.genfill import encoding
    from training.genfill_train import calibration_loss
    vae = tiny_components(0)[0]
    zc = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, zc, 1, 1, 1)
    inv = 1.0 / torch.tensor(vae.config.latents_std).view(1, zc, 1, 1, 1)
    x0 = torch.randn(1, 16, 2, 4, 6) * 0.1
    with torch.no_grad():
        img = vae.decode(x0[:, :, :1] / inv + mean, return_dict=False)[0][:, :, 0].clamp(-1, 1)
        y = (encoding.decode(img)[0] * torch.tensor([.2722287168, .6740817658, .0536895174]).view(3, 1, 1)).sum(0)
    core = torch.ones(32, 48)
    truth = torch.log2(y.clamp_min(1e-6))
    over = calibration_loss(vae, x0, core, truth - 0.5, 4.0, mean, inv)     # prediction 0.5 stop too bright
    under = calibration_loss(vae, x0, core, truth + 0.5, 4.0, mean, inv)    # 0.5 stop too dark
    assert float(over) == pytest.approx(4 * float(under), rel=1e-3)
    assert float(calibration_loss(vae, x0, torch.zeros(32, 48), truth, 4.0, mean, inv)) == 0.0
    x = x0.clone().requires_grad_(True)
    calibration_loss(vae, x, core, truth - 0.5, 4.0, mean, inv).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0                    # it reaches the latents


def test_evaluator_v3_reports_the_brightness_gate(tmp_path):
    g, _ = _tree(tmp_path)
    from rudra.genfill.brightness import FEATURES, BrightnessEstimator
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURES))).astype(np.float32)
    BrightnessEstimator().fit(X, np.abs(X[:, 0]).astype(np.float32), epochs=20).save(tmp_path / 'b.json')
    out = tmp_path / 'ev3'
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_genfill.py'), '--out', str(out), '--tiny',
                        '--root', str(g).replace('\\', '/') + '/', '--frames', '5', '--height', '64', '--width', '96',
                        '--window', '5', '--overlap', '2', '--steps', '2', '--ph-offset', '0', '--fxtd-pool', 'extra',
                        '--brightness-model', str(tmp_path / 'b.json'), '--brightness-offset', '0.5',
                        '--registration', str(ROOT / 'configs/experiments/claude_genfill_v1_1_20261029.json')],
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads((out / 'report.json').read_text())
    assert any('b_mae_est' in x for x in rep['rows'])
    assert 'B_brightness' in rep['gates'] and rep['gates']['R']['pass']
    fx = [x for x in rep['rows'] if not x['scene_id'].startswith('ph:')]
    assert fx and all(x['secondary'] for x in fx)                 # reused FXTD scenes: reported, not gated
