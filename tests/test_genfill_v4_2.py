"""AI fill v4.2 (14 Nov 2026): held-still region clips, the truth's structure in the region, sampling
weighted by it, and the frame-to-frame term. Registration: configs/experiments/research_genfill_v4_2_20261114.json."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
torch = pytest.importorskip('torch')
pytest.importorskip('cv2')


def test_region_structure_reads_detail_not_level():
    from tools.build_genfill_clips import region_structure
    rng = np.random.default_rng(0)
    mask = np.zeros((96, 160), bool); mask[16:80, 32:128] = True
    flat = np.full((96, 160, 3), 4.0, np.float32)
    assert region_structure(flat, mask) < 1e-4
    assert region_structure(flat * 8, mask) < 1e-4                       # brighter is not more structure
    tex = flat * np.exp2(rng.normal(0, 0.3, (96, 160, 1))).astype(np.float32)
    assert region_structure(tex, mask) > 0.05
    assert region_structure(tex, np.zeros_like(mask)) == 0.0               # nothing left after the erosion


def test_a_held_still_region_clip_has_a_still_truth():
    pytest.importorskip('PyOpenColorIO')
    from tools.build_genfill_clips import region_clip
    rng = np.random.default_rng(3)
    H, W = 360, 640
    plate = np.full((H, W, 3), 0.18, np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    sky = (yy < 150)
    plate[sky] = 3.0 * np.exp2(0.4 * np.sin(xx[sky] / 7.0) * np.cos(yy[sky] / 5.0))[:, None]
    job = {'kind': 'plate', 'asset_id': 'a', 'scene_id': 's', 'source': 'x', 'licence': 'CC0', 'split': 'train'}
    clip = None
    for _ in range(20):
        clip = region_clip(job, rng, plate, True, 9, 96, 160, static_share=1.0)
        if clip is not None:
            break
    assert clip is not None, 'no region found in the synthetic plate'
    m = clip['meta']
    assert m['static'] is True and m['kind'] == 'region'
    t = clip['truth'].astype(np.float32)
    assert np.abs(t - t[:1]).max() == 0                                    # the truth does not move
    assert m['structure'] > 0.0
    clip0, rng0 = None, np.random.default_rng(4)
    for _ in range(20):
        clip0 = region_clip(job, rng0, plate, True, 9, 96, 160)    # v4: never still
        if clip0 is not None:
            break
    assert clip0 is not None and clip0['meta']['static'] is False


def test_sampling_weights_favour_structured_region_clips():
    from training.genfill_train import sampling_weights
    meta = [{'kind': 'region', 'structure': 0.0}, {'kind': 'region', 'structure': 0.05},
            {'kind': 'region', 'structure': 0.4}, {'kind': 'highlight'}, {'kind': 'region'}]
    w = sampling_weights(meta, 3.0, 0.10).tolist()
    assert w == pytest.approx([1.0, 2.5, 4.0, 1.0, 1.0])
    assert sampling_weights(meta, 0.0, 0.10).tolist() == [1.0] * 5


def test_temporal_loss_is_zero_when_the_fill_moves_as_the_truth():
    from training.genfill_train import temporal_loss
    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(16, 4, 8, 12, generator=g)
    mask = torch.zeros(13, 64, 96); mask[:, 16:48, 24:72] = 1
    assert float(temporal_loss(z0[None].clone(), z0, mask)) == pytest.approx(0.0, abs=1e-9)
    assert float(temporal_loss(z0[None] + 5.0, z0, mask)) == pytest.approx(0.0, abs=1e-9)     # a constant offset is not flicker
    flick = z0[None].clone(); flick[:, :, 2] += 1.0
    assert float(temporal_loss(flick, z0, mask)) > 0.5
    assert float(temporal_loss(flick, z0, torch.zeros_like(mask))) == 0.0                       # outside the region: free
    assert float(temporal_loss(z0[None, :, :2], z0[:, :2], mask)) == 0.0                         # too few latent frames


def _store(root: Path, n=6, frames=9):
    """An encoded store as the builder writes it, with random latents (tiny Wan shapes)."""
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    f = (frames - 1) // 4 + 1
    z0 = rng.standard_normal((n, 16, f, 8, 12)).astype(np.float16)
    ctx = rng.standard_normal((n, 32, f, 8, 12)).astype(np.float16)
    mask = np.zeros((n, frames, 64, 96), bool); mask[:, :, 16:48, 24:72] = True
    np.save(root / 's000_z0.npy', z0); np.save(root / 's000_ctx.npy', ctx)
    np.save(root / 's000_mask.npy', np.stack([np.packbits(m, axis=None) for m in mask]))
    meta = []
    for i in range(n):
        kind = 'region' if i % 2 == 0 else 'highlight'
        m = {'shard': 's000', 'index': i, 'split': 'dev' if i == n - 1 else 'train', 'kind': kind, 'licence': 'CC0',
             'asset_id': f'a{i}', 'scene_id': f's{i}', 'source': 'x'}
        if kind == 'region':
            m.update({'structure': 0.02 * i, 'static': i % 4 == 0})
        meta.append(m)
    (root / 'clips.jsonl').write_text('\n'.join(json.dumps(m) for m in meta) + '\n')
    (root / 'summary.json').write_text(json.dumps({'clips': n, 'frames': frames, 'size': [64, 96], 'encoded': True,
                                                   'commercial_clean': True, 'recipe': 'v4_2', 'has_core': False}))


def test_trainer_runs_with_structure_sampling_and_the_temporal_term(tmp_path):
    pytest.importorskip('diffusers')
    clips = tmp_path / 'clips'
    _store(clips)
    out = tmp_path / 'run'
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, str(ROOT / 'training/genfill_train.py'), '--clips', str(clips), '--base', 'tiny',
                        '--out', str(out), '--steps', '6', '--save-every', '3', '--accumulate', '2', '--rank', '4',
                        '--lr', '3e-3', '--warmup', '1', '--device', 'cpu', '--tiny', '--no-checkpointing',
                        '--weight-on', 'mask', '--structure-boost', '3', '--temporal-weight', '1', '--temporal-sigma', '1.0'],
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads((out / 'report.json').read_text())
    assert rep['steps_done'] == 6 and rep['sampling']['structure_boost'] == 3.0
    assert rep['sampling']['weighted_clips'] >= 1
    logs = [json.loads(l) for l in (out / 'log.jsonl').read_text().splitlines()]
    assert any(l.get('temporal') is not None and l['temporal'] > 0 for l in logs)
