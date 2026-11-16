"""AI fill's GPU memory modes (16 Nov 2026, RUDRA 1.2.5): which mode a card gets, and what each mode
changes on the model. The memory each mode needs is measured on a GPU by tools/bench_genfill_memory.py."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from rudra.genfill.model import WanFill  # noqa: E402
from rudra.genfill.run import LOW_MEMORY_BELOW_GB, memory_mode  # noqa: E402


def test_auto_keeps_the_gated_fill_on_16_gb_and_saves_memory_below():
    assert memory_mode('auto', 23.6) == 'off'
    assert memory_mode('auto', 15.6) == 'off'          # a 16 GB card reports a little under 16
    assert memory_mode('auto', 11.7) == 'on'           # the 12 GB RTX 4080 Laptop GPU
    assert memory_mode('auto', 9.8) == 'on'
    assert memory_mode('auto', None) == 'off'          # no CUDA: the system's memory, nothing to swap
    assert memory_mode('swap', 24.0) == 'swap'         # asked for: as asked
    assert LOW_MEMORY_BELOW_GB == 15.5


class _Vae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(temperal_downsample=[False, True, True], z_dim=16,
                                      latents_mean=[0.0] * 16, latents_std=[1.0] * 16)
        self.use_tiling = False

    def enable_tiling(self):
        self.use_tiling = True


class _Transformer(torch.nn.Linear):
    def __init__(self):
        super().__init__(4, 4)
        self.config = SimpleNamespace(patch_size=[1, 2, 2])


def test_each_mode_sets_tiling_and_swapping_and_is_recorded():
    m = WanFill(_Vae(), _Transformer(), scheduler=None, prompts=None, device='cpu')
    assert m.memory_mode == 'off' and not m.vae.use_tiling
    m.set_memory_mode('tile')
    assert m.vae.use_tiling and m.identity['memory_mode'] == 'tile'
    m.set_memory_mode('on')
    assert m.vae.use_tiling
    assert not m._swap                                 # swapping is for CUDA only
    m.set_memory_mode('off')
    assert not m.vae.use_tiling and m.identity['memory_mode'] == 'off'
    with pytest.raises(ValueError):
        m.set_memory_mode('half')


def test_swap_moves_the_transformer_off_the_device_for_the_vae_and_back():
    m = WanFill(_Vae(), _Transformer(), scheduler=None, prompts=None, device='cpu')
    m._swap = True                                     # as on CUDA; the moves are what is checked
    moves = []
    m.transformer.to = lambda where: moves.append(str(where)) or m.transformer
    m._transformer_to_device(False)
    m._transformer_to_device(False)                    # already off: no second copy
    m._transformer_to_device(True)
    assert moves == ['cpu', 'cpu']                     # the device here is the CPU itself
    assert m._transformer_on_device
