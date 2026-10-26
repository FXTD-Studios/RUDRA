import pytest
import torch

from rudra.sdr2hdr import SDR2HDRNet
from tools.export_run05_direct import DirectHDR


def test_direct_export_preserves_active_restoration_at_unseen_size(tmp_path):
    torch.set_num_threads(2)
    model = SDR2HDRNet(base_channels=4, baseline_source=True, bounded_source=True,
                       reconstruction_policy='observed_clip_v1').eval()
    source = torch.full((1, 3, 31, 45), .5)
    with torch.inference_mode():
        original = model(source).hdr
        model.dequant.bias.fill_(1)
        expected = model(source).hdr
        assert (expected - original).abs().max() > 1e-7
        graph = torch.jit.trace(DirectHDR(model), torch.zeros(1, 3, 16, 16))
        graph.save(str(tmp_path / 'graph.pt'))
        restored = torch.jit.load(str(tmp_path / 'graph.pt'))
        torch.testing.assert_close(restored(source), expected, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize('flag', ['gate_conditioning', 'shadow_conditioning'])
def test_frame_conditioning_requires_explicit_export_support(flag):
    model = SDR2HDRNet(base_channels=4, baseline_source=True, bounded_source=True,
                       **{flag: True})
    with pytest.raises(ValueError, match='frame-conditioned'):
        DirectHDR(model)
