import numpy as np
import pytest
import torch

from pipeline.run05_corpus import canonical, declared_inverse
from training.contract_checks import roundtrip
from training.sdr2hdr_dataset import _crop_params, _resize_crop


def test_native_crop_keeps_nonlinear_source_target_relationship():
    rng = np.random.default_rng(31)
    src = rng.uniform(.03, .93, (60, 80, 3)).astype(np.float32)
    canon = canonical(src)
    hdr = declared_inverse(canon, 'bt1886')
    params = _crop_params(60, 80, 24, False, native=True)
    assert params == (60, 80, 18, 28)
    def crop(a):
        return _resize_crop(torch.from_numpy(a).permute(2, 0, 1), params, 24).permute(1, 2, 0).numpy()
    assert np.array_equal(crop(canon), canon[18:42, 28:52])
    assert roundtrip(crop(hdr), crop(canon))['r_pass']


def test_native_crop_refuses_upsampling_and_legacy_resize_stays_available():
    with pytest.raises(ValueError, match='native dimensions'):
        _crop_params(20, 40, 24, False, native=True)
    assert _crop_params(60, 80, 24, False) == (24, 32, 0, 4)
