import pytest
from rudra.sdr2hdr import SDR2HDRNet
from tools.export_model import NativeModel


@pytest.mark.parametrize('bounded',[False,True])
def test_native_v1_refuses_unsupported_picker_and_restoration_heads(bounded):
    net=SDR2HDRNet(base_channels=4,baseline_source=True,bounded_source=bounded)
    with pytest.raises(ValueError,match='cannot represent'):
        NativeModel(net)


def test_native_v1_still_accepts_legacy_model():
    net=SDR2HDRNet(base_channels=4)
    NativeModel(net)
