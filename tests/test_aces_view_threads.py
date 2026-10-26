import numpy as np
import pytest
from rudra.aces_view import ACES2View
from rudra.delivery.aces_video import ACES2HDR1000View


def test_parallel_exact_view_handles_odd_sized_buffers_and_signed_hdr():
    rng=np.random.default_rng(24)
    source=rng.random((271,1025,3),dtype=np.float32)
    source[:2]=0;source[-2:]=1
    serial=ACES2View();parallel=ACES2View(threads=4)
    inverse=serial.inverse(source)
    np.testing.assert_array_equal(parallel.inverse(source),inverse)
    hdr=rng.uniform(-.1,128,source.shape).astype(np.float32)
    np.testing.assert_array_equal(parallel.forward(hdr),serial.forward(hdr))
    np.testing.assert_array_equal(ACES2HDR1000View(threads=4).forward(hdr),ACES2HDR1000View().forward(hdr))


@pytest.mark.parametrize('threads',[0,17,1.5])
def test_invalid_thread_count_rejected(threads):
    with pytest.raises(ValueError):ACES2View(threads=threads)
