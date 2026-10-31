import numpy as np
import OpenEXR
import pytest

from training.exr_any import read_rgb


@pytest.mark.parametrize("compression", [OpenEXR.ZIP_COMPRESSION, OpenEXR.PIZ_COMPRESSION, OpenEXR.NO_COMPRESSION])
def test_compressed_rgb_round_trip(tmp_path, compression):
    rgb = np.random.default_rng(0).uniform(0, 60000, (12, 20, 3)).astype(np.float16)
    path = tmp_path / "x.exr"
    with OpenEXR.File({"compression": compression, "type": OpenEXR.scanlineimage}, {"RGB": rgb}) as f:
        f.write(str(path))
    out = read_rgb(path)
    assert out.dtype == np.float32 and out.shape == (12, 20, 3)
    np.testing.assert_array_equal(out, rgb.astype(np.float32))


def test_separate_channels(tmp_path):
    planes = {c: np.full((4, 5), v, np.float32) for c, v in zip("RGB", (1.0, 2.0, 3.0))}
    path = tmp_path / "y.exr"
    with OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION}, planes) as f:
        f.write(str(path))
    assert read_rgb(path)[0, 0].tolist() == [1.0, 2.0, 3.0]
