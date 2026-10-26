import numpy as np
import pytest

from rudra.delivery.exr import read_exr, write_exr
from rudra.delivery.aces import write_acescg_exr


def test_float32_mask_and_alpha_are_distinct_and_independently_readable(tmp_path):
    OpenEXR = pytest.importorskip('OpenEXR')
    Imath = pytest.importorskip('Imath')
    rgb = np.arange(80, dtype=np.float32).reshape(4, 5, 4) / 13
    mask = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    path = write_exr(tmp_path / 'mask.exr', rgb, half=False, reconstruction_mask=mask)
    image, attrs = read_exr(path)
    np.testing.assert_array_equal(image, rgb)
    np.testing.assert_array_equal(attrs['reconstruction_mask'], mask)
    handle = OpenEXR.InputFile(str(path))
    try:
        channels = handle.header()['channels']
        assert set(channels) == {'R', 'G', 'B', 'A', 'rudra.reconstructed'}
        for name, expected in [('R', rgb[..., 0]), ('A', rgb[..., 3]), ('rudra.reconstructed', mask)]:
            assert channels[name].type == Imath.PixelType(Imath.PixelType.FLOAT)
            values = np.frombuffer(handle.channel(name, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(4, 5)
            np.testing.assert_array_equal(values, expected)
    finally:
        handle.close()


@pytest.mark.parametrize('mask', [np.zeros((2, 3)), np.full((4, 5), np.nan), np.full((4, 5), 1.1)])
def test_invalid_mask_rejected_before_file_creation(tmp_path, mask):
    path = tmp_path / 'bad.exr'
    with pytest.raises(ValueError, match='reconstruction_mask'):
        write_exr(path, np.zeros((4, 5, 3)), reconstruction_mask=mask)
    assert not path.exists()


def test_acescg_float32_keeps_large_values_and_mask(tmp_path):
    pixels = np.full((2, 3, 3), 70000, np.float32)
    mask = np.zeros((2, 3), np.float32)
    mask[0, 0] = 1
    path = write_acescg_exr(pixels, tmp_path / 'cg.exr', source_space='ap1',
                           half=False, reconstruction_mask=mask)
    image, attrs = read_exr(path)
    np.testing.assert_allclose(image, pixels, rtol=1e-6)
    np.testing.assert_array_equal(attrs['reconstruction_mask'], mask)
