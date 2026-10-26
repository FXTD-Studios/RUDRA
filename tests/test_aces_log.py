import numpy as np
from rudra.delivery.aces_log import LogEncoder


def test_arri_reference_grey_and_black():
    grey = np.full((1, 1, 3), 0.18, np.float32)
    black = np.zeros((1, 1, 3), np.float32)
    assert abs(LogEncoder('logc3').encode(grey)[0].ravel()[0] - 0.391007) < 2e-6   # ARRI LogC3 EI800
    assert abs(LogEncoder('logc4').encode(grey)[0].ravel()[0] - 0.278396) < 2e-6   # ARRI LogC4
    assert abs(LogEncoder('logc3').encode(black)[0].ravel()[0] - 0.092809) < 2e-6


def test_neutrals_stay_neutral_and_clipping_is_counted():
    x = np.array([[[0.01] * 3, [1.0] * 3, [40.0] * 3, [300.0] * 3]], np.float32)
    codes, clipped = LogEncoder('logc3').encode(x)
    assert np.abs(codes - codes[..., :1]).max() < 1e-6
    assert clipped == 1                       # only 300 exceeds LogC3 EI800
    assert LogEncoder('logc4').encode(x)[1] == 0
    assert LogEncoder('logc3', -3.0).encode(x)[1] == 0   # exposure offset fits it


def test_encode_does_not_modify_input():
    x = np.full((2, 2, 3), 5.0, np.float32)
    LogEncoder('logc4', -1.0).encode(x)
    assert (x == 5.0).all()
