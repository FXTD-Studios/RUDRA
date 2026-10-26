"""Phase-compatible fixed-size crops for the two-stride ACES predictor."""


def aligned_partial_origin(cy, cx, height, width, crop, border):
    if crop % 4 or height % 4 or width % 4:
        raise ValueError('aligned fixed crops require dimensions divisible by four; use full frames for other shapes')
    if border < 64 or crop <= 2 * border or min(height, width) < crop:
        raise ValueError('aligned crops require border>=64 and a nonempty interior')
    y = min(max(int(cy) - crop // 2, 0), height - crop) // 4 * 4
    x = min(max(int(cx) - crop // 2, 0), width - crop) // 4 * 4
    return y, x
