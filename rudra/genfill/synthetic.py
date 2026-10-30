"""A small synthetic RUDRA master (EXR folder) with a moving clipped sun and a crushed corner,
written the way the native engine writes masters (uncompressed scanlines, chromaticities,
rudra.reconstructed). For tests and the GENFILL.ps1 smoke fill."""
from pathlib import Path

import numpy as np

AP1 = (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)
AP0 = (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767)


def write_master(folder: Path, n=7, h=48, w=64, ap0=False, half=False):
    import OpenEXR
    from rudra.delivery.colorspace import convert
    folder.mkdir()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for i in range(n):
        base = 0.05 + 0.5 * (xx / w)[..., None] * np.array([1.0, 0.9, 0.8], np.float32)
        sun = 500.0 * np.exp(-(((xx - 20 - 2 * i) ** 2 + (yy - 14) ** 2) / 30.0))[..., None]
        img = (base + sun).astype(np.float32)
        img[36:, :20] = 0.0003                              # crushed corner
        mask = (sun[..., 0] > 30).astype(np.float32)
        if ap0:
            img = convert(img, 'ap1', 'ap0')
        dt = np.float16 if half else np.float32
        ch = {'R': img[..., 0].astype(dt), 'G': img[..., 1].astype(dt), 'B': img[..., 2].astype(dt),
              'rudra.reconstructed': mask.astype(dt)}
        hdr = {'compression': OpenEXR.NO_COMPRESSION, 'type': OpenEXR.scanlineimage,
               'chromaticities': AP0 if ap0 else AP1, 'rudra:container': 'test'}
        OpenEXR.File(hdr, ch).write(str(folder / f'shot.{1001 + i:06d}.exr'))
    (folder / 'provenance.json').write_text('{"test": true}')
