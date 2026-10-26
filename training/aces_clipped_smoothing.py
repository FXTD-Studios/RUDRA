"""Remove unsupported texture inside fully clipped highlights (research, opt-in).

Inside source pixels where every channel clipped, the SDR carries no spatial
information. Measured on new_motion DEV (spatial600/parent), predicted
log-luminance gradients there are nearly uncorrelated with the HDR reference,
so they add error. This keeps each pixel's chroma ratio and replaces its
log2 luminance with a support-normalised Gaussian average over the clipped
region, blended in by distance from the clipping boundary. Pixels outside the
fully clipped support are returned bit-exact.
"""
import cv2
import numpy as np

LUMA_AP1 = np.array([.2722287168, .6740817658, .0536895174])


def smooth_clipped(pred, source, sigma=16.0, ramp=1.0, start=0.0, threshold=254.5/255, mode='gaussian'):
    pred = np.asarray(pred, np.float64)
    full = np.asarray(source).min(-1) >= threshold
    y = pred @ LUMA_AP1
    ok = full & (y > 1e-10)
    if not ok.any():
        return pred.astype(np.float32), full
    ly = np.where(ok, np.log2(np.maximum(y, 1e-10)), 0.0)
    n = 0
    if mode == 'component':
        n, labels = cv2.connectedComponents(full.astype(np.uint8), connectivity=8)
        sums = np.bincount(labels[ok], ly[ok], minlength=n)
        counts = np.bincount(labels[ok], minlength=n)
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        ls = np.where(ok, means[labels], ly)
    else:
        k = int(4 * sigma) | 1
        num = cv2.GaussianBlur(ly, (k, k), sigma)
        den = cv2.GaussianBlur(ok.astype(np.float64), (k, k), sigma)
        ls = np.where(den > 1e-6, num / np.maximum(den, 1e-6), ly)
        if mode == 'hybrid':
            # Exactly flat log-luminance in each eroded clipped interior
            # (no unsupported gradient), smooth Gaussian field near edges.
            interior = cv2.erode(full.astype(np.uint8), np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8),
                                 iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) & ok
            n, labels = cv2.connectedComponents(interior.astype(np.uint8), connectivity=8)
            if n > 1:
                sums = np.bincount(labels[interior], ls[interior], minlength=n)
                counts = np.bincount(labels[interior], minlength=n)
                ls = np.where(interior, (sums / np.maximum(counts, 1))[labels], ls)
    d = cv2.distanceTransform(full.astype(np.uint8), cv2.DIST_L2, 5)
    w = np.clip((d - start) / ramp, 0, 1) * ok
    scale = np.where(ok, 2.0 ** (w * (ls - ly)), 1.0)
    out = pred * scale[..., None]
    out[~full] = pred[~full]
    if mode == 'hybrid' and n > 1:
        # Identical RGB across each interior: chroma is also unobserved there,
        # and float32 rounding of per-pixel chroma would otherwise leave tiny gradients.
        for c in range(3):
            sums = np.bincount(labels[interior], out[..., c][interior], minlength=n)
            out[..., c] = np.where(interior, (sums / np.maximum(counts, 1))[labels], out[..., c])
    return out.astype(np.float32), full
