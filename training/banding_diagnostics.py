"""Reference-anchored diagnostics supplementing the historical flat-step score.

The flat-step count can be reduced by adding noise, so it cannot establish
dequantization quality alone. These errors use one mask derived from truth,
include RGB rather than only luma, and compare values and spatial derivatives.
No acceptance threshold is inferred from the synthetic regression tests.
"""
import numpy as np

from training.contract_checks import PUSHES, view_forward, canonical_to_source

LUMA709 = np.array([.2126, .7152, .0722], np.float32)


def reference_gradient_errors(output, target, view='bt1886'):
    import cv2
    output, target = np.asarray(output), np.asarray(target)
    if output.shape != target.shape or output.ndim != 3 or output.shape[2] != 3:
        raise ValueError('expected matching HWC RGB images')
    if min(output.shape[:2]) < 3 or not np.isfinite(output).all() or not np.isfinite(target).all():
        raise ValueError('expected finite images at least 3x3')
    rows = []
    for push in PUSHES:
        pred = canonical_to_source(view_forward(output * 2.0**push, view)) * 1023
        truth = canonical_to_source(view_forward(target * 2.0**push, view)) * 1023
        luma = truth @ LUMA709
        smooth = cv2.GaussianBlur(luma, (0, 0), 4)
        sy, sx = np.gradient(smooth)
        slope = np.hypot(sx, sy)
        keep = (slope > .03) & (slope < 1.5) & (luma > 2) & (luma < 1020)
        # Borders have no two-sided derivatives. Mask is identical for every
        # candidate compared with this target; candidates cannot select pixels.
        keep[[0, -1], :] = False
        keep[:, [0, -1]] = False
        pixels = int(keep.sum())
        row = {'push': push, 'reference_gentle_pixels': pixels}
        if pixels < 500:
            row.update(code_rmse=None, gradient_rmse=None, curvature_rmse=None)
            rows.append(row)
            continue
        residual = (pred - truth).astype(np.float64)
        row['code_rmse'] = float(np.sqrt(np.mean(residual[keep]**2)))
        gradients, curvatures = [], []
        for axis in (0, 1):
            before = np.roll(residual, 1, axis=axis)
            after = np.roll(residual, -1, axis=axis)
            # Forward differences do not cancel alternating pixel noise.
            gradients.append((after - residual)[keep])
            curvatures.append((after - 2 * residual + before)[keep])
        row['gradient_rmse'] = float(np.sqrt(np.mean(np.concatenate(gradients)**2)))
        row['curvature_rmse'] = float(np.sqrt(np.mean(np.concatenate(curvatures)**2)))
        rows.append(row)
    return {'pushes': rows,
            'measured_pushes': sum(r['code_rmse'] is not None for r in rows),
            'units': '10-bit Rec709 source-view codes; reference-defined gentle regions',
            'acceptance_threshold': None}
