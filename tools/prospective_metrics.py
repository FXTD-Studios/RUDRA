"""Prospective still metrics and explicit equal-frame coverage aggregation."""
import math


def measure(methods, truth, regions, view):
    import numpy as np
    from scipy.ndimage import binary_erosion
    from tools.audit_aces_learned_detail import common_detail
    if set(methods) != {'analytic', 'candidate'}:
        raise ValueError('Both methods required')
    if any(v.shape != truth.shape or not np.isfinite(v).all() for v in [truth, *methods.values()]):
        raise ValueError('Invalid metric arrays')
    metrics = {k: {} for k in methods}
    for name, rgb in methods.items():
        for region, support in regions.items():
            metrics[name][region + '_asinh_mse'] = float(np.mean((np.arcsinh(rgb[support]) - np.arcsinh(truth[support]))**2)) if support.any() else None
        for ev in (-3, 0, 3):
            delta = (view.forward(rgb * 2.**ev) - view.forward(truth * 2.**ev)) * 255
            metrics[name][f'grade_{ev}_mse'] = float(np.mean(delta[regions['highlight']]**2)) if regions['highlight'].any() else None
    for region in ('highlight', 'fully_clipped'):
        detail = common_detail(methods, truth, binary_erosion(regions[region], iterations=2))
        for name in methods:
            m = detail['methods'][name]
            metrics[name][region + '_detail_rmse_ev'] = m['rmse_ev'] if m['edges'] > 0 else None
    return metrics


def aggregate(rows, scene_ids, precisions):
    if not rows:
        raise ValueError('No evaluation rows')
    metrics = set(rows[0]['metrics']['candidate'])
    for row in rows:
        if any(set(row['metrics'][method]) != metrics for method in ('analytic', 'candidate')):
            raise ValueError('Inconsistent metric keys')
        for metric in metrics:
            pair = [row['metrics'][method][metric] for method in ('analytic', 'candidate')]
            if (pair[0] is None) != (pair[1] is None):
                raise ValueError('Unmatched metric support')
            if pair[0] is not None and any(not math.isfinite(v) or v < 0 for v in pair):
                raise ValueError('Invalid metric value')
    groups = {}
    for scene in [None, *scene_ids]:
        for bits in precisions:
            subset = [r for r in rows if r['case']['source_bits'] == bits and (scene is None or r['case']['scene_id'] == scene)]
            summary = {}
            for metric in sorted(metrics):
                valid = [r for r in subset if r['metrics']['candidate'][metric] is not None]
                values = {m: math.fsum(r['metrics'][m][metric] for r in valid) / len(valid) if valid else None for m in ('analytic', 'candidate')}
                summary[metric] = {**values, 'supporting_cases': len(valid), 'total_cases': len(subset),
                                   'no_worse_analytic': bool(valid) and values['candidate'] <= values['analytic']}
            groups[f'{scene or "pooled"}/bits{bits}'] = summary
    return groups
