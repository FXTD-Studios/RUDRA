import numpy as np
from tools.audit_restoration_bounds import bound_metrics


def test_bound_is_attained_by_projecting_clean_target_to_source_interval():
    clean = np.array([10., 20., 30., 40.])
    observed = np.array([10., 19., 34., 39.8])
    for budget in (.49, 1., 2.):
        oracle = np.clip(clean, observed-budget, observed+budget)
        result = bound_metrics(clean, observed, budget)
        assert np.isclose(result['oracle_rmse'], np.sqrt(np.mean((oracle-clean)**2)))
