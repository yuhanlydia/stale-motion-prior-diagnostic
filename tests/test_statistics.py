from stale_motion_prior.metrics import post_change_distance_auc, recovery_lag
from stale_motion_prior.statistics import paired_bootstrap


def test_mechanism_metrics():
    distances = [5, 6, 5, 4, 3]
    assert post_change_distance_auc(distances, 0, 3) == 5
    assert recovery_lag(distances, 0) == 2


def test_paired_bootstrap_positive():
    result = paired_bootstrap([1, 2, 3], draws=1000, seed=1)
    assert result["estimate"] == 2
    assert result["ci_low"] > 0
