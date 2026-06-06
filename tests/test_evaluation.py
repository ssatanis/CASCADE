"""Tests for the Phase-3 statistical evaluation suite (synthetic, seeded — TEST ONLY)."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from cascade import evaluation as ev


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.integers(0, 2, n)
    strong = y * 0.9 + rng.normal(0, 1, n)
    weak = y * 0.3 + rng.normal(0, 1, n)
    return y, strong, weak


def test_delong_auroc_matches_sklearn(data):
    y, strong, _ = data
    auc, var = ev.delong_roc_variance(y, strong)
    assert abs(auc - roc_auc_score(y, strong)) < 1e-9
    assert var > 0


def test_delong_variance_matches_bootstrap(data):
    y, strong, _ = data
    _, var = ev.delong_roc_variance(y, strong)
    rng = np.random.default_rng(1)
    n = len(y)
    boot = [roc_auc_score(y[i], strong[i]) for i in (rng.integers(0, n, n) for _ in range(1500))]
    assert abs(np.sqrt(var) - np.std(boot)) < 0.003


def test_delong_paired_detects_difference(data):
    y, strong, weak = data
    r = ev.delong_roc_test(y, strong, weak)
    assert r["delta"] > 0 and r["p"] < 1e-6


def test_delong_ties_handled():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    binary_score = (y * 0.5 + rng.normal(0, 1, 2000) > 0.5).astype(float)
    auc, _ = ev.delong_roc_variance(y, binary_score)
    assert abs(auc - roc_auc_score(y, binary_score)) < 1e-9


def test_perfect_calibration_low_ece():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)  # y ~ Bernoulli(p) → perfectly calibrated
    assert ev.expected_calibration_error(y, p, 15) < 0.02
    assert ev.adaptive_ece(y, p, 15) < 0.02


def test_miscalibration_detected():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < np.clip(p - 0.3, 0, 1)).astype(int)  # overconfident
    assert ev.expected_calibration_error(y, p, 15) > 0.1


def test_wilson_ci_brackets_rate():
    lo, hi = ev.wilson_ci(50, 100)
    assert lo < 0.5 < hi and 0 <= lo and hi <= 1


def test_murphy_uncertainty_is_base_variance():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 5000)
    p = rng.uniform(0, 1, 5000)
    d = ev.murphy_decomposition(y, p, 15)
    base = y.mean()
    assert abs(d["uncertainty"] - base * (1 - base)) < 1e-9
    assert d["reliability"] >= 0 and d["resolution"] >= 0


def test_negative_control_permuted_label_is_chance(data):
    y, strong, _ = data
    nc = ev.negative_control_permuted_label(y, strong, n_perm=200, seed=0)
    assert nc["passes"] and abs(nc["mean_auroc"] - 0.5) < 0.02


def test_cluster_bootstrap_ci_contains_point(data):
    y, strong, _ = data
    groups = np.arange(len(y)) % 200  # 200 clusters
    cb = ev.cluster_bootstrap_auroc(y, strong, groups, n_boot=500, seed=0)
    assert cb["ci_low"] <= cb["auroc"] <= cb["ci_high"]
    assert cb["n_groups"] == 200


def test_permutation_test_significant_for_real_signal(data):
    y, strong, _ = data
    pt = ev.permutation_test_auroc(y, strong, n_perm=500, seed=0)
    assert pt["p_value"] < 0.01 and pt["auroc"] > 0.6
