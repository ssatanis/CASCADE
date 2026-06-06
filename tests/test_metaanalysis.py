import numpy as np
import pytest

from cascade.metaanalysis import fixed_effect, random_effects


def test_fixed_effect_matches_closed_form():
    betas = np.array([1.0, 2.0, 1.5])
    variances = np.array([0.25, 1.0, 0.5])
    m = fixed_effect(betas, variances)
    w = 1 / variances
    expect = (w * betas).sum() / w.sum()
    assert m.effect == pytest.approx(expect)
    assert m.variance == pytest.approx(1 / w.sum())


def test_pooled_variance_not_worse_than_best_single():
    variances = np.array([0.5, 0.2, 0.8])
    m = fixed_effect(np.array([1.0, 1.1, 0.9]), variances)
    # inverse-variance pooling: pooled variance <= smallest single variance
    assert m.variance <= variances.min() + 1e-12


def test_inverse_variance_beats_uniform_average_variance():
    # Gauss-Markov: 1/Σ(1/σ²) <= (Σσ²)/k²  (AM-HM inequality)
    variances = np.array([0.1, 0.5, 2.0, 0.3])
    betas = np.array([1.0, 1.0, 1.0, 1.0])
    m = fixed_effect(betas, variances)
    uniform_var = variances.sum() / len(variances) ** 2
    assert m.variance <= uniform_var + 1e-12


def test_random_effects_zero_tau2_when_homogeneous():
    betas = np.array([1.0, 1.0, 1.0])
    variances = np.array([0.2, 0.2, 0.2])
    m = random_effects(betas, variances)
    assert m.tau2 == pytest.approx(0.0, abs=1e-9)
    assert m.effect == pytest.approx(1.0)


def test_random_effects_positive_tau2_when_heterogeneous():
    betas = np.array([0.0, 5.0, -3.0, 8.0])
    variances = np.array([0.05, 0.05, 0.05, 0.05])
    m = random_effects(betas, variances)
    assert m.tau2 > 0
    assert m.i2 > 50  # substantial heterogeneity
    # RE is wider than FE under heterogeneity
    fe = fixed_effect(betas, variances)
    assert m.variance >= fe.variance


def test_quality_weighting_downweights_low_quality_outlier():
    betas = np.array([1.0, 1.0, 10.0])  # third is a bad outlier
    variances = np.array([0.2, 0.2, 0.2])
    high_q = random_effects(betas, variances, quality=np.array([1.0, 1.0, 0.01]))
    flat = random_effects(betas, variances, quality=np.array([1.0, 1.0, 1.0]))
    assert abs(high_q.effect - 1.0) < abs(flat.effect - 1.0)


def test_single_study_returns_itself():
    m = random_effects(np.array([2.5]), np.array([0.4]))
    assert m.effect == pytest.approx(2.5)
    assert m.k == 1
    assert m.tau2 == 0.0


def test_validation_errors():
    with pytest.raises(ValueError):
        fixed_effect(np.array([1.0]), np.array([-0.1]))
    with pytest.raises(ValueError):
        random_effects(np.array([]), np.array([]))
