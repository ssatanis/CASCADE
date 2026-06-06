import math

import numpy as np
import pytest

from cascade.federated.dp import (
    RDPAccountant,
    amplify_by_subsampling,
    analytic_gaussian_sigma,
    clip_l2,
    dp_sgd_noisy_mean,
    gaussian_delta,
)


def test_calibrated_sigma_reproduces_target_delta():
    for eps in (0.5, 1.0, 3.0, 5.0):
        delta = 1e-6
        sigma = analytic_gaussian_sigma(eps, delta, sensitivity=1.0)
        achieved = gaussian_delta(eps, sigma, sensitivity=1.0)
        assert achieved == pytest.approx(delta, rel=1e-3)


def test_larger_epsilon_needs_less_noise():
    s_loose = analytic_gaussian_sigma(5.0, 1e-6)
    s_tight = analytic_gaussian_sigma(0.5, 1e-6)
    assert s_tight > s_loose  # stronger privacy → more noise


def test_sensitivity_scales_sigma_linearly():
    s1 = analytic_gaussian_sigma(2.0, 1e-6, sensitivity=1.0)
    s10 = analytic_gaussian_sigma(2.0, 1e-6, sensitivity=10.0)
    assert s10 == pytest.approx(10 * s1, rel=1e-3)


def test_rdp_upper_bounds_analytic_epsilon():
    # For a single Gaussian mechanism, RDP-derived epsilon must be >= the exact
    # analytic epsilon (RDP is an upper bound).
    delta = 1e-6
    target_eps = 2.0
    sigma = analytic_gaussian_sigma(target_eps, delta)
    acct = RDPAccountant().add_gaussian(noise_multiplier=sigma, steps=1)
    rdp_eps, _ = acct.get_epsilon(delta)
    assert rdp_eps >= target_eps - 1e-6


def test_composition_scales_epsilon():
    nm = 4.0
    single = RDPAccountant().add_gaussian(nm, steps=1).get_epsilon(1e-6)[0]
    many = RDPAccountant().add_gaussian(nm, steps=100).get_epsilon(1e-6)[0]
    assert many > single
    # T compositions of σ == one release with σ/sqrt(T): RDP scales linearly in T
    a = RDPAccountant().add_gaussian(nm, steps=100)
    b = RDPAccountant().add_gaussian(nm / math.sqrt(100), steps=1)
    assert a.get_epsilon(1e-6)[0] == pytest.approx(b.get_epsilon(1e-6)[0], rel=1e-6)


def test_clip_l2_bounds_norm():
    v = np.array([3.0, 4.0])  # norm 5
    c = clip_l2(v, 1.0)
    assert np.linalg.norm(c) == pytest.approx(1.0)
    small = np.array([0.1, 0.1])
    assert np.allclose(clip_l2(small, 1.0), small)  # unchanged below bound


def test_subsampling_amplifies_privacy():
    assert amplify_by_subsampling(2.0, 0.1) < 2.0
    assert amplify_by_subsampling(2.0, 1.0) == pytest.approx(2.0)


def test_dp_sgd_noisy_mean_shape_and_clipping():
    rng = np.random.default_rng(0)
    grads = rng.normal(0, 10, (32, 5))  # large grads to force clipping
    out = dp_sgd_noisy_mean(grads, max_norm=1.0, noise_multiplier=0.0, rng=rng)
    assert out.shape == (5,)
    # with no noise, output is the mean of clipped grads → norm <= max_norm
    assert np.linalg.norm(out) <= 1.0 + 1e-9


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        analytic_gaussian_sigma(0.0, 1e-6)
    with pytest.raises(ValueError):
        RDPAccountant().add_gaussian(0.0)
