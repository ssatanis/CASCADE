"""Differential privacy for FSCP (spec §5 — non-negotiable, MELLODDY's gap).

Provides:
  - gaussian_delta / analytic_gaussian_sigma: the *exact* (ε, δ) relationship for
    the Gaussian mechanism (Balle & Wang 2018). δ is monotone in σ, so we
    calibrate σ to a target (ε, δ) by exact bisection on the closed-form δ.
  - RDPAccountant: Rényi-DP accounting for composing many Gaussian releases
    (Mironov 2017), converted to (ε, δ).
  - clip_l2 / dp_sgd_noisy_mean: the DP-SGD primitives (per-example clipping +
    calibrated Gaussian noise on the summed gradient).
  - amplify_by_subsampling: the classic privacy-amplification-by-subsampling bound.

Everything is checkable: tests assert the calibrated σ reproduces the target δ,
that RDP ε upper-bounds the exact analytic ε, and that composition scales right.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

_DEFAULT_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 65))


def gaussian_delta(epsilon: float, sigma: float, sensitivity: float = 1.0) -> float:
    """Exact δ of the Gaussian mechanism at a given ε (Balle & Wang 2018)."""
    if sigma <= 0:
        return 1.0
    mu = sensitivity / sigma  # = 1 / noise_multiplier
    # δ(ε) = Φ(μ/2 − ε/μ) − e^ε Φ(−μ/2 − ε/μ)
    return float(norm.cdf(mu / 2 - epsilon / mu) - math.exp(epsilon) * norm.cdf(-mu / 2 - epsilon / mu))


def analytic_gaussian_sigma(
    epsilon: float, delta: float, sensitivity: float = 1.0, tol: float = 1e-9
) -> float:
    """Smallest σ achieving (ε, δ)-DP for the Gaussian mechanism, by bisection.

    δ is strictly decreasing in σ, so we bracket then bisect on the exact δ.
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    lo, hi = 1e-6, 1.0
    # expand hi until δ(hi) <= delta
    while gaussian_delta(epsilon, hi, sensitivity) > delta:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("failed to bracket sigma")
    # bisect
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        d = gaussian_delta(epsilon, mid, sensitivity)
        if abs(d - delta) < tol:
            return mid
        if d > delta:
            lo = mid  # too little noise → δ too big → increase σ
        else:
            hi = mid
    return 0.5 * (lo + hi)


class RDPAccountant:
    """Rényi-DP accountant for composed Gaussian mechanisms."""

    def __init__(self, orders: list[float] | None = None):
        self.orders = list(orders) if orders is not None else list(_DEFAULT_ORDERS)
        self._rdp = np.zeros(len(self.orders), dtype=float)

    def add_gaussian(self, noise_multiplier: float, steps: int = 1) -> "RDPAccountant":
        """Compose `steps` Gaussian releases with the given noise multiplier σ/Δ."""
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be > 0")
        nm2 = noise_multiplier**2
        for i, a in enumerate(self.orders):
            self._rdp[i] += steps * a / (2.0 * nm2)
        return self

    def get_epsilon(self, delta: float) -> tuple[float, float]:
        """Convert accumulated RDP to (ε, optimal order) at target δ."""
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        best_eps = math.inf
        best_order = self.orders[0]
        for a, rdp_a in zip(self.orders, self._rdp):
            if a <= 1:
                continue
            eps = rdp_a + math.log(1.0 / delta) / (a - 1.0)
            if eps < best_eps:
                best_eps, best_order = eps, a
        return float(best_eps), float(best_order)


def clip_l2(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale a vector so its L2 norm is at most `max_norm`."""
    vec = np.asarray(vec, dtype=float)
    norm_v = float(np.linalg.norm(vec))
    if norm_v <= max_norm or norm_v == 0:
        return vec
    return vec * (max_norm / norm_v)


def dp_sgd_noisy_mean(
    per_example: np.ndarray,
    max_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One DP-SGD aggregation: clip each row, sum, add Gaussian noise, average.

    The sum's L2 sensitivity to one example is `max_norm`, so noise std =
    noise_multiplier · max_norm. Returns the noisy mean.
    """
    rng = rng or np.random.default_rng()
    X = np.atleast_2d(np.asarray(per_example, dtype=float))
    clipped = np.vstack([clip_l2(row, max_norm) for row in X])
    summed = clipped.sum(axis=0)
    noise = rng.normal(0.0, noise_multiplier * max_norm, size=summed.shape)
    return (summed + noise) / X.shape[0]


def amplify_by_subsampling(epsilon: float, q: float) -> float:
    """Privacy amplification by subsampling: ε' = log(1 + q(e^ε − 1))."""
    if not (0 < q <= 1):
        raise ValueError("q must be in (0, 1]")
    return float(math.log(1.0 + q * (math.exp(epsilon) - 1.0)))
