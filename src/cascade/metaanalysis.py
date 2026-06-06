"""Inverse-variance meta-analysis (CASCADE component C3 core).

This is the provably-correct heart of the system. Pooling per-gene effect sizes
across labs with inverse-variance weights is the minimum-variance unbiased linear
combiner (Gauss-Markov), so it is *by construction* at least as good as uniform
averaging or any single screen. DerSimonian-Laird random effects add a
between-lab heterogeneity term τ², which is exactly the cross-lab variability the
Replication Oracle later predicts.

An optional `quality` weight (from `provenance.quality_weight`) multiplies the
inverse-variance weights — the novel provenance signal. The pooled-variance
formula uses the general sandwich form so it stays correct for arbitrary weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Z_975 = 1.959963984540054  # Φ^{-1}(0.975)


@dataclass
class MetaResult:
    effect: float
    se: float
    variance: float
    tau2: float
    q: float
    i2: float
    k: int
    weights: np.ndarray
    ci_low: float
    ci_high: float

    @property
    def z(self) -> float:
        return self.effect / self.se if self.se > 0 else 0.0


def _validate(betas: np.ndarray, variances: np.ndarray, quality: np.ndarray) -> None:
    if betas.shape != variances.shape or betas.shape != quality.shape:
        raise ValueError("betas, variances, quality must have the same shape")
    if betas.size == 0:
        raise ValueError("need at least one study")
    if np.any(variances <= 0):
        raise ValueError("all variances must be > 0")
    if np.any(quality < 0):
        raise ValueError("quality weights must be >= 0")


def _pool(betas: np.ndarray, eff_var: np.ndarray, quality: np.ndarray):
    """Weighted pooled estimate with general-form variance.

    weights w_i = quality_i / eff_var_i (eff_var = σ² for FE, σ²+τ² for RE).
    Var(pooled) = Σ w_i² eff_var_i / (Σ w_i)²  (exact for arbitrary weights;
    reduces to 1/Σ(1/eff_var) when quality ≡ 1).
    """
    w = quality / eff_var
    sw = w.sum()
    if sw <= 0:
        raise ValueError("total weight is zero (all quality weights vanished)")
    effect = float((w * betas).sum() / sw)
    var = float((w**2 * eff_var).sum() / sw**2)
    return effect, var, w


def fixed_effect(betas, variances, quality=None) -> MetaResult:
    """Fixed-effect inverse-variance meta-analysis."""
    betas = np.asarray(betas, dtype=float)
    variances = np.asarray(variances, dtype=float)
    quality = np.ones_like(betas) if quality is None else np.asarray(quality, dtype=float)
    _validate(betas, variances, quality)

    k = betas.size
    effect, var, w = _pool(betas, variances, quality)

    # Cochran's Q with inverse-variance weights (independent of the quality term).
    wi = 1.0 / variances
    mu_iv = float((wi * betas).sum() / wi.sum())
    q = float((wi * (betas - mu_iv) ** 2).sum())
    df = max(k - 1, 0)
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0

    se = float(np.sqrt(var))
    return MetaResult(
        effect=effect,
        se=se,
        variance=var,
        tau2=0.0,
        q=q,
        i2=i2,
        k=k,
        weights=w,
        ci_low=effect - Z_975 * se,
        ci_high=effect + Z_975 * se,
    )


def _dersimonian_laird_tau2(betas: np.ndarray, variances: np.ndarray) -> tuple[float, float]:
    """DerSimonian-Laird between-study variance τ² and Cochran's Q."""
    wi = 1.0 / variances
    sw = wi.sum()
    mu = (wi * betas).sum() / sw
    q = float((wi * (betas - mu) ** 2).sum())
    k = betas.size
    df = k - 1
    if df <= 0:
        return 0.0, q
    c = sw - (wi**2).sum() / sw
    tau2 = (q - df) / c if c > 0 else 0.0
    return max(0.0, float(tau2)), q


def random_effects(betas, variances, quality=None) -> MetaResult:
    """DerSimonian-Laird random-effects meta-analysis (optionally quality-weighted)."""
    betas = np.asarray(betas, dtype=float)
    variances = np.asarray(variances, dtype=float)
    quality = np.ones_like(betas) if quality is None else np.asarray(quality, dtype=float)
    _validate(betas, variances, quality)

    k = betas.size
    tau2, q = _dersimonian_laird_tau2(betas, variances)
    eff_var = variances + tau2
    effect, var, w = _pool(betas, eff_var, quality)

    df = max(k - 1, 0)
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    se = float(np.sqrt(var))
    return MetaResult(
        effect=effect,
        se=se,
        variance=var,
        tau2=tau2,
        q=q,
        i2=i2,
        k=k,
        weights=w,
        ci_low=effect - Z_975 * se,
        ci_high=effect + Z_975 * se,
    )
