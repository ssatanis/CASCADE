"""Baselines + the always-beat-baseline gate (CASCADE component C4).

The load-bearing result of Ahlmann-Eltze, Huber & Anders (Nature Methods 2025):
deep perturbation models routinely fail to beat the mean or a linear model. So
CASCADE reports every prediction against those baselines and promotes a model
only if it beats BOTH on a lab-holdout split. We never pretend the architecture
is the moat; if ridge wins, ship ridge and route value to C3/C4/C5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def mean_baseline(y_train, n_test: int) -> np.ndarray:
    """Constant predictor: the training mean (the trivial baseline)."""
    return np.full(int(n_test), float(np.mean(y_train)), dtype=float)


def ridge_baseline(X_train, y_train, X_test, alpha: float = 1.0) -> np.ndarray:
    """Linear ridge baseline — the bar deep models keep failing to clear."""
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=alpha)
    model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float))
    return model.predict(np.asarray(X_test, dtype=float))


def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


@dataclass
class BaselineGate:
    model_mse: float
    mean_mse: float
    ridge_mse: float
    beats_mean: bool
    beats_ridge: bool
    promote: bool
    target_variance: float
    low_variance_warning: bool


def beats_baseline_gate(
    y_true,
    model_pred,
    mean_pred,
    ridge_pred,
    rel_margin: float = 0.0,
    low_var_threshold: float = 1e-3,
) -> BaselineGate:
    """Promote the model only if it beats mean AND ridge by `rel_margin`.

    `rel_margin` is a fractional improvement requirement (e.g. 0.02 = must be ≥2%
    better). A low-variance target is flagged because it makes any model look like
    the mean (the spec's variance-aware caveat).
    """
    y_true = np.asarray(y_true, dtype=float)
    m_model = mse(y_true, model_pred)
    m_mean = mse(y_true, mean_pred)
    m_ridge = mse(y_true, ridge_pred)
    var = float(np.var(y_true))

    beats_mean = m_model <= m_mean * (1 - rel_margin)
    beats_ridge = m_model <= m_ridge * (1 - rel_margin)
    return BaselineGate(
        model_mse=m_model,
        mean_mse=m_mean,
        ridge_mse=m_ridge,
        beats_mean=beats_mean,
        beats_ridge=beats_ridge,
        promote=bool(beats_mean and beats_ridge),
        target_variance=var,
        low_variance_warning=var < low_var_threshold,
    )
