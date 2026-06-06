"""Conformal calibration (CASCADE component C4).

Distribution-free coverage by construction — no assumptions on the model. Two
tools:

  - MondrianConformalRegressor: split-conformal intervals stratified by group
    (e.g. lineage × modality), so the 1−α coverage guarantee holds *within* each
    context, not just marginally. This is what stops a model from looking
    calibrated overall while being miscalibrated for rare lineages.
  - AdaptiveConformal (ACI, Gibbs & Candès 2021): online α adjustment so coverage
    holds under distribution drift — the streaming/federated setting.

Both are mechanically simple and verifiable: the tests assert empirical coverage
≈ 1−α on held-out data and within each group.
"""

from __future__ import annotations

import math

import numpy as np


class MondrianConformalRegressor:
    """Split-conformal regression with per-group nonconformity quantiles."""

    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.group_scores: dict[object, np.ndarray] = {}
        self.global_scores: np.ndarray | None = None

    def fit(self, y, yhat, groups) -> "MondrianConformalRegressor":
        y = np.asarray(y, dtype=float)
        yhat = np.asarray(yhat, dtype=float)
        groups = list(groups)
        if not (len(y) == len(yhat) == len(groups)):
            raise ValueError("y, yhat, groups must be equal length")
        if len(y) == 0:
            raise ValueError("need calibration data")
        resid = np.abs(y - yhat)
        self.global_scores = np.sort(resid)
        by_group: dict[object, list[float]] = {}
        for g, r in zip(groups, resid):
            by_group.setdefault(g, []).append(float(r))
        self.group_scores = {g: np.sort(np.array(v)) for g, v in by_group.items()}
        return self

    @staticmethod
    def _quantile(scores: np.ndarray, alpha: float) -> float:
        """Conformal (1−α) quantile: the ceil((n+1)(1−α))-th smallest score."""
        n = len(scores)
        rank = math.ceil((n + 1) * (1 - alpha))
        if rank > n:
            return math.inf
        return float(scores[rank - 1])

    def quantile(self, group, alpha: float | None = None) -> tuple[float, bool]:
        """Return (half-width, used_global)."""
        a = self.alpha if alpha is None else alpha
        if self.global_scores is None:
            raise RuntimeError("call fit() first")
        scores = self.group_scores.get(group)
        used_global = False
        if scores is None:
            scores, used_global = self.global_scores, True
        else:
            q = self._quantile(scores, a)
            if math.isinf(q):
                scores, used_global = self.global_scores, True
        q = self._quantile(scores, a)
        return q, used_global

    def predict_interval(self, yhat: float, group, alpha: float | None = None):
        """Return (lower, upper, used_global) for a point prediction."""
        q, used_global = self.quantile(group, alpha)
        return yhat - q, yhat + q, used_global

    def has_support(self, group, alpha: float | None = None, min_size: int | None = None) -> bool:
        """Is there enough in-group calibration data for a finite, useful interval?"""
        a = self.alpha if alpha is None else alpha
        scores = self.group_scores.get(group)
        if scores is None:
            return False
        need = math.ceil((1 - a) / a) if min_size is None else min_size
        return len(scores) >= need and not math.isinf(self._quantile(scores, a))


class AdaptiveConformal:
    """Adaptive Conformal Inference — online α adaptation under drift."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = alpha
        self._covered_history: list[int] = []

    def effective_alpha(self) -> float:
        return float(min(max(self.alpha_t, 0.0), 1.0))

    def update(self, covered: bool) -> float:
        """ACI update: α_{t+1} = α_t + γ(α − err_t), err_t = 1 if not covered."""
        err = 0 if covered else 1
        self._covered_history.append(1 - err)
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err)
        self.alpha_t = float(min(max(self.alpha_t, 0.0), 1.0))
        return self.alpha_t

    def realized_coverage(self) -> float:
        if not self._covered_history:
            return float("nan")
        return float(np.mean(self._covered_history))
