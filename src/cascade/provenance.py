"""Provenance-quality weighting (CASCADE component C3 multiplier — the moat).

Each screen contributes with weight w_s = (1/σ²) · QC_s · 1[E-dist > τ]. The
inverse-variance core lives in `metaanalysis`; this module computes QC_s — a
scalar in (0, 1) from the wet-lab QC bundle:

    QC_s = σ( a·replicate_r + b·log(coverage) + c·control_separation
              + d·library_complexity − e·representation_skew + bias )

No public dataset and no prior federated-learning system carries wet-lab QC, so
this is the genuinely novel signal. The coefficients default to sensible signs
and can be *learned* to predict held-out replicability (tying C3 to the
Replication Oracle, C5), which is what makes the weight earn its keep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import QCBundle


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def qc_features(qc: QCBundle) -> np.ndarray:
    """Transform a QC bundle into the model feature vector (log on coverage)."""
    return np.array(
        [
            qc.replicate_r,
            np.log(max(qc.coverage, 1.0)),
            qc.control_separation,
            qc.library_complexity,
            qc.representation_skew,
        ],
        dtype=float,
    )


@dataclass
class QCWeightParams:
    """Coefficients for the QC weight. Signs default to domain priors."""

    coef: np.ndarray = None  # type: ignore[assignment]
    bias: float = -1.5

    def __post_init__(self) -> None:
        if self.coef is None:
            # [replicate_r, log(coverage), control_separation, complexity, -skew]
            self.coef = np.array([2.0, 0.5, 2.0, 1.0, -3.0], dtype=float)
        else:
            self.coef = np.asarray(self.coef, dtype=float)
        if self.coef.shape != (5,):
            raise ValueError("coef must have length 5")

    def fit(self, qcs: list[QCBundle], replicated: np.ndarray, l2: float = 1.0) -> "QCWeightParams":
        """Learn coefficients to predict replication (logistic regression).

        Ties the quality weight to the thing it should predict: whether a screen's
        hits actually replicate. Uses scikit-learn for a well-tested optimizer.
        """
        from sklearn.linear_model import LogisticRegression

        X = np.vstack([qc_features(q) for q in qcs])
        y = np.asarray(replicated, dtype=int)
        if len(np.unique(y)) < 2:
            # Degenerate label set — keep priors rather than overfit.
            return self
        # Standardize so coefficients are comparable; fold scaling back in.
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        clf = LogisticRegression(C=1.0 / l2, max_iter=1000)
        clf.fit(Xs, y)
        # Map standardized coefs back to raw-feature space.
        self.coef = clf.coef_[0] / sd
        self.bias = float(clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum())
        return self


def quality_weight(qc: QCBundle, params: QCWeightParams | None = None) -> float:
    """Map a full QC bundle to a quality weight in (0, 1)."""
    p = params or QCWeightParams()
    z = float(p.coef @ qc_features(qc) + p.bias)
    return float(_sigmoid(z))


def effective_quality(qc: QCBundle, params: QCWeightParams | None = None) -> float:
    """Quality weight robust to real-data carriers.

    Real screens (from a gene-effect matrix) expose only `control_separation` as a
    real, comparative quality scalar in (0,1); the guide-level QC fields are NaN
    ("not measured"). When any guide-level field is NaN we use control_separation
    directly; otherwise the full learned `quality_weight`. Never fabricates.
    """
    fields = [qc.replicate_r, qc.coverage, qc.library_complexity, qc.representation_skew]
    if any(np.isnan(x) for x in fields):
        cs = float(qc.control_separation)
        return float(min(max(cs, 0.0), 1.0))
    return quality_weight(qc, params)
