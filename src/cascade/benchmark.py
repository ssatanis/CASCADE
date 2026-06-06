"""The replication-prediction benchmark (CASCADE — the eval SplicR should own).

There is no standard benchmark for cross-lab CRISPR hit replication. This module
defines a generic one: split labeled replication pairs, fit the Oracle, and report
AUROC, calibration (ECE), conformal coverage, abstention rate, and an
uncalibrated-logistic comparison.

The SHIPPED v0 evaluation runs on REAL data with a context-holdout split (see
`cascade.train.evaluate`); this generic helper is used there and by the
ground-truth property tests on the seeded fixture. No synthetic data is imported
here — the caller supplies the pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .oracle import ReplicationOracle, expected_calibration_error
from .provenance import QCWeightParams
from .types import ReplicationPair


@dataclass
class BenchmarkReport:
    n_pairs: int
    base_rate: float
    auroc: float
    ece: float
    coverage: float
    coverage_target: float
    abstention_rate: float
    auroc_uncalibrated_logistic: float
    ece_uncalibrated_logistic: float
    beats_mean_rate: bool

    def summary(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "base_rate": round(self.base_rate, 4),
            "auroc": round(self.auroc, 4),
            "ece": round(self.ece, 4),
            "coverage": round(self.coverage, 4),
            "coverage_target": self.coverage_target,
            "abstention_rate": round(self.abstention_rate, 4),
            "auroc_uncalibrated_logistic": round(self.auroc_uncalibrated_logistic, 4),
            "ece_uncalibrated_logistic": round(self.ece_uncalibrated_logistic, 4),
            "beats_mean_rate": self.beats_mean_rate,
        }


def run_replication_benchmark(
    pairs: list[ReplicationPair],
    alpha: float = 0.1,
    test_frac: float = 0.3,
    qc_params: QCWeightParams | None = None,
    seed: int = 0,
) -> BenchmarkReport:
    """Generic benchmark on a (random) split of supplied pairs.

    NOTE: the shipped v0 evaluation uses a CONTEXT-HOLDOUT split on REAL data
    (`cascade.train.evaluate`), not this random split. This helper is for the
    property tests and quick checks.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from .features import featurize

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_test = int(round(test_frac * len(pairs)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train = [pairs[i] for i in train_idx]
    test = [pairs[i] for i in test_idx]

    oracle = ReplicationOracle(alpha=alpha).fit(train, qc_params=qc_params, seed=seed)
    scored = oracle.score(test)

    Xtr, ytr = featurize(train, qc_params)
    Xte, yte = featurize(test, qc_params)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, ytr)
    p_uncal = clf.predict_proba((Xte - mu) / sd)[:, 1]
    auroc_uncal = float(roc_auc_score(yte, p_uncal)) if len(np.unique(yte)) >= 2 else float("nan")
    ece_uncal = expected_calibration_error(yte, p_uncal)

    base_rate = float(np.mean([1 if p.label else 0 for p in pairs]))
    return BenchmarkReport(
        n_pairs=len(pairs),
        base_rate=base_rate,
        auroc=scored["auroc"],
        ece=scored["ece"],
        coverage=scored["coverage"],
        coverage_target=1 - alpha,
        abstention_rate=scored["abstention_rate"],
        auroc_uncalibrated_logistic=auroc_uncal,
        ece_uncalibrated_logistic=ece_uncal,
        beats_mean_rate=bool(scored["auroc"] > 0.5),
    )
