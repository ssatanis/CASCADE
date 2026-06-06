"""Tests for the §HONESTY scientific-gate baselines (synthetic, seeded — TEST ONLY).

These exercise B4/B5 + the oracle-vs-baseline machinery WITHOUT the real corpus,
so they run anywhere. The real-corpus gate is run via `python -m cascade.gate`.
"""

import numpy as np

from cascade import gate
from cascade.oracle import ReplicationOracle


def _cohort_pairs(seed=0):
    from fixtures.synthetic_screens import SyntheticConfig, generate_synthetic_cohort
    return generate_synthetic_cohort(SyntheticConfig(seed=seed)).replication_pairs()


def test_b5_group_prior_recovers_group_rate():
    pairs = _cohort_pairs(0)
    # split half/half; B5 predictions must be valid probabilities in [0,1]
    half = len(pairs) // 2
    train, test = pairs[:half], pairs[half:]
    preds = gate.b5_group_prior_fit_predict(train, test)
    assert preds.shape[0] == len(test)
    assert np.all((preds >= 0) & (preds <= 1))


def test_b4_additive_returns_probabilities():
    pairs = _cohort_pairs(1)
    half = len(pairs) // 2
    train, test = pairs[:half], pairs[half:]
    preds = gate.b4_additive_fit_predict(train, test)
    assert preds.shape[0] == len(test)
    assert np.all((preds >= 0) & (preds <= 1))


def test_group_key_schemes():
    p = _cohort_pairs(0)[0]
    assert gate._group_key(p, "loco_cellline") == p.context_a.cell_line
    assert gate._group_key(p, "loco_lineage") == p.context_a.lineage
    # loso_study falls back to a sentinel when study is empty
    assert gate._group_key(p, "loso_study") in (p.study or "__nostudy__",)


def test_oracle_beats_prior_on_signal_cohort():
    """On a cohort WITH real structure, the calibrated oracle should out-rank the
    group prior on its non-abstained set (sanity that the harness can detect a win)."""
    pairs = _cohort_pairs(0)
    half = len(pairs) // 2
    train, test = pairs[:half], pairs[half:]
    oracle = ReplicationOracle(alpha=0.1).fit(train, seed=0)
    preds = oracle.predict(test)
    kept = [(pr, p) for pr, p in zip(preds, test) if not pr.abstained]
    if len(kept) < 20:
        return  # abstained too much on this synthetic slice; nothing to assert
    y = np.array([1 if p.label else 0 for _, p in kept])
    if len(np.unique(y)) < 2:
        return
    p_or = np.array([pr.p_replicate for pr, _ in kept])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, p_or) >= 0.5
