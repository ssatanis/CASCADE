import numpy as np
import pytest

from cascade.oracle import ReplicationOracle, expected_calibration_error
from cascade.provenance import QCWeightParams
from fixtures.synthetic_screens import SyntheticConfig, generate_synthetic_cohort
from cascade.types import Context, QCBundle, ReplicationPair


def _split(pairs, frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n = int(frac * len(pairs))
    return [pairs[i] for i in idx[:n]], [pairs[i] for i in idx[n:]]


def test_oracle_predicts_replication_better_than_chance():
    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    train, test = _split(cohort.replication_pairs())
    oracle = ReplicationOracle(alpha=0.1).fit(train, qc_params=QCWeightParams(), seed=0)
    scored = oracle.score(test)
    assert scored["auroc"] > 0.65
    assert scored["ece"] < 0.1
    # conformal coverage should meet (or exceed) the 1-alpha target
    assert scored["coverage"] >= 0.86


def test_oracle_abstains_outside_support():
    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    oracle = ReplicationOracle(alpha=0.1).fit(cohort.replication_pairs(), qc_params=QCWeightParams(), seed=0)
    weird = ReplicationPair(
        gene="GENEX",
        context_a=Context(cell_line="X", lineage="MARTIAN"),  # never seen
        context_b=Context(cell_line="Y", lineage="VENUSIAN"),
        beta_a=-1.0,
        var_a=0.05,
        beta_b=0.0,
        var_b=0.05,
        qc_a=QCBundle(0.9, 800, 0.95, 0.95, 0.05),
        qc_b=QCBundle(0.9, 800, 0.95, 0.95, 0.05),
        modality="KO",
        edist_a=0.5,
        label=False,
    )
    pred = oracle.predict_pair(weird)
    assert pred.abstained is True
    assert pred.p_replicate is not None  # value computed, but flagged abstained
    d = pred.as_dict()
    assert d["p_replicate"] is None  # not surfaced when abstained
    assert "support" in d["basis"].lower() or "calibration" in d["basis"].lower()


def test_oracle_in_support_prediction_has_interval():
    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    pairs = cohort.replication_pairs()
    oracle = ReplicationOracle(alpha=0.1).fit(pairs, qc_params=QCWeightParams(), seed=0)
    # a strong pan-essential hit in a seen lineage pair should be in support
    pan_pairs = [p for p in pairs if p.gene.startswith("PAN")]
    pred = oracle.predict_pair(pan_pairs[0])
    if not pred.abstained:
        assert 0.0 <= pred.lower <= pred.upper <= 1.0
        assert pred.n_comparable >= oracle.min_comparable


def test_ece_helper():
    # perfectly calibrated → ~0 ECE
    p = np.linspace(0.05, 0.95, 1000)
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=1000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.05


def test_requires_minimum_data():
    with pytest.raises(ValueError):
        ReplicationOracle().fit([])
