import numpy as np

from cascade.provenance import QCWeightParams, quality_weight
from cascade.types import QCBundle


def good_qc():
    return QCBundle(replicate_r=0.9, coverage=800, control_separation=0.95, library_complexity=0.95, representation_skew=0.05)


def bad_qc():
    return QCBundle(replicate_r=0.2, coverage=40, control_separation=0.55, library_complexity=0.3, representation_skew=0.6)


def test_weight_in_unit_interval():
    for qc in (good_qc(), bad_qc()):
        w = quality_weight(qc)
        assert 0.0 < w < 1.0


def test_good_qc_outweighs_bad():
    assert quality_weight(good_qc()) > quality_weight(bad_qc())


def test_monotonic_in_replicate_r():
    base = dict(coverage=500, control_separation=0.9, library_complexity=0.9, representation_skew=0.1)
    lo = quality_weight(QCBundle(replicate_r=0.3, **base))
    hi = quality_weight(QCBundle(replicate_r=0.95, **base))
    assert hi > lo


def test_fit_learns_to_predict_replication():
    rng = np.random.default_rng(0)
    qcs, labels = [], []
    for _ in range(400):
        tier = rng.uniform(0.2, 1.0)
        qc = QCBundle(
            replicate_r=float(np.clip(0.5 + 0.45 * tier + rng.normal(0, 0.03), 0, 1)),
            coverage=float(80 + 900 * tier),
            control_separation=float(np.clip(0.6 + 0.38 * tier, 0, 1)),
            library_complexity=float(np.clip(0.5 + 0.45 * tier, 0, 1)),
            representation_skew=float(np.clip(0.5 * (1 - tier), 0, 1)),
        )
        # higher tier → more likely to replicate
        replicated = rng.uniform() < tier
        qcs.append(qc)
        labels.append(replicated)
    params = QCWeightParams().fit(qcs, np.array(labels))
    # learned weights should separate replicating from non-replicating screens
    w_rep = np.mean([quality_weight(q, params) for q, l in zip(qcs, labels) if l])
    w_non = np.mean([quality_weight(q, params) for q, l in zip(qcs, labels) if not l])
    assert w_rep > w_non
