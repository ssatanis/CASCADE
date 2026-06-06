"""FSCP 2-party federation tests — the hard privacy assertion + MELLODDY proof."""

import numpy as np
import pytest

from cascade.corpus import RAW
from cascade.fscp_pilot import EgressLog, _local_gradient, federated_train

pytestmark = pytest.mark.skipif(
    not (RAW / "CRISPRGeneEffect.csv").exists() or not (RAW / "gene_effect.csv").exists(),
    reason="real Broad/Sanger data not downloaded",
)


def test_egress_log_flags_only_masked_dp():
    log = EgressLog()
    log.record(0, "A", 18)
    assert log.any_raw_crossed() is False
    # a malformed entry that leaks raw data would be caught
    log.rounds.append({"contains_raw_effect": True, "contains_label": False, "contains_per_gene_summary": False})
    assert log.any_raw_crossed() is True


def test_secure_agg_dp_no_raw_crosses_and_budget():
    rng = np.random.default_rng(0)
    Xa, ya = rng.normal(size=(400, 12)), rng.integers(0, 2, 400).astype(float)
    Xb, yb = rng.normal(size=(400, 12)), rng.integers(0, 2, 400).astype(float)
    w, b, egress, acct, nm = federated_train(Xa, ya, Xb, yb, dim=12, epsilon=4.0, delta=1e-6, rounds=30, seed=0)
    # only masked DP gradients crossed the boundary
    assert egress.any_raw_crossed() is False
    assert all(r["payload"] == "masked_dp_gradient" for r in egress.rounds)
    # privacy budget is bounded (target met within composition)
    eps, _ = acct.get_epsilon(1e-6)
    assert eps <= 4.0 + 1e-6
    assert nm > 0


def test_clipped_gradient_bounded():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 10, (50, 5))
    y = rng.integers(0, 2, 50).astype(float)
    g = _local_gradient(np.zeros(5), 0.0, X, y, clip_norm=1.0)
    # sum of 50 unit-clipped per-example grads -> norm <= 50
    assert np.linalg.norm(g) <= 50 + 1e-6


@pytest.mark.skipif(not (RAW / "Model.csv").exists(), reason="Model.csv needed")
def test_federated_beats_each_party_alone():
    from cascade.fscp_pilot import federated_beats_alone, run_pilot

    r = run_pilot(epsilon=4.0, delta=1e-6, seed=0)
    assert r["privacy"]["raw_data_crossed_boundary"] is False
    assert r["privacy"]["rdp_epsilon_total_spent"] <= 5.0  # ε in the target 1–5 range
    assert federated_beats_alone(r)  # federated >= each party alone
    a = r["auroc"]
    assert a["federated_AB"] <= a["centralized_AB"] + 0.01  # federated ≤ centralized (DP cost)
