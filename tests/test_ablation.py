"""Ablation result-shape + invariant tests (checks committed JSON, no recompute)."""
import json
from pathlib import Path
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _load():
    p = RESULTS / "ablation.json"
    if not p.exists():
        pytest.skip("ablation.json not generated")
    return json.loads(p.read_text())


def test_ablation_shape():
    r = _load()
    for v in ("A_logistic_only", "B_plus_isotonic", "C_plus_conformal",
              "D_plus_provenance", "E_full_cascade"):
        assert v in r["variants"], f"missing variant {v}"
    assert r["trained_on_real_data"] is True


def test_E_matches_oracle_v0():
    """Variant E must reproduce oracle_v0.pkl held-out AUROC within 0.002."""
    r = _load()
    e = r["variants"]["E_full_cascade"]["auroc_overall"]
    bench = json.loads((RESULTS / "benchmark_real.json").read_text())
    oracle = bench["evaluation"]["oracle"]["auroc"]
    assert abs(e - oracle) <= 0.002, f"E {e} vs oracle_v0 {oracle}"


def test_auroc_non_decreasing_A_to_E():
    """A<=B<=C<=D<=E (isotonic/conformal are monotone → ties allowed; tol for clip)."""
    r = _load()
    order = ["A_logistic_only", "B_plus_isotonic", "C_plus_conformal",
             "D_plus_provenance", "E_full_cascade"]
    aurocs = [r["variants"][v]["auroc_overall"] for v in order]
    for i in range(1, len(aurocs)):
        assert aurocs[i] >= aurocs[i - 1] - 0.003, f"{order[i]} {aurocs[i]} < {order[i-1]} {aurocs[i-1]}"


def test_calibration_improves_with_isotonic():
    r = _load()
    assert r["variants"]["B_plus_isotonic"]["ece"] <= r["variants"]["A_logistic_only"]["ece"]
