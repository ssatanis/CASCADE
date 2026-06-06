"""WBC + MAIC baseline result-shape tests (do NOT recompute in CI — check the
committed JSONs and the honest comparison vs CASCADE)."""

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _load(name):
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} not generated (run the baseline first)")
    return json.loads(p.read_text())


def _cascade_cross_lab():
    bench = _load("benchmark_real.json")
    return bench["evaluation"]["stratified_by_pair_type"]["cross_lab"]["auroc"]


def test_wbc_json_shape():
    r = _load("wbc_baseline.json")
    for k in ("method", "reference", "n_genes_scored", "auroc_cross_lab", "auroc_overall",
              "trained_on_real_data", "provenance_hash"):
        assert k in r, f"missing {k}"
    assert r["trained_on_real_data"] is True
    assert r["n_genes_scored"] > 1000
    assert "37201508" in r["reference"]  # Billmann PMID


def test_cascade_beats_wbc_on_cross_lab():
    """Acceptance criterion: CASCADE AUROC > WBC AUROC on cross_lab (else fail loudly)."""
    wbc = _load("wbc_baseline.json")["auroc_cross_lab"]
    cascade = _cascade_cross_lab()
    assert cascade > wbc, f"CASCADE cross_lab {cascade} must exceed WBC {wbc}"


def test_maic_json_shape():
    r = _load("maic_baseline.json")
    for k in ("method", "reference", "auroc_cross_lab", "auroc_overall", "trained_on_real_data"):
        assert k in r
    assert r["trained_on_real_data"] is True
    # MAIC_approx is documented when the package is unavailable
    assert "MAIC" in r["method"]


def test_leaderboard_has_baselines():
    lb = _load("../benchmark/replication_benchmark_v1/leaderboard.json") if (
        RESULTS.parent / "benchmark/replication_benchmark_v1/leaderboard.json").exists() else None
    lb_path = RESULTS.parent / "benchmark" / "replication_benchmark_v1" / "leaderboard.json"
    if not lb_path.exists():
        pytest.skip("leaderboard.json absent")
    lb = json.loads(lb_path.read_text())
    keys = " ".join(lb.keys())
    assert "WBC" in keys, "WBC row missing from leaderboard"
    assert "MAIC" in keys, "MAIC row missing from leaderboard"
