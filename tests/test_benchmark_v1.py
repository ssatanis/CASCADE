"""Tests for the frozen Replication Benchmark v1."""

import csv
import json

import pytest

from cascade.benchmark_v1 import BENCH_DIR, score_predictions

pytestmark = pytest.mark.skipif(
    not (BENCH_DIR / "test_pairs.csv").exists() or not (BENCH_DIR / "leaderboard.json").exists(),
    reason="benchmark not frozen (run `cascade freeze-benchmark`)",
)


def test_manifest_and_leakage():
    m = json.loads((BENCH_DIR / "manifest.json").read_text())
    assert m["leakage_check"]["train_test_cellline_disjoint_on_holdout"] is True
    assert m["leakage_check"]["train_test_gene_disjoint_on_holdout"] is True
    assert m["n_test"] > 1000
    assert set(m["data_releases"]) >= {"broad", "sanger", "orcs", "replogle"}
    assert m["provenance_hash"]


def test_harness_reproduces_cascade_entry():
    preds = list(csv.DictReader(open(BENCH_DIR / "predictions_cascade.csv")))
    scored = score_predictions(BENCH_DIR, preds)
    leaderboard = json.loads((BENCH_DIR / "leaderboard.json").read_text())
    assert scored["overall"]["auroc"] == pytest.approx(leaderboard["CASCADE_v0"]["overall"]["auroc"], abs=1e-6)
    # CASCADE beats the ridge + mean baselines overall
    assert scored["overall"]["auroc"] > leaderboard["baseline_ridge"]["overall"]["auroc"]
    assert leaderboard["baseline_mean_rate"]["overall"]["auroc"] == 0.5


def test_strata_present():
    lb = json.loads((BENCH_DIR / "leaderboard.json").read_text())
    strata = lb["CASCADE_v0"]["by_stratum"]
    assert "cross_lab" in strata and "cross_cell_type" in strata and "cross_study" in strata
