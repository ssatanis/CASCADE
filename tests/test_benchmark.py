from cascade.benchmark import run_replication_benchmark
from fixtures.synthetic_screens import meta_analysis_benefit
from cascade.provenance import QCWeightParams
from fixtures.synthetic_screens import SyntheticConfig, generate_synthetic_cohort


def test_replication_benchmark_metrics():
    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    pairs = cohort.replication_pairs()
    report = run_replication_benchmark(pairs, qc_params=QCWeightParams(), seed=0)
    assert report.auroc > 0.65
    assert report.beats_mean_rate is True
    assert report.coverage >= report.coverage_target - 0.04
    assert report.ece < 0.1
    # calibration should not be worse than the uncalibrated logistic
    assert report.ece <= report.ece_uncalibrated_logistic + 0.02
    assert 0.0 <= report.abstention_rate <= 0.5


def test_meta_analysis_benefit_provable_win():
    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    benefit = meta_analysis_benefit(cohort, QCWeightParams())
    assert benefit["pooled_beats_single"] is True
    assert benefit["pooled_beats_uniform"] is True
    assert benefit["pooled_mse"] < benefit["single_screen_mse"]
    assert benefit["n_estimands"] > 0
