import numpy as np

from cascade.provenance import quality_weight
from fixtures.synthetic_screens import SyntheticConfig, generate_synthetic_cohort


def test_reproduces_dlfc_collapse():
    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    r_raw, r_dlfc = cohort.raw_vs_dlfc_correlation()
    # The documented Broad↔Sanger collapse: raw ~0.81, dLFC ~0.47.
    assert r_raw > r_dlfc + 0.2, "raw correlation must exceed deviation correlation"
    assert 0.7 <= r_raw <= 0.9
    assert 0.35 <= r_dlfc <= 0.6


def test_pan_essentials_replicate_more_than_cross_lineage_selective():
    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    pairs = cohort.replication_pairs()
    pan = [p for p in pairs if cohort.gene_type[p.gene] == "pan"]
    # selective hits compared in a DIFFERENT lineage than where they're selective
    sel_cross = [
        p
        for p in pairs
        if cohort.gene_type[p.gene] == "selective"
        and p.context_b.lineage != cohort.gene_pref_lineage[p.gene]
    ]
    pan_rate = np.mean([p.label for p in pan])
    sel_rate = np.mean([p.label for p in sel_cross])
    assert pan_rate > sel_rate + 0.2


def test_qc_predicts_replication():
    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    pairs = cohort.replication_pairs()
    q = np.array([quality_weight(p.qc_a) for p in pairs])
    y = np.array([1.0 if p.label else 0.0 for p in pairs])
    hi = y[q >= np.median(q)].mean()
    lo = y[q < np.median(q)].mean()
    assert hi > lo  # higher-QC source screens replicate more


def test_cohort_shape():
    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    assert len(cohort.screens) == 8 * 4  # n_labs × lineages
    pairs = cohort.replication_pairs(max_pairs=500)
    assert len(pairs) <= 500
    assert all(isinstance(p.label, bool) for p in pairs)
