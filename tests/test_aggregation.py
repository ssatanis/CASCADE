import numpy as np

from cascade.federated.aggregation import FederatedMetaAnalysis, GeneStatistic
from cascade.provenance import QCWeightParams
from fixtures.synthetic_screens import SyntheticConfig, generate_synthetic_cohort
from cascade.types import Context, GeneEffect, QCBundle, ScreenResult


def _screen(sid, beta, var, qc_tier=0.9, edist=0.5, gene="G", lineage="lung"):
    qc = QCBundle(0.5 + 0.4 * qc_tier, 80 + 900 * qc_tier, 0.6 + 0.38 * qc_tier, 0.5 + 0.45 * qc_tier, 0.5 * (1 - qc_tier))
    return ScreenResult(
        screen_id=sid,
        lab_id=sid,
        context=Context(cell_line="CL", lineage=lineage),
        modality="KO",
        qc=qc,
        effects={gene: GeneEffect(gene=gene, beta=beta, variance=var)},
        pos_control_edistance=edist,
    )


def test_aggregate_pan_gene_strong_negative():
    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    fma = FederatedMetaAnalysis(qc_params=QCWeightParams())
    m = fma.aggregate_gene(cohort.screens, "PAN000")
    assert m is not None
    assert m.effect < -0.5  # pan-essential depletes strongly
    gs = GeneStatistic.from_meta("PAN000", m)
    assert gs.k == m.k and gs.effect == m.effect


def test_validity_gate_excludes_failed_screens():
    screens = [
        _screen("good1", -1.0, 0.04, edist=0.5),
        _screen("good2", -1.1, 0.04, edist=0.5),
        _screen("bad", 5.0, 0.04, edist=0.0),  # failed positive controls → excluded
    ]
    fma = FederatedMetaAnalysis(validity_edist=0.05)
    betas, variances, quality = fma.collect(screens, "G")
    assert len(betas) == 2  # the bad screen is gated out
    m = fma.aggregate_gene(screens, "G")
    assert m.effect < 0  # not dragged positive by the excluded outlier


def test_private_path_recovers_plain_estimate_within_noise():
    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    fma = FederatedMetaAnalysis(qc_params=QCWeightParams())
    plain = fma.aggregate_gene(cohort.screens, "PAN000")
    # large epsilon → little noise → private ~ plain
    private = fma.private_aggregate_gene(cohort.screens, "PAN000", epsilon=50.0, delta=1e-5, seed=0)
    assert private["n_clients"] >= 2
    assert abs(private["effect"] - plain.effect) < 0.2


def test_private_path_uses_secure_aggregation_and_dp():
    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    fma = FederatedMetaAnalysis(qc_params=QCWeightParams())
    out = fma.private_aggregate_gene(cohort.screens, "PAN000", epsilon=3.0, delta=1e-6, seed=1)
    assert out["sigma"] > 0
    assert out["epsilon"] == 3.0
    assert np.isfinite(out["effect"])
