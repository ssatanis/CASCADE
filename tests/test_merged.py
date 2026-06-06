"""Tests for the merged real corpus (Broad↔Sanger + ORCS + Replogle)."""

import pytest

from cascade.corpus import RAW
from cascade.features import FEATURE_NAMES, featurize
from cascade.types import Context, ReplicationPair, QCBundle

NAN = float("nan")


def _pair(source, task, pair_type):
    qc = QCBundle(NAN, NAN, 0.8, NAN, NAN)
    return ReplicationPair(
        gene="G", context_a=Context("A", "lin"), context_b=Context("B", "lin2"),
        beta_a=-1.0, var_a=0.05, beta_b=-0.9, var_b=0.05, qc_a=qc, qc_b=qc,
        modality="KO", edist_a=0.5, label=True, quality_a=0.8, quality_b=0.8,
        source=source, task=task, pair_type=pair_type,
    )


def test_feature_vector_includes_source_and_task():
    assert "src_orcs" in FEATURE_NAMES
    assert "src_replogle" in FEATURE_NAMES
    assert "task_transcriptomic" in FEATURE_NAMES
    X, _ = featurize([
        _pair("depmap_sanger", "fitness", "cross_lab"),
        _pair("orcs", "fitness", "cross_study"),
        _pair("replogle", "transcriptomic", "cross_cell_type"),
    ])
    assert X.shape == (3, len(FEATURE_NAMES))
    # the source one-hots differ across rows
    src_cols = [FEATURE_NAMES.index(c) for c in ("src_depmap_sanger", "src_orcs", "src_replogle")]
    assert X[0, src_cols].tolist() == [1.0, 0.0, 0.0]
    assert X[1, src_cols].tolist() == [0.0, 1.0, 0.0]
    assert X[2, src_cols].tolist() == [0.0, 0.0, 1.0]
    # the transcriptomic row is flagged
    assert X[2, FEATURE_NAMES.index("task_transcriptomic")] == 1.0


@pytest.mark.skipif(
    not (RAW / "CRISPRGeneEffect.csv").exists()
    or not (RAW / "BIOGRID-ORCS-ALL-homo_sapiens-2.0.18.screens.tar.gz").exists(),
    reason="real data not downloaded",
)
def test_merged_corpus_has_all_sources():
    from cascade.merged import build_merged_corpus

    m = build_merged_corpus(seed=0)
    assert set(m.stats["by_source"]) >= {"depmap_sanger", "orcs"}
    assert m.stats["n_non_hits"] > 0  # real negatives kept
    assert "fitness" in m.stats["by_task"]
    assert m.provenance_hash  # pinned to the data release
