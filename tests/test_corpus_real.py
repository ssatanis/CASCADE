"""Tests on the REAL Broad↔Sanger corpus (skipped if data not downloaded)."""

import numpy as np
import pytest

from cascade.corpus import RAW, build_pairs, compute_collapse, load_aligned, screens_for_gene

pytestmark = pytest.mark.skipif(
    not (RAW / "CRISPRGeneEffect.csv").exists() or not (RAW / "gene_effect.csv").exists(),
    reason="real DepMap/Sanger data not downloaded (run cascade/data/acquire.py --only core)",
)


@pytest.fixture(scope="module")
def aligned():
    return load_aligned()


def test_real_overlap_nontrivial(aligned):
    assert len(aligned.common_cell_lines) >= 100  # documented Broad↔Sanger overlap is ~150+
    assert len(aligned.common_genes) >= 10000


def test_real_collapse_direction(aligned):
    col = compute_collapse(aligned)
    assert col["r_raw_fitness"] > col["r_dlfc_deviation"]  # the documented collapse
    assert col["r_raw_fitness"] > 0.7
    assert col["reproduces_documented_collapse_direction"] is True


def test_real_quality_is_comparative(aligned):
    # essential-separation quality spans a real range, not a constant
    assert aligned.q_broad.min() < aligned.q_broad.max()
    assert 0.0 <= aligned.q_broad.min() and aligned.q_broad.max() <= 1.0


def test_build_real_pairs_labeled_and_balanced(aligned):
    pairs = build_pairs(aligned, theta=0.5, max_cross_lab=5000, max_cross_context=2000, seed=0)
    assert len(pairs) > 1000
    labels = np.array([p.label for p in pairs])
    rate = labels.mean()
    assert 0.4 < rate < 0.95  # mostly replicate, but real negatives present
    # real quality scalars present (not the QCBundle sigmoid fallback)
    assert all(p.quality_a is not None and p.quality_b is not None for p in pairs[:50])


def test_screens_for_gene_real(aligned):
    # pick the most-depleted common gene (a real pan-essential) dynamically
    gene = aligned.broad.mean(axis=0).idxmin()
    screens = screens_for_gene(aligned, gene)
    assert len(screens) >= 100
    betas = [s.effects[gene].beta for s in screens]
    assert np.mean(betas) < -0.5  # strongly depleted across cell lines
