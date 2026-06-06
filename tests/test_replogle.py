"""Tests for Replogle cross-cell-type effects + pairs."""

import numpy as np
import pytest

from cascade.replogle import CellTypeEffects, PertEffect, _energy_distance, build_replogle_pairs
from cascade.corpus import DEFAULT_DATA


def test_energy_distance_zero_for_identical():
    x = np.random.default_rng(0).normal(size=(20, 5))
    assert _energy_distance(x, x.copy()) == pytest.approx(0.0, abs=1e-9)


def test_energy_distance_positive_for_separated():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (40, 5))
    y = rng.normal(8, 1, (40, 5))
    assert _energy_distance(x, y) > 1.0


def _effects(cell_line, spec):
    """spec: {gene: (delta_vec, effect_size)}"""
    delta = {g: np.array(d, dtype=float) for g, (d, _e) in spec.items()}
    effects = {g: PertEffect(g, e, e, n_cells=300) for g, (_d, e) in spec.items()}
    return CellTypeEffects(cell_line, list(spec), delta, effects, n_control=500, n_common_genes=3)


def test_build_replogle_pairs_concordance():
    src = _effects("K562", {"G1": ([1, 0, 0], 100.0), "G2": ([1, 0, 0], 100.0), "G3": ([1, 0, 0], 100.0)})
    tgt = _effects("RPE1", {
        "G1": ([1, 0, 0], 100.0),    # concordant + real effect -> replicates
        "G2": ([-1, 0, 0], 100.0),   # opposite direction -> does not replicate
        "G3": ([1, 0, 0], 100.0),
    })
    corpus = build_replogle_pairs(src, tgt, cos_threshold=0.5, eff_quantile=0.0, seed=0)
    by_gene = {p.gene: p for p in corpus.pairs}
    assert by_gene["G1"].label is True
    assert by_gene["G2"].label is False
    for p in corpus.pairs:
        assert p.source == "replogle" and p.task == "transcriptomic" and p.pair_type == "cross_cell_type"
        assert p.modality == "CRISPRi"


@pytest.mark.skipif(
    not (DEFAULT_DATA / "replogle" / "rpe1.npz").exists()
    or not (DEFAULT_DATA / "replogle" / "K562_essential.npz").exists(),
    reason="Replogle effects not computed/cached",
)
def test_real_replogle_corpus():
    from cascade.replogle import build_replogle_corpus

    corpus, k, r = build_replogle_corpus(seed=0)
    assert len(corpus.pairs) > 200
    assert 0.1 < corpus.base_rate < 0.95  # real concordance, both classes present
    # real E-distances computed (some perturbations move the cell cloud)
    assert max(e.edistance for e in r.effects.values()) > 0
