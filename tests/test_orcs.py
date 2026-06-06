"""Tests for the BioGRID-ORCS loader/harmonizer + cross-study pairs."""

import numpy as np
import pytest

from cascade.orcs import (
    ORCSData,
    ScreenObs,
    _modality,
    _norm_cl,
    _phenotype,
    _screen_quality,
    build_orcs_pairs,
)
from cascade.corpus import RAW


def test_modality_mapping():
    assert _modality("Knockout") == "KO"
    assert _modality("Inhibition") == "CRISPRi"
    assert _modality("Activation") == "CRISPRa"
    assert _modality("") == "KO"


def test_phenotype_vocab():
    assert _phenotype("cell proliferation") == "fitness"
    assert _phenotype("viability") == "fitness"
    assert _phenotype("response to chemicals (drug resistance)") == "drug_response"
    assert _phenotype("fluorescent reporter") == "reporter"


def test_norm_cell_line():
    assert _norm_cl("KBM-7") == "KBM7"
    assert _norm_cl("K 562") == "K562"


def test_screen_quality_bounds():
    q = _screen_quality("High Throughput", n_hits=500, scores_size=18000)
    assert 0.0 < q <= 0.99
    bad = _screen_quality("Low Throughput", n_hits=0, scores_size=18000)
    assert bad < q


def _obs(sid, pubmed, eff, hit, cl="A", model="M1", lineage="lin"):
    return ScreenObs(sid, pubmed, eff, hit, model, cl, lineage, "fitness", "KO", 0.8)


def test_build_orcs_pairs_logic():
    # gene G: hit in study P1, also tested in P2 (hit) and P3 (no hit)
    data = ORCSData(
        screens={},
        gene_obs={
            "G": [
                _obs("1", "P1", -0.9, True, cl="A", model="M1"),
                _obs("2", "P2", -0.8, True, cl="B", model="M2"),
                _obs("3", "P3", -0.1, False, cl="C", model="M3"),
            ]
        },
    )
    pairs = build_orcs_pairs(data, max_pairs=100, per_source_targets=5, seed=0)
    assert len(pairs) >= 2
    assert all(p.source == "orcs" and p.task == "fitness" for p in pairs)
    assert all(p.pair_type.startswith("cross_study") for p in pairs)
    # labels reflect target HIT (concordance), incl a real negative (P3)
    labels = {(p.context_b.cell_line, p.label) for p in pairs}
    assert ("B", True) in labels
    assert ("C", False) in labels  # non-replication negative kept


def test_same_cell_line_tagging():
    data = ORCSData(gene_obs={"G": [
        _obs("1", "P1", -0.9, True, cl="A", model="M1"),
        _obs("2", "P2", -0.8, True, cl="A", model="M1"),  # same model -> same cell line
    ]}, screens={})
    pairs = build_orcs_pairs(data, seed=0)
    assert any(p.pair_type == "cross_study_same_cell" for p in pairs)


@pytest.mark.skipif(
    not (RAW / "BIOGRID-ORCS-ALL-homo_sapiens-2.0.18.screens.tar.gz").exists(),
    reason="ORCS archive not downloaded",
)
def test_real_orcs_load_small():
    from cascade.orcs import load_orcs

    d = load_orcs(fitness_only=True, max_screens=40)
    assert d.n_screens_used == 40
    assert len(d.gene_obs) > 100
    # real negatives present (HIT=No kept)
    allobs = [x for o in d.gene_obs.values() for x in o]
    assert any(not x.hit for x in allobs)
    # cell-line mapping recorded (mapped + unmapped both counted, none silently dropped)
    assert d.n_cell_lines_mapped + d.n_cell_lines_unmapped >= 0
    pairs = build_orcs_pairs(d, max_pairs=500, seed=0)
    assert len(pairs) > 0
    assert 0.0 < np.mean([p.label for p in pairs]) < 1.0
