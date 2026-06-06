"""Feature construction for the Replication Oracle (CASCADE component C5 inputs).

The oracle predicts P(replicate | gene p, context A→B). The spec's feature set is
g(E_pert(p), E_cell(A), E_cell(B), Δcontext, E_mod, β_A, σ²_A, QC_A, QC_B, E-dist_A).
With real frozen embeddings these slot straight in; without them we fall back to
the structured signals SplicR always has (effect magnitude, signal-to-noise, QC,
categorical context deltas, modality). The feature order is fixed and named so
provenance and the abstention/support check stay stable.
"""

from __future__ import annotations

import numpy as np

from .provenance import QCWeightParams, quality_weight
from .types import Context, ReplicationPair

FEATURE_NAMES: list[str] = [
    "abs_beta_a",
    "snr_a",
    "log_var_a",
    "qc_a",
    "qc_b",
    "edist_a",
    "same_lineage",
    "same_genetic_bg",
    "same_condition",
    "ctx_emb_dist",
    "mod_KO",
    "mod_CRISPRi",
    "mod_CRISPRa",
    "src_depmap_sanger",
    "src_orcs",
    "src_replogle",
    "task_transcriptomic",
]

_MODALITIES = ["KO", "CRISPRi", "CRISPRa"]
_SOURCES = ["depmap_sanger", "orcs", "replogle"]


def _context_embedding_distance(a: Context, b: Context) -> float:
    if a.embedding is not None and b.embedding is not None and a.embedding.shape == b.embedding.shape:
        return float(np.linalg.norm(a.embedding - b.embedding))
    return 0.0


def pair_features(pair: ReplicationPair, qc_params: QCWeightParams | None = None) -> np.ndarray:
    a, b = pair.context_a, pair.context_b
    var_a = max(pair.var_a, 1e-12)
    snr_a = abs(pair.beta_a) / np.sqrt(var_a)
    mod_onehot = [1.0 if pair.modality == m else 0.0 for m in _MODALITIES]
    src_onehot = [1.0 if pair.source == s else 0.0 for s in _SOURCES]
    # Prefer real data-derived quality scalars when present (real path); fall back
    # to the QCBundle sigmoid (fixtures/tests).
    q_a = pair.quality_a if pair.quality_a is not None else quality_weight(pair.qc_a, qc_params)
    q_b = pair.quality_b if pair.quality_b is not None else quality_weight(pair.qc_b, qc_params)
    vec = [
        abs(pair.beta_a),
        snr_a,
        float(np.log(var_a)),
        q_a,
        q_b,
        pair.edist_a,
        1.0 if a.lineage == b.lineage else 0.0,
        1.0 if a.genetic_background == b.genetic_background else 0.0,
        1.0 if a.condition == b.condition else 0.0,
        _context_embedding_distance(a, b),
        *mod_onehot,
        *src_onehot,
        1.0 if pair.task == "transcriptomic" else 0.0,
    ]
    return np.array(vec, dtype=float)


def featurize(pairs: list[ReplicationPair], qc_params: QCWeightParams | None = None):
    """Return (X, y) for a list of replication pairs."""
    if not pairs:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0)
    X = np.vstack([pair_features(p, qc_params) for p in pairs])
    y = np.array([1 if p.label else 0 for p in pairs], dtype=int)
    return X, y


def context_group(pair: ReplicationPair) -> tuple:
    """Mondrian group key: stratify conformal coverage by
    (source × task × modality × lineage-pair × pair_type)."""
    return (
        pair.source,
        pair.task,
        pair.modality,
        pair.context_a.lineage,
        pair.context_b.lineage,
        pair.pair_type,
    )
