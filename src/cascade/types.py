"""Core data types for CASCADE.

These mirror what SplicR's local install base already produces per screen: a
per-gene effect-size + variance (from MAGeCK β / BAGEL2 BF / DrugZ normZ) plus a
wet-lab QC bundle (the provenance signal that is the moat). Nothing here stores
raw counts/FASTQ — only the sufficient statistics that federation shares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

Modality = str  # one of {"KO", "CRISPRi", "CRISPRa"}


@dataclass(frozen=True)
class QCBundle:
    """Wet-lab quality signals for one screen (the novel weighting signal).

    All fields are in their natural units; `quality_weight` maps them to a scalar
    in (0, 1]. Higher is better for every field except `representation_skew`.
    """

    replicate_r: float  # Pearson r between replicate guide-level LFCs, [-1, 1]
    coverage: float  # mean reads per guide (or cells per guide), > 0
    control_separation: float  # AUROC / SSMD of positive vs negative controls, [0, 1]-ish
    library_complexity: float  # fraction of library represented, (0, 1]
    representation_skew: float  # Gini of guide read distribution, [0, 1] (lower better)

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.replicate_r,
                self.coverage,
                self.control_separation,
                self.library_complexity,
                self.representation_skew,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class Context:
    """Biological context of a screen — what makes a hit context-specific."""

    cell_line: str
    lineage: str
    genetic_background: str = "unknown"
    condition: str = "baseline"  # e.g. drug, hypoxia
    embedding: Optional[np.ndarray] = None  # frozen E_cell vector if available

    def key(self) -> tuple:
        return (self.cell_line, self.lineage, self.genetic_background, self.condition)


@dataclass(frozen=True)
class GeneEffect:
    """Per-gene effect-size + variance — the federated sufficient statistic."""

    gene: str
    beta: float  # effect size (sign: negative = depletion/dependency)
    variance: float  # sampling variance σ² of beta (> 0)
    n_sgrna: int = 4
    n_cells: Optional[int] = None

    def __post_init__(self) -> None:
        if self.variance <= 0:
            raise ValueError(f"variance must be > 0 for gene {self.gene}, got {self.variance}")


@dataclass
class ScreenResult:
    """A complete screen result — the full distribution, hits AND non-hits."""

    screen_id: str
    lab_id: str
    context: Context
    modality: Modality
    qc: QCBundle
    effects: dict[str, GeneEffect] = field(default_factory=dict)
    pos_control_edistance: float = 1.0  # scPerturb E-distance of positive controls

    def genes(self) -> list[str]:
        return list(self.effects.keys())

    def beta(self, gene: str) -> float:
        return self.effects[gene].beta

    def is_valid(self, edist_threshold: float = 0.05) -> bool:
        """Screen validity gate (C2): positive controls must separate."""
        return self.pos_control_edistance > edist_threshold


@dataclass(frozen=True)
class ReplicationPair:
    """A labeled cross-lab replication example (gene observed in A and B).

    This is the training signal the Replication Oracle predicts — and the signal
    that exists only in a federated install base (or the Broad↔Sanger overlap).
    """

    gene: str
    context_a: Context
    context_b: Context
    beta_a: float
    var_a: float
    beta_b: float
    var_b: float
    qc_a: QCBundle
    qc_b: QCBundle
    modality: Modality
    edist_a: float
    label: bool  # did the hit replicate in B?
    # Real-data quality scalars (essential-separation QC computed from the screen).
    # When set, the Oracle uses these directly instead of the QCBundle sigmoid —
    # the gene-effect matrix exposes only control-separation, not the full bundle.
    quality_a: Optional[float] = None
    quality_b: Optional[float] = None
    pair_type: str = "cross_lab"  # "cross_lab" | "cross_context" | "cross_study" | "cross_cell_type"
    # Dataset provenance — the Oracle conditions on this so fitness (KO) and
    # transcriptomic (CRISPRi) replication are never silently mixed as one metric.
    source: str = "depmap_sanger"  # "depmap_sanger" | "orcs" | "replogle"
    # Replication task being measured (kept distinct, never pooled across metrics).
    task: str = "fitness"  # "fitness" | "transcriptomic"
    # Source study/pubmed (ORCS) — enables held-out-STUDY splits. Empty otherwise.
    study: str = ""


@dataclass
class OraclePrediction:
    """A calibrated replication prediction with honest abstention."""

    gene: str
    p_replicate: float
    abstained: bool
    lower: float
    upper: float
    n_comparable: int
    basis: str
    in_support: bool

    def as_dict(self) -> dict:
        return {
            "gene": self.gene,
            "p_replicate": None if self.abstained else round(self.p_replicate, 4),
            "abstained": self.abstained,
            "interval": None if self.abstained else [round(self.lower, 4), round(self.upper, 4)],
            "n_comparable": self.n_comparable,
            "in_support": self.in_support,
            "basis": self.basis,
        }
