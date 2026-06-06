"""Real-data loaders for the v0 Replication Oracle (spec §7).

The v0 trains on the Broad↔Sanger cross-institute overlap (147 cell lines screened
independently by both) — a ready-made cross-lab concordance corpus. These loaders
parse the standard effect-matrix CSV (rows = cell lines, columns = genes; e.g.
DepMap `CRISPRGeneEffect.csv` or Sanger Project Score) into `ScreenResult`s, and
mint replication pairs from two institutes' matrices over shared cell lines.

They run on real files when present; the tests exercise them on small fixtures so
the parsing/labeling logic is verified without the multi-hundred-MB downloads.
"""

from __future__ import annotations

import csv

import numpy as np

from .types import Context, GeneEffect, QCBundle, ReplicationPair, ScreenResult

_DEFAULT_QC = QCBundle(
    replicate_r=0.85,
    coverage=500.0,
    control_separation=0.9,
    library_complexity=0.9,
    representation_skew=0.1,
)


def load_effect_matrix(
    path: str,
    lab_id: str,
    *,
    lineage_map: dict[str, str] | None = None,
    qc: QCBundle | None = None,
    default_variance: float = 0.04,
    modality: str = "KO",
    pos_control_edistance: float = 0.5,
) -> list[ScreenResult]:
    """Parse an effect matrix CSV into one ScreenResult per cell line (row)."""
    qc = qc or _DEFAULT_QC
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        genes = header[1:]
        screens: list[ScreenResult] = []
        for row in reader:
            if not row:
                continue
            cell_line = row[0]
            effects: dict[str, GeneEffect] = {}
            for g, val in zip(genes, row[1:]):
                if val == "" or val is None:
                    continue
                try:
                    beta = float(val)
                except ValueError:
                    continue
                if not np.isfinite(beta):
                    continue
                effects[g] = GeneEffect(gene=g, beta=beta, variance=default_variance)
            lineage = (lineage_map or {}).get(cell_line, "unknown")
            screens.append(
                ScreenResult(
                    screen_id=f"{lab_id}:{cell_line}",
                    lab_id=lab_id,
                    context=Context(cell_line=cell_line, lineage=lineage),
                    modality=modality,
                    qc=qc,
                    effects=effects,
                    pos_control_edistance=pos_control_edistance,
                )
            )
    return screens


def cross_lab_replication_pairs(
    screens_a: list[ScreenResult],
    screens_b: list[ScreenResult],
    *,
    hit_threshold: float = 0.5,
    same_cell_line_only: bool = True,
) -> list[ReplicationPair]:
    """Mint replication pairs from two institutes over shared cell lines.

    A pair is created for each gene that is a hit (|β| > threshold) in screen A;
    its label is whether it replicates (same sign, above threshold) in B.
    """
    by_cl_b: dict[str, ScreenResult] = {s.context.cell_line: s for s in screens_b}
    pairs: list[ReplicationPair] = []
    for sa in screens_a:
        candidates: list[ScreenResult]
        if same_cell_line_only:
            sb = by_cl_b.get(sa.context.cell_line)
            candidates = [sb] if sb is not None else []
        else:
            candidates = screens_b
        for sb in candidates:
            for g, ge_a in sa.effects.items():
                if abs(ge_a.beta) < hit_threshold:
                    continue
                ge_b = sb.effects.get(g)
                if ge_b is None:
                    continue
                replicated = (np.sign(ge_b.beta) == np.sign(ge_a.beta)) and (abs(ge_b.beta) >= hit_threshold)
                pairs.append(
                    ReplicationPair(
                        gene=g,
                        context_a=sa.context,
                        context_b=sb.context,
                        beta_a=ge_a.beta,
                        var_a=ge_a.variance,
                        beta_b=ge_b.beta,
                        var_b=ge_b.variance,
                        qc_a=sa.qc,
                        qc_b=sb.qc,
                        modality=sa.modality,
                        edist_a=sa.pos_control_edistance,
                        label=bool(replicated),
                    )
                )
    return pairs
