"""Build the REAL Broad↔Sanger cross-lab replication corpus (spec Part C).

Sources (both Chronos gene-effect, from the DepMap-hosted, checksummed API):
  - Broad : DepMap Public CRISPRGeneEffect.csv  (rows = ACH model IDs, cols = "SYMBOL (ENTREZ)")
  - Sanger: Project Score Chronos gene_effect.csv (same layout, same units)

For each (gene × common cell line) we derive, from REAL data only:
  - the replication label  y = 1[ sign(β_Broad)==sign(β_Sanger) AND |β_Sanger| > θ ]
  - real per-screen QC = essential-vs-non-essential separation (the standard screen
    quality metric, computable from the matrix)
  - real per-screen noise variance = spread of non-essential gene effects (the null floor)
  - the dLFC (context-specific deviation = effect minus the gene's cross-line mean)

No value here is fabricated; everything is computed from the downloaded matrices.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .types import Context, GeneEffect, QCBundle, ReplicationPair, ScreenResult

NAN = float("nan")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = Path(os.environ.get("CASCADE_DATA_DIR", REPO_ROOT / "data"))
RAW = DEFAULT_DATA / "cascade" / "raw"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class AlignedData:
    broad: pd.DataFrame
    sanger: pd.DataFrame
    lineage: dict[str, str]
    essential_genes: set[str]
    common_cell_lines: list[str]
    common_genes: list[str]
    # per-cell-line real quality (essential separation, comparative within cohort)
    q_broad: pd.Series
    q_sanger: pd.Series
    # per-cell-line real noise variance (non-essential effect spread)
    var_broad: pd.Series
    var_sanger: pd.Series


def _screen_quality(df: pd.DataFrame, essential_cols: list[str], nonessential_cols: list[str]) -> pd.Series:
    """Per-cell-line essential-vs-non-essential separation → comparative quality (0,1)."""
    ess = df[essential_cols].mean(axis=1)
    non = df[nonessential_cols].mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    sep = (non - ess) / sd  # large positive = essentials clearly deplete = good screen
    med = np.nanmedian(sep.values)
    mad = np.nanmedian(np.abs(sep.values - med)) * 1.4826 or 1.0
    return pd.Series(_sigmoid((sep.values - med) / mad), index=df.index)


def _screen_noise_var(df: pd.DataFrame, nonessential_cols: list[str]) -> pd.Series:
    """Per-cell-line noise floor = variance of non-essential gene effects (null spread)."""
    v = df[nonessential_cols].var(axis=1)
    v = v.fillna(v.median())
    return v.clip(lower=1e-4)


def load_aligned(
    broad_path: str | Path | None = None,
    sanger_path: str | Path | None = None,
    model_path: str | Path | None = None,
    essentials_path: str | Path | None = None,
) -> AlignedData:
    broad_path = Path(broad_path or RAW / "CRISPRGeneEffect.csv")
    sanger_path = Path(sanger_path or RAW / "gene_effect.csv")
    model_path = Path(model_path or RAW / "Model.csv")
    essentials_path = Path(essentials_path or RAW / "common_essentials.csv")
    for p in (broad_path, sanger_path, model_path, essentials_path):
        if not p.exists():
            raise FileNotFoundError(f"Required real dataset missing: {p}. Run cascade/data/acquire.py --only core")

    broad = pd.read_csv(broad_path, index_col=0)
    sanger = pd.read_csv(sanger_path, index_col=0)
    model = pd.read_csv(model_path)
    ess = pd.read_csv(essentials_path)
    essential_genes = set(ess.iloc[:, 0].astype(str))

    common_cl = sorted(set(broad.index) & set(sanger.index))
    common_g = sorted(set(broad.columns) & set(sanger.columns))
    if not common_cl or not common_g:
        raise ValueError("No common cell lines/genes between Broad and Sanger matrices")

    broad = broad.loc[common_cl, common_g]
    sanger = sanger.loc[common_cl, common_g]

    lineage_col = "OncotreeLineage" if "OncotreeLineage" in model.columns else model.columns[5]
    lineage = {
        str(r["ModelID"]): str(r[lineage_col]) if pd.notna(r[lineage_col]) else "unknown"
        for _, r in model.iterrows()
    }

    ess_cols = [g for g in common_g if g in essential_genes]
    non_cols = [g for g in common_g if g not in essential_genes]
    if len(ess_cols) < 50 or len(non_cols) < 50:
        raise ValueError("Too few essential/non-essential genes to compute screen QC")

    q_broad = _screen_quality(broad, ess_cols, non_cols)
    q_sanger = _screen_quality(sanger, ess_cols, non_cols)
    var_broad = _screen_noise_var(broad, non_cols)
    var_sanger = _screen_noise_var(sanger, non_cols)

    return AlignedData(
        broad=broad, sanger=sanger, lineage=lineage, essential_genes=essential_genes,
        common_cell_lines=common_cl, common_genes=common_g,
        q_broad=q_broad, q_sanger=q_sanger, var_broad=var_broad, var_sanger=var_sanger,
    )


def compute_collapse(aligned: AlignedData) -> dict:
    """Real Broad↔Sanger concordance collapse: corr(raw effect) vs corr(dLFC)."""
    b = aligned.broad.values
    s = aligned.sanger.values
    mask = np.isfinite(b) & np.isfinite(s)
    bv, sv = b[mask], s[mask]
    r_raw = float(np.corrcoef(bv, sv)[0, 1])

    # dLFC = effect minus the gene's cross-cell-line mean (context-specific deviation)
    bd = b - np.nanmean(b, axis=0, keepdims=True)
    sd = s - np.nanmean(s, axis=0, keepdims=True)
    r_dlfc = float(np.corrcoef(bd[mask], sd[mask])[0, 1])

    return {
        "n_common_cell_lines": len(aligned.common_cell_lines),
        "n_common_genes": len(aligned.common_genes),
        "n_gene_celline_pairs": int(mask.sum()),
        "r_raw_fitness": round(r_raw, 4),
        "r_dlfc_deviation": round(r_dlfc, 4),
        "collapse_delta": round(r_raw - r_dlfc, 4),
        "reproduces_documented_collapse_direction": bool(r_raw > r_dlfc + 0.1),
        "reference": "Broad↔Sanger documented r 0.81 (raw) → 0.47 (dLFC); exact dLFC magnitude is release/method dependent",
    }


def _qc_carrier(separation_quality: float) -> QCBundle:
    """Carrier QCBundle storing only the real, computable field (control_separation);
    unavailable guide-level fields are NaN (honest 'not measured'), never fabricated."""
    return QCBundle(
        replicate_r=NAN,
        coverage=NAN,
        control_separation=float(separation_quality),
        library_complexity=NAN,
        representation_skew=NAN,
    )


def build_pairs(
    aligned: AlignedData,
    theta: float = 0.5,
    max_cross_lab: int = 45000,
    max_cross_context: int = 20000,
    cross_context_per_hit: int = 2,
    seed: int = 0,
) -> list[ReplicationPair]:
    rng = np.random.default_rng(seed)
    cl = aligned.common_cell_lines
    genes = aligned.common_genes
    B = aligned.broad.values
    S = aligned.sanger.values
    qb = aligned.q_broad.reindex(cl).values
    qs = aligned.q_sanger.reindex(cl).values
    vb = aligned.var_broad.reindex(cl).values
    vs = aligned.var_sanger.reindex(cl).values
    lineage = [aligned.lineage.get(c, "unknown") for c in cl]

    pairs: list[ReplicationPair] = []

    # ---- cross-lab: Broad(source) → Sanger(target), same cell line ----
    hit = (np.abs(B) > theta) & np.isfinite(B) & np.isfinite(S)
    hit_idx = np.argwhere(hit)  # (cell_line_i, gene_j)
    if len(hit_idx) > max_cross_lab:
        sel = rng.choice(len(hit_idx), size=max_cross_lab, replace=False)
        hit_idx = hit_idx[sel]
    for i, j in hit_idx:
        ba, bb = float(B[i, j]), float(S[i, j])
        label = (np.sign(ba) == np.sign(bb)) and (abs(bb) > theta)
        ctx = Context(cell_line=cl[i], lineage=lineage[i])
        pairs.append(
            ReplicationPair(
                gene=genes[j], context_a=ctx, context_b=ctx,
                beta_a=ba, var_a=float(vb[i]), beta_b=bb, var_b=float(vs[i]),
                qc_a=_qc_carrier(qb[i]), qc_b=_qc_carrier(qs[i]),
                modality="KO", edist_a=float(qb[i]), label=bool(label),
                quality_a=float(qb[i]), quality_b=float(qs[i]), pair_type="cross_lab",
            )
        )

    # ---- cross-context: within Broad, source cell line → other cell line ----
    cc_idx = np.argwhere((np.abs(B) > theta) & np.isfinite(B))
    if len(cc_idx) * cross_context_per_hit > max_cross_context:
        sel = rng.choice(len(cc_idx), size=max(1, max_cross_context // cross_context_per_hit), replace=False)
        cc_idx = cc_idx[sel]
    n_cl = len(cl)
    for i, j in cc_idx:
        ba = float(B[i, j])
        targets = rng.choice(n_cl, size=min(cross_context_per_hit, n_cl), replace=False)
        for t in targets:
            if t == i or not np.isfinite(B[t, j]):
                continue
            bb = float(B[t, j])
            label = (np.sign(ba) == np.sign(bb)) and (abs(bb) > theta)
            pairs.append(
                ReplicationPair(
                    gene=genes[j],
                    context_a=Context(cell_line=cl[i], lineage=lineage[i]),
                    context_b=Context(cell_line=cl[t], lineage=lineage[t]),
                    beta_a=ba, var_a=float(vb[i]), beta_b=bb, var_b=float(vb[t]),
                    qc_a=_qc_carrier(qb[i]), qc_b=_qc_carrier(qb[t]),
                    modality="KO", edist_a=float(qb[i]), label=bool(label),
                    quality_a=float(qb[i]), quality_b=float(qb[t]), pair_type="cross_context",
                )
            )
            if len([p for p in pairs if p.pair_type == "cross_context"]) >= max_cross_context:
                break

    return pairs


def screens_for_gene(aligned: AlignedData, gene: str) -> list[ScreenResult]:
    """Build one real Broad ScreenResult per cell line for `gene` (for the
    federated meta-analysis demo on real data). Carries real quality + noise."""
    if gene not in aligned.broad.columns:
        raise KeyError(f"gene '{gene}' not in the common gene set")
    col = aligned.broad[gene]
    out: list[ScreenResult] = []
    for c in aligned.common_cell_lines:
        beta = col.loc[c]
        if not np.isfinite(beta):
            continue
        q = float(aligned.q_broad.loc[c])
        v = float(aligned.var_broad.loc[c])
        out.append(
            ScreenResult(
                screen_id=f"broad:{c}", lab_id=f"broad:{c}",
                context=Context(cell_line=c, lineage=aligned.lineage.get(c, "unknown")),
                modality="KO", qc=_qc_carrier(q),
                effects={gene: GeneEffect(gene=gene, beta=float(beta), variance=v)},
                pos_control_edistance=q,
            )
        )
    return out


def provenance_hash(manifest_path: str | Path | None = None) -> str:
    """sha256 over the provenance manifest's per-entry sha256 list (release pin)."""
    manifest_path = Path(manifest_path or DEFAULT_DATA / "PROVENANCE.json")
    if not manifest_path.exists():
        return "no-manifest"
    m = json.loads(manifest_path.read_text())
    items = sorted(
        f"{k}:{v.get('sha256')}" for k, v in m["entries"].items() if v.get("status") == "ok"
    )
    return hashlib.sha256("\n".join(items).encode()).hexdigest()
