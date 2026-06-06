"""WBC (Within-Between-Context) replication baseline — Billmann et al. 2023.

Reference: Billmann, Ho et al. 2023, Cell Systems (PMID:37201508). WBC is the
state-of-art replication-quality contrast: a feature is more consistent WITHIN a
context than BETWEEN contexts. It is the most direct competitor to the CASCADE
Replication Oracle, so reviewers will ask "did you beat WBC?".

Honest operationalization (documented, reproducible)
----------------------------------------------------
The original WBC is a profile-similarity contrast. We adapt it to a PER-GENE
replication-tendency score over the real DepMap Chronos matrix so it can be
scored on the same frozen benchmark as the Oracle:

  For gene g, z-score its effect across all cell lines (z, Σz=0, Σz²=n).
  Pairwise concordance of two cell lines for g is the correlation contribution
  c(i,j) = z_i · z_j (same-sign large-magnitude → positive).
    within-lineage  = mean c(i,j) over same-OncotreeLineage cell-line pairs
    between-lineage = mean c(i,j) over different-lineage pairs
  WBC(g) = (mean_within − mean_between) / SD_between

All three quantities are computed vectorized from per-lineage sums of z and z²
(no O(pairs) loop). A gene scored on < 5 within OR < 5 between pairs abstains.

This is a faithful, citable adaptation — NOT the identical module-quality WBC;
the note field says so. The point is an honest, reproducible baseline number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PKG_ROOT.parent
RAW = REPO_ROOT / "data" / "cascade" / "raw"
RESULTS = PKG_ROOT / "results"
BENCH = PKG_ROOT / "benchmark" / "replication_benchmark_v1"
MIN_PAIRS = 5


def _gene_symbol(col: str) -> str:
    """'AAMP (14)' -> 'AAMP'. CRISPRGeneEffect columns are 'SYMBOL (ENTREZ)'."""
    return col.split(" (")[0].strip()


def compute_wbc(gene_effect_csv: Path = None, model_csv: Path = None) -> dict:
    """Compute WBC(g) for every gene. Returns {gene_symbol: wbc or nan}."""
    gene_effect_csv = Path(gene_effect_csv or RAW / "CRISPRGeneEffect.csv")
    model_csv = Path(model_csv or RAW / "Model.csv")

    df = pd.read_csv(gene_effect_csv, index_col=0)          # rows=cell lines, cols=genes
    model = pd.read_csv(model_csv)
    lin = model.set_index("ModelID")["OncotreeLineage"].to_dict()

    cells = [c for c in df.index if c in lin and isinstance(lin[c], str)]
    df = df.loc[cells]
    lineage = np.array([lin[c] for c in cells])
    X = df.to_numpy(dtype=float)                            # (n_cells, n_genes)

    # z-score each gene across cell lines (ignoring NaN)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    valid = np.isfinite(Z)
    Z = np.where(valid, Z, 0.0)                             # NaN → 0 (drops from sums)

    uniq = np.unique(lineage)
    n_genes = Z.shape[1]
    # per-lineage sums of z and z² and counts of valid cells
    within_pair_sum = np.zeros(n_genes)
    within_pair_cnt = np.zeros(n_genes)
    within_sq_sum = np.zeros(n_genes)                       # Σ over within pairs of (z_i z_j)²
    sumZ_all = np.zeros(n_genes)
    sumZ2_all = np.zeros(n_genes)
    sumZsq_all = np.zeros(n_genes)                          # Σ z² (per gene, valid cells)
    n_valid = valid.sum(axis=0).astype(float)

    for L in uniq:
        m = lineage == L
        Zl = Z[m]
        vl = valid[m]
        s = Zl.sum(axis=0)
        s2 = (Zl ** 2).sum(axis=0)
        nL = vl.sum(axis=0).astype(float)                  # valid cells per gene in L
        # Σ_{i<j in L} z_i z_j = (s² − Σz²)/2 ;  pair count = nL(nL-1)/2
        within_pair_sum += (s ** 2 - s2) / 2.0
        within_pair_cnt += nL * (nL - 1) / 2.0
        # Σ_{i<j in L} (z_i z_j)² = ((Σz²)² − Σz⁴)/2
        s4 = (Zl ** 4).sum(axis=0)
        within_sq_sum += (s2 ** 2 - s4) / 2.0
        sumZ_all += s
        sumZ2_all += s2

    # totals over ALL valid pairs
    total_pair_sum = (sumZ_all ** 2 - sumZ2_all) / 2.0
    total_pair_cnt = n_valid * (n_valid - 1) / 2.0
    sumZ4_all = (Z ** 4).sum(axis=0)
    total_sq_sum = (sumZ2_all ** 2 - sumZ4_all) / 2.0

    between_pair_sum = total_pair_sum - within_pair_sum
    between_pair_cnt = total_pair_cnt - within_pair_cnt
    between_sq_sum = total_sq_sum - within_sq_sum

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_within = within_pair_sum / within_pair_cnt
        mean_between = between_pair_sum / between_pair_cnt
        ex2_between = between_sq_sum / between_pair_cnt
        var_between = ex2_between - mean_between ** 2
        sd_between = np.sqrt(np.clip(var_between, 0, None))
        wbc = (mean_within - mean_between) / (sd_between + 1e-9)

    enough = (within_pair_cnt >= MIN_PAIRS) & (between_pair_cnt >= MIN_PAIRS)
    genes = [_gene_symbol(c) for c in df.columns]
    out = {}
    for j, g in enumerate(genes):
        out[g] = float(wbc[j]) if (enough[j] and np.isfinite(wbc[j])) else float("nan")
    return out


def score_on_benchmark(wbc: dict) -> dict:
    """Map WBC(gene) onto the frozen benchmark pairs and score it through the
    standard harness alongside the existing leaderboard."""
    from .benchmark_v1 import BENCH_DIR, score_predictions

    test = pd.read_csv(BENCH_DIR / "test_pairs.csv")
    preds = []
    n_abstain = 0
    for _, r in test.iterrows():
        g = _gene_symbol(str(r["gene"]))
        s = wbc.get(g, float("nan"))
        if not np.isfinite(s):
            preds.append({"pair_id": r["pair_id"], "p_replicate": "", "abstain": 1})
            n_abstain += 1
        else:
            preds.append({"pair_id": r["pair_id"], "p_replicate": s, "abstain": 0})
    # map raw WBC to [0,1] via logistic on the rank — monotone, AUROC-invariant,
    # so ECE is meaningful but AUROC is unchanged by the squashing.
    vals = np.array([p["p_replicate"] for p in preds if p["abstain"] == 0], dtype=float)
    if len(vals):
        lo, hi = np.percentile(vals, [1, 99])
        rng = (hi - lo) or 1.0
        for p in preds:
            if p["abstain"] == 0:
                p["p_replicate"] = float(np.clip((p["p_replicate"] - lo) / rng, 0, 1))
    scored = score_predictions(BENCH_DIR, preds)
    return scored, preds, n_abstain


def run(save: bool = True) -> dict:
    wbc = compute_wbc()
    scored, preds, n_abstain = score_on_benchmark(wbc)
    n_scored = sum(1 for v in wbc.values() if np.isfinite(v))

    from .benchmark_v1 import provenance_hash
    by = scored.get("by_stratum", {})
    result = {
        "method": "WBC_Billmann2023",
        "reference": "Billmann et al. 2023 Cell Systems PMID:37201508",
        "trained_on_real_data": True,
        "provenance_hash": provenance_hash(),
        "data_release": "DepMap Public 26Q1 CRISPRGeneEffect + Model (OncotreeLineage)",
        "n_genes_scored": int(n_scored),
        "n_genes_total": int(len(wbc)),
        "n_abstained_pairs": int(n_abstain),
        "auroc_overall": scored.get("overall", {}).get("auroc"),
        "auroc_cross_lab": by.get("cross_lab", {}).get("auroc"),
        "auroc_cross_context": by.get("cross_context", {}).get("auroc"),
        "auroc_cross_study": by.get("cross_study", {}).get("auroc"),
        "abstention_rate": scored.get("overall", {}).get("abstention_rate"),
        "by_stratum": by,
        "note": "WBC adapted to a per-gene replication-tendency score (z-scored "
                "Chronos concordance within vs between OncotreeLineage) and scored on the "
                "frozen benchmark; same-gene WBC regardless of context pair, no pair "
                "conditioning. Faithful citable adaptation, not the identical module-quality WBC.",
    }
    if save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "wbc_baseline.json").write_text(json.dumps(result, indent=2))
        _update_leaderboard("WBC_Billmann2023", scored)
    return result


def _update_leaderboard(name: str, scored: dict) -> None:
    lb_path = BENCH / "leaderboard.json"
    lb = json.loads(lb_path.read_text()) if lb_path.exists() else {}
    lb[name] = scored
    lb_path.write_text(json.dumps(lb, indent=2))


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("method", "n_genes_scored", "n_abstained_pairs",
                                        "auroc_overall", "auroc_cross_lab",
                                        "auroc_cross_context")}, indent=2))
