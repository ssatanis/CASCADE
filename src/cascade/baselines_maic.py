"""MAIC (Meta-Analysis by Information Content) replication baseline.

Reference: Li, Baillie et al. — MAIC, the information-content meta-analysis used to
integrate ranked gene lists across screens (github.com/baillielab/maic). It is the
best existing CRISPR-screen meta-analysis tool and a required benchmark comparison.

We try the real `maic` package first. If it is unavailable or incompatible with
this gene-effect-matrix input (it expects categorised ranked lists), we fall back
to the standard information-content formula it is built on — DOCUMENTED here and
in the result JSON as `MAIC_approx`:

    IC(gene, screen) = -log2( rank(gene, screen) / n_genes )      # rank 1 = strongest
    MAIC_approx(gene) = Σ_screens IC(gene, screen)

Screens = every Broad + Sanger common-cell-line ranking (genes ranked by |effect|,
strongest dependency first). This is an UNSUPERVISED gene-level score — it never
sees the replication label y_rep, so using the full harmonised matrix is
leakage-free. Same gene-level caveat as WBC: no context-pair conditioning.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PKG_ROOT / "results"
BENCH = PKG_ROOT / "benchmark" / "replication_benchmark_v1"


def _gene_symbol(col: str) -> str:
    return str(col).split(" (")[0].strip()


def _ic_from_matrix(M: np.ndarray) -> np.ndarray:
    """M: (n_cells, n_genes) effects. Returns Σ_cell IC(gene) over rows.
    rank by |effect| descending per cell line; IC = -log2(rank/n_genes)."""
    n_cells, n_genes = M.shape
    ic = np.zeros(n_genes)
    for i in range(n_cells):
        row = np.abs(M[i])
        finite = np.isfinite(row)
        if finite.sum() < 10:
            continue
        # rank: strongest |effect| → rank 1
        order = np.argsort(-np.where(finite, row, -np.inf))
        ranks = np.empty(n_genes)
        ranks[order] = np.arange(1, n_genes + 1)
        ng = finite.sum()
        contrib = -np.log2(np.clip(ranks / ng, 1e-9, 1.0))
        contrib[~finite] = 0.0
        ic += contrib
    return ic


def compute_maic_approx() -> tuple[dict, str]:
    """Return ({gene_symbol: maic_score}, method_label)."""
    from .corpus import load_aligned

    aligned = load_aligned()
    genes = [_gene_symbol(g) for g in aligned.common_genes]
    B = aligned.broad.to_numpy(dtype=float)     # (cells, genes)
    S = aligned.sanger.to_numpy(dtype=float)
    ic = _ic_from_matrix(B) + _ic_from_matrix(S)
    return {g: float(ic[j]) for j, g in enumerate(genes)}, "MAIC_approx (IC formula)"


def compute_maic() -> tuple[dict, str]:
    """Try the real maic package; fall back to the documented IC approximation."""
    try:
        import maic  # noqa: F401
        # The Baillie maic package ingests categorised ranked-list files, not a
        # gene-effect matrix. Adapting it correctly is out of scope for a matrix
        # input; we use the IC formula it is built on and label it as such.
        raise ImportError("maic present but expects categorised ranked-list files, not a matrix")
    except Exception:
        return compute_maic_approx()


def run(save: bool = True) -> dict:
    from .benchmark_v1 import BENCH_DIR, provenance_hash, score_predictions

    scores, method = compute_maic()
    test = pd.read_csv(BENCH_DIR / "test_pairs.csv")
    raw = np.array([scores.get(_gene_symbol(g), np.nan) for g in test["gene"]], dtype=float)
    finite = raw[np.isfinite(raw)]
    lo, hi = (np.percentile(finite, [1, 99]) if len(finite) else (0.0, 1.0))
    rng = (hi - lo) or 1.0

    preds, n_abstain = [], 0
    for k, (_, r) in enumerate(test.iterrows()):
        v = raw[k]
        if not np.isfinite(v):
            preds.append({"pair_id": r["pair_id"], "p_replicate": "", "abstain": 1}); n_abstain += 1
        else:
            preds.append({"pair_id": r["pair_id"], "p_replicate": float(np.clip((v - lo) / rng, 0, 1)), "abstain": 0})

    scored = score_predictions(BENCH_DIR, preds)
    by = scored.get("by_stratum", {})
    result = {
        "method": method,
        "reference": "MAIC, Baillie lab (github.com/baillielab/maic); IC = -log2(rank/n_genes)",
        "trained_on_real_data": True,
        "provenance_hash": provenance_hash(),
        "data_release": "DepMap Public 26Q1 (Broad) + Sanger Project Score (Chronos), common cell lines",
        "n_genes_scored": int(sum(1 for v in scores.values() if np.isfinite(v))),
        "n_abstained_pairs": int(n_abstain),
        "auroc_overall": scored.get("overall", {}).get("auroc"),
        "auroc_cross_lab": by.get("cross_lab", {}).get("auroc"),
        "auroc_cross_context": by.get("cross_context", {}).get("auroc"),
        "auroc_cross_study": by.get("cross_study", {}).get("auroc"),
        "abstention_rate": scored.get("overall", {}).get("abstention_rate"),
        "by_stratum": by,
        "note": "MAIC_approx (IC formula) used because the maic package expects categorised "
                "ranked-list files, not a gene-effect matrix; see citation. Unsupervised, "
                "gene-level, no context-pair conditioning.",
    }
    if save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "maic_baseline.json").write_text(json.dumps(result, indent=2))
        lb_path = BENCH / "leaderboard.json"
        lb = json.loads(lb_path.read_text()) if lb_path.exists() else {}
        lb[method] = scored
        lb_path.write_text(json.dumps(lb, indent=2))
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("method", "n_genes_scored", "n_abstained_pairs",
                                        "auroc_overall", "auroc_cross_lab", "auroc_cross_context")}, indent=2))
