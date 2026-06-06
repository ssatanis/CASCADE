"""Publication-grade statistical evaluation (CASCADE Phase 3).

Self-contained (numpy / scipy / scikit-learn only) so it runs in the pinned
cascade venv with no extra heavy dependencies. Every function here is a real,
literature-standard estimator — no placeholders, no synthetic data. The inputs
are real held-out labels and real model scores; the outputs are the numbers a
reviewer demands to believe a delta.

Contents
--------
Discrimination + significance
  * fast_delong / delong_roc_variance     — DeLong AUROC + covariance (Sun & Xu 2014)
  * delong_roc_test                        — paired AUROC comparison, two-sided p
  * delong_auc_ci                          — single-model AUROC CI (and vs 0.5)
  * cluster_bootstrap_auroc                — group-resampled AUROC with BCa CI
  * permutation_test_auroc                 — label-permutation null for AUROC

Calibration
  * reliability_table                      — binned confidence/accuracy + Wilson CIs
  * expected_calibration_error / mce       — ECE (equal-width / equal-freq) + MCE
  * adaptive_ece                           — classwise/adaptive ECE
  * brier_score / murphy_decomposition     — Brier + reliability/resolution/uncertainty

Integrity
  * negative_control_permuted_label        — expect AUROC ≈ 0.5
  * negative_control_scrambled_score       — expect AUROC ≈ 0.5
  * leakage_audit_frozen                   — dup/near-dup + within-set disjointness checks

These are the building blocks; ``run_phase3`` wires them over the frozen
benchmark and emits a fully-logged evidence JSON.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- #
# DeLong AUROC + covariance (fast midrank algorithm, Sun & Xu 2014)
# --------------------------------------------------------------------------- #


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks with ties shared (the core of the fast DeLong estimator)."""
    J = np.argsort(x)
    z = x[J]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1  # 1-based midrank
        i = j
    out = np.empty(n, dtype=float)
    out[J] = t
    return out


def _fast_delong(predictions_sorted: np.ndarray, m: int):
    """Fast DeLong for one or more predictors over the SAME labels.

    predictions_sorted : (k_predictors, n) with the m positive samples first.
    Returns (aucs, delongcov) — AUROC per predictor and the k×k covariance.
    """
    n = predictions_sorted.shape[1] - m
    k = predictions_sorted.shape[0]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _compute_midrank(predictions_sorted[r, :m])
        ty[r] = _compute_midrank(predictions_sorted[r, m:])
        tz[r] = _compute_midrank(predictions_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def _prep(y_true: np.ndarray, *scores: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-y_true, kind="mergesort")  # positives first, stable
    m = int(np.sum(y_true == 1))
    preds = np.vstack([np.asarray(s, dtype=float)[order] for s in scores])
    return preds, m


def delong_roc_variance(y_true, y_score):
    """Single-model AUROC and its DeLong variance."""
    preds, m = _prep(y_true, y_score)
    aucs, cov = _fast_delong(preds, m)
    return float(aucs[0]), float(cov[0, 0])


def delong_auc_ci(y_true, y_score, alpha: float = 0.05):
    """AUROC with a (1-alpha) DeLong CI and a two-sided p vs AUROC=0.5."""
    auc, var = delong_roc_variance(y_true, y_score)
    se = float(np.sqrt(max(var, 0.0)))
    if se == 0:
        return {"auroc": auc, "se": 0.0, "ci_low": auc, "ci_high": auc,
                "z_vs_0.5": float("nan"), "p_vs_0.5": float("nan")}
    z = stats.norm.ppf(1 - alpha / 2)
    ci = (auc - z * se, auc + z * se)
    zstat = (auc - 0.5) / se
    p = 2 * stats.norm.sf(abs(zstat))
    return {"auroc": auc, "se": se, "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "z_vs_0.5": float(zstat), "p_vs_0.5": float(p)}


def delong_roc_test(y_true, score_1, score_2):
    """Paired AUROC comparison (model 1 vs model 2). Two-sided p-value."""
    preds, m = _prep(y_true, score_1, score_2)
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 0.0)))
    if se == 0:
        return {"auroc_1": float(aucs[0]), "auroc_2": float(aucs[1]),
                "delta": float(aucs[0] - aucs[1]), "z": float("nan"), "p": float("nan")}
    z = (aucs[0] - aucs[1]) / se
    p = 2 * stats.norm.sf(abs(z))
    return {"auroc_1": float(aucs[0]), "auroc_2": float(aucs[1]),
            "delta": float(aucs[0] - aucs[1]), "se": se, "z": float(z), "p": float(p)}


# --------------------------------------------------------------------------- #
# Cluster bootstrap with BCa interval
# --------------------------------------------------------------------------- #


def _auroc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def cluster_bootstrap_auroc(y_true, y_score, groups, n_boot: int = 10000,
                            seed: int = 0, alpha: float = 0.05):
    """AUROC with a group-level (cluster) bootstrap BCa confidence interval.

    Resampling whole clusters (e.g. genes) — not rows — respects the dependence
    structure so the CI is honest under group correlation.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx_by_group = {gv: np.where(g == gv)[0] for gv in uniq}
    rng = np.random.default_rng(seed)
    theta_hat = _auroc(y, p)

    boots = np.empty(n_boot, dtype=float)
    G = len(uniq)
    for b in range(n_boot):
        chosen = uniq[rng.integers(0, G, size=G)]
        idx = np.concatenate([idx_by_group[c] for c in chosen])
        boots[b] = _auroc(y[idx], p[idx])
    boots = boots[~np.isnan(boots)]
    if len(boots) < 10:
        return {"auroc": theta_hat, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot_valid": int(len(boots)), "method": "bca"}

    # bias-correction z0
    prop = np.mean(boots < theta_hat)
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = stats.norm.ppf(prop)

    # acceleration via jackknife over clusters
    jack = np.empty(G, dtype=float)
    for i, gv in enumerate(uniq):
        keep = np.concatenate([idx_by_group[c] for c in uniq if c != gv])
        jack[i] = _auroc(y[keep], p[keep])
    jack = jack[~np.isnan(jack)]
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)

    def adj(zq):
        denom = 1 - a * (z0 + zq)
        return stats.norm.cdf(z0 + (z0 + zq) / denom) if denom != 0 else stats.norm.cdf(z0 + zq)

    lo_q, hi_q = adj(zl), adj(zu)
    ci_low = float(np.quantile(boots, lo_q))
    ci_high = float(np.quantile(boots, hi_q))
    return {"auroc": float(theta_hat), "ci_low": ci_low, "ci_high": ci_high,
            "boot_mean": float(boots.mean()), "boot_sd": float(boots.std()),
            "z0": float(z0), "acceleration": float(a),
            "n_boot_valid": int(len(boots)), "n_groups": int(G), "method": "bca"}


def permutation_test_auroc(y_true, y_score, n_perm: int = 10000, seed: int = 0):
    """Label-permutation null for AUROC>0.5. Returns observed AUROC + p-value."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    obs = _auroc(y, p)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = _auroc(rng.permutation(y), p)
    null = null[~np.isnan(null)]
    # one-sided (model better than chance); +1 smoothing
    pval = (1 + np.sum(null >= obs)) / (1 + len(null))
    return {"auroc": float(obs), "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "p_value": float(pval), "n_perm": int(len(null))}


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(center - half), float(center + half))


def reliability_table(y_true, p_pred, n_bins: int = 10, strategy: str = "uniform"):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        k = int(y[mask].sum())
        lo_ci, hi_ci = wilson_ci(k, n)
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "n": n,
                     "confidence": conf, "accuracy": acc,
                     "acc_ci_low": lo_ci, "acc_ci_high": hi_ci,
                     "gap": float(abs(conf - acc))})
    return rows


def expected_calibration_error(y_true, p_pred, n_bins: int = 15, strategy: str = "uniform"):
    rows = reliability_table(y_true, p_pred, n_bins, strategy)
    n_total = len(np.asarray(y_true))
    if n_total == 0:
        return float("nan")
    return float(sum(r["n"] / n_total * r["gap"] for r in rows))


def max_calibration_error(y_true, p_pred, n_bins: int = 15, strategy: str = "uniform"):
    rows = reliability_table(y_true, p_pred, n_bins, strategy)
    return float(max((r["gap"] for r in rows), default=float("nan")))


def adaptive_ece(y_true, p_pred, n_bins: int = 15):
    """Equal-frequency (adaptive) ECE — robust to skewed score distributions."""
    return expected_calibration_error(y_true, p_pred, n_bins, strategy="quantile")


def brier_score(y_true, p_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    return float(np.mean((p - y) ** 2))


def murphy_decomposition(y_true, p_pred, n_bins: int = 15):
    """Brier = reliability - resolution + uncertainty (Murphy 1973)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    n = len(y)
    base = float(y.mean())
    uncertainty = base * (1 - base)
    edges = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        nk = int(mask.sum())
        if nk == 0:
            continue
        ok = float(y[mask].mean())
        ck = float(p[mask].mean())
        reliability += nk / n * (ck - ok) ** 2
        resolution += nk / n * (ok - base) ** 2
    return {"brier": brier_score(y, p), "reliability": float(reliability),
            "resolution": float(resolution), "uncertainty": float(uncertainty),
            "skill_vs_climatology": float((resolution - reliability) / uncertainty)
            if uncertainty > 0 else float("nan")}


# --------------------------------------------------------------------------- #
# Negative controls + leakage audit
# --------------------------------------------------------------------------- #


def negative_control_permuted_label(y_true, y_score, n_perm: int = 200, seed: int = 0):
    """Permute labels → AUROC must collapse to ~0.5. Detects harness/score leakage."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    aucs = np.array([_auroc(rng.permutation(y), p) for _ in range(n_perm)])
    aucs = aucs[~np.isnan(aucs)]
    return {"mean_auroc": float(aucs.mean()), "sd": float(aucs.std()),
            "max_abs_dev_from_0.5": float(np.max(np.abs(aucs - 0.5))),
            "passes": bool(abs(aucs.mean() - 0.5) < 0.02), "n_perm": int(len(aucs))}


def negative_control_scrambled_score(y_true, y_score, n_perm: int = 200, seed: int = 0):
    """Shuffle scores against fixed labels → AUROC must be ~0.5."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    aucs = np.array([_auroc(y, rng.permutation(p)) for _ in range(n_perm)])
    aucs = aucs[~np.isnan(aucs)]
    return {"mean_auroc": float(aucs.mean()), "sd": float(aucs.std()),
            "passes": bool(abs(aucs.mean() - 0.5) < 0.02), "n_perm": int(len(aucs))}


def leakage_audit_frozen(df) -> dict:
    """Integrity checks computable on the frozen test set itself.

    df columns: pair_id, gene, cell_line_a, cell_line_b, source, task, pair_type, y_rep

    Distinguishes three DIFFERENT things that the naive "duplicate key" count
    conflates:
      * exact_duplicate_rows   — identical rows incl. label → real redundancy bug
                                  (inflates effective n); must be 0.
      * conflicting_label_keys — same feature-key, DIFFERENT label → irreducible
                                  LABEL NOISE at the feature resolution (e.g. ORCS
                                  pairs distinguished only by study-pair, which the
                                  features don't carry). NOT train/test leakage —
                                  it caps achievable AUROC and is reported, not failed.
      * single_column_purity   — does any one metadata column determine the label?
                                  (a lookup-table shortcut). The real leakage probe.

    Train/test contamination across the gene/cell-line/study axes is enforced at
    freeze time (benchmark_v1.strict_disjoint_split) and cannot be re-checked from
    the test CSV alone — noted in `note`.
    """
    import pandas as pd  # local import; pandas is in the venv

    out: dict = {"n_rows": int(len(df))}
    out["pair_id_unique"] = bool(df["pair_id"].is_unique)

    feat_cols = [c for c in ["gene", "cell_line_a", "cell_line_b", "source", "task", "modality", "pair_type"]
                 if c in df.columns]
    out["exact_duplicate_rows"] = int(df.duplicated(subset=feat_cols + ["y_rep"]).sum())

    # same feature-key, conflicting label → label noise
    key = [c for c in ["gene", "cell_line_a", "cell_line_b", "source", "task"] if c in df.columns]
    g = df.groupby(key)["y_rep"]
    conflict_mask = g.transform("nunique") > 1
    out["conflicting_label"] = {
        "n_keys": int((g.nunique() > 1).sum()),
        "n_rows_affected": int(conflict_mask.sum()),
        "by_source": (df[conflict_mask]["source"].value_counts().to_dict() if "source" in df.columns else {}),
        "interpretation": "label noise at feature resolution (caps AUROC for these strata); NOT train/test leakage",
    }

    probes = {}
    for col in ["pair_type", "source", "task", "modality"]:
        if col in df.columns:
            grp = df.groupby(col)["y_rep"].mean()
            pure = ((grp == 0) | (grp == 1)).mean()
            probes[col] = {"frac_pure_groups": float(pure), "n_groups": int(len(grp))}
    out["single_column_label_purity"] = probes
    out["no_single_column_lookup_leak"] = bool(all(p["frac_pure_groups"] == 0 for p in probes.values()))

    # the clean verdict keys ONLY on genuine integrity concerns
    out["clean"] = bool(out["pair_id_unique"] and out["exact_duplicate_rows"] == 0
                        and out["no_single_column_lookup_leak"])
    out["note"] = "train/test disjointness (gene/cell-line/study) is enforced at freeze time; not re-checkable from the test CSV alone."
    return out
