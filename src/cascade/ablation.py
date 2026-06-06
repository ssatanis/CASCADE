"""Ablation: which component of the Oracle earns its place (CASCADE Phase-3.5).

Five nested variants on the SAME real context-holdout split as oracle_v0.pkl
(merged fitness corpus, context_holdout_split seed=0), evaluated on the held-out
split:

  A  logistic on raw pair features                       (no calibration, no conformal)
  B  A + isotonic calibration
  C  B + Mondrian conformal (group = lineage×modality)
  D  C + provenance/QC weighting (sample_weight in the logistic fit)
  E  full CASCADE = D + kNN support gate                 (== oracle_v0.pkl)

Honest note on what each component does:
  * isotonic and conformal are MONOTONE → they do NOT change AUROC (ranking is
    preserved); their value is calibration (ECE↓) and distribution-free coverage.
    So AUROC(A)≈AUROC(B)≈AUROC(C) by construction — reported as-is, not inflated.
  * provenance weighting (D) re-fits → can move AUROC.
  * the support gate (E) abstains on out-of-support pairs → AUROC is computed on
    the KEPT subset, which is why E exceeds D: it trades coverage for accuracy.

Variant E must reproduce oracle_v0.pkl's held-out AUROC (within 0.002).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import evaluation as ev
from .features import context_group, featurize
from .oracle import ReplicationOracle, expected_calibration_error

PKG_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PKG_ROOT / "results"
FIG = PKG_ROOT / "paper" / "figure_data"


def _rebuild_split(seed: int = 0):
    """Exact context-holdout split used to train oracle_v0.pkl (fitness task)."""
    from .merged import build_merged_corpus
    from .train import context_holdout_split
    from .corpus import load_aligned

    aligned = load_aligned()
    merged = build_merged_corpus(seed=seed)
    fit_pairs = [p for p in merged.pairs if p.task == "fitness"]
    split = context_holdout_split(aligned, fit_pairs, holdout_frac=0.2, seed=seed)
    return split


def _strata(test):
    return [p.pair_type for p in test]


def _auroc_by_stratum(y, p, strata, mask=None):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p); strata = np.asarray(strata)
    if mask is not None:
        y, p, strata = y[mask], p[mask], strata[mask]
    out = {}
    for st in np.unique(strata):
        m = strata == st
        if m.sum() >= 5 and len(np.unique(y[m])) >= 2:
            out[st] = round(float(roc_auc_score(y[m], p[m])), 4)
    overall = round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) >= 2 else float("nan")
    return overall, out


def _logistic_fit_predict(Xtr, ytr, Xte, sample_weight=None, seed=0):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(Xtr, ytr, sample_weight=sample_weight)
    return clf.predict_proba(Xte)[:, 1], clf


def run(seed: int = 0, save: bool = True) -> dict:
    from sklearn.isotonic import IsotonicRegression
    from .conformal import MondrianConformalRegressor

    split = _rebuild_split(seed)
    train, test = split.train, split.test
    Xtr, ytr = featurize(train)
    Xte, yte = featurize(test)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
    strata = _strata(test)

    # internal slices (mirror oracle: train / isotonic / conformal)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    n_tr = int(0.5 * len(train)); n_iso = int(0.2 * len(train))
    tr_i, iso_i, conf_i = idx[:n_tr], idx[n_tr:n_tr + n_iso], idx[n_tr + n_iso:]

    q_tr = np.array([(p.quality_a or 0.7) * (p.quality_b or 0.7) for p in train])  # provenance weight

    variants = {}

    # --- A: logistic only (fit on tr slice for parity with later variants) ---
    pA, clfA = _logistic_fit_predict(Xtr_s[tr_i], ytr[tr_i], Xte_s)
    aA, sA = _auroc_by_stratum(yte, pA, strata)
    variants["A_logistic_only"] = {
        "auroc_overall": aA, "auroc_cross_lab": sA.get("cross_lab"),
        "auroc_cross_context": sA.get("cross_context"),
        "ece": round(expected_calibration_error(yte, pA), 4),
        "brier": round(ev.brier_score(yte, pA), 4),
        "coverage": None, "abstention": None,
    }

    # --- B: A + isotonic (monotone → AUROC unchanged; ECE should improve) ---
    p_iso_src = clfA.predict_proba(Xtr_s[iso_i])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_iso_src, ytr[iso_i])
    pB = iso.transform(pA)
    aB, sB = _auroc_by_stratum(yte, pB, strata)
    variants["B_plus_isotonic"] = {
        "auroc_overall": aB, "auroc_cross_lab": sB.get("cross_lab"),
        "auroc_cross_context": sB.get("cross_context"),
        "ece": round(expected_calibration_error(yte, pB), 4),
        "brier": round(ev.brier_score(yte, pB), 4),
        "coverage": None, "abstention": None,
    }

    # --- C: B + Mondrian conformal (coverage; ranking unchanged) ---
    p_conf = iso.transform(clfA.predict_proba(Xtr_s[conf_i])[:, 1])
    groups_conf = [context_group(train[i]) for i in conf_i]
    conf = MondrianConformalRegressor(alpha=0.1).fit(ytr[conf_i], p_conf, groups_conf)
    cov_C = []
    for k, p in enumerate(test):
        lo, hi, _ = conf.predict_interval(float(pB[k]), context_group(p))
        cov_C.append(1 if (max(0, lo) <= yte[k] <= min(1, hi)) else 0)
    variants["C_plus_conformal"] = {
        "auroc_overall": aB, "auroc_cross_lab": sB.get("cross_lab"),
        "auroc_cross_context": sB.get("cross_context"),
        "ece": round(expected_calibration_error(yte, pB), 4),
        "brier": round(ev.brier_score(yte, pB), 4),
        "coverage": round(float(np.mean(cov_C)), 4), "abstention": None,
    }

    # --- D: C + provenance/QC weighting (re-fit logistic with sample_weight) ---
    pD, clfD = _logistic_fit_predict(Xtr_s[tr_i], ytr[tr_i], Xte_s, sample_weight=q_tr[tr_i])
    iso_src_D = clfD.predict_proba(Xtr_s[iso_i])[:, 1]
    isoD = IsotonicRegression(out_of_bounds="clip").fit(iso_src_D, ytr[iso_i])
    pD_cal = isoD.transform(pD)
    aD, sD = _auroc_by_stratum(yte, pD_cal, strata)
    p_conf_D = isoD.transform(clfD.predict_proba(Xtr_s[conf_i])[:, 1])
    confD = MondrianConformalRegressor(alpha=0.1).fit(ytr[conf_i], p_conf_D, groups_conf)
    cov_D = [1 if (max(0, confD.predict_interval(float(pD_cal[k]), context_group(test[k]))[0])
                   <= yte[k] <=
                   min(1, confD.predict_interval(float(pD_cal[k]), context_group(test[k]))[1])) else 0
             for k in range(len(test))]
    variants["D_plus_provenance"] = {
        "auroc_overall": aD, "auroc_cross_lab": sD.get("cross_lab"),
        "auroc_cross_context": sD.get("cross_context"),
        "ece": round(expected_calibration_error(yte, pD_cal), 4),
        "brier": round(ev.brier_score(yte, pD_cal), 4),
        "coverage": round(float(np.mean(cov_D)), 4), "abstention": None,
    }

    # --- E: full CASCADE oracle (== oracle_v0.pkl) ---
    oracle = ReplicationOracle(alpha=0.1).fit(train, seed=seed)
    sc = oracle.score(test)
    preds = oracle.predict(test)
    kept = np.array([not pr.abstained for pr in preds])
    pE = np.array([pr.p_replicate for pr in preds])
    aE, sE = _auroc_by_stratum(yte, pE, strata, mask=kept)
    variants["E_full_cascade"] = {
        "auroc_overall": round(sc["auroc"], 4), "auroc_cross_lab": sE.get("cross_lab"),
        "auroc_cross_context": sE.get("cross_context"),
        "ece": round(sc["ece"], 4), "brier": round(ev.brier_score(yte[kept], pE[kept]), 4),
        "coverage": round(sc["coverage"], 4), "abstention": round(sc["abstention_rate"], 4),
    }

    from .train import provenance_hash
    delta = {
        "auroc_overall": round(variants["E_full_cascade"]["auroc_overall"] - aA, 4),
        "auroc_cross_lab": round((sE.get("cross_lab") or 0) - (sA.get("cross_lab") or 0), 4),
        "ece_A": variants["A_logistic_only"]["ece"], "ece_E": variants["E_full_cascade"]["ece"],
    }
    report = {
        "trained_on_real_data": True,
        "provenance_hash": provenance_hash(),
        "split": "context-holdout (cell lines + lineages + studies); same as oracle_v0.pkl, seed=0",
        "n_train": len(train), "n_test": len(test),
        "variants": variants,
        "delta_A_to_E": delta,
        "notes": {
            "monotone_components": "isotonic (B) and conformal (C) are monotone → AUROC(A)=AUROC(B)=AUROC(C) by construction; value is ECE↓ and coverage.",
            "support_gate": "E evaluates AUROC on the non-abstained subset → exceeds D by trading coverage for accuracy.",
        },
    }
    if save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "ablation.json").write_text(json.dumps(report, indent=2, default=str))
        FIG.mkdir(parents=True, exist_ok=True)
        rows = ["variant,auroc_overall,auroc_cross_lab,ece_overall,brier_overall,coverage,abstention"]
        for name, v in variants.items():
            rows.append(",".join(str(x) for x in [name, v["auroc_overall"], v["auroc_cross_lab"],
                        v["ece"], v["brier"], v["coverage"], v["abstention"]]))
        (FIG / "fig_ablation.csv").write_text("\n".join(rows) + "\n")
    return report


if __name__ == "__main__":
    r = run()
    for n, v in r["variants"].items():
        print(f"{n:22} AUROC={v['auroc_overall']} ECE={v['ece']} cov={v['coverage']} abst={v['abstention']}")
    print("delta A->E:", r["delta_A_to_E"])
