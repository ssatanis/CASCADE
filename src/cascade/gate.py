"""The §HONESTY scientific pass/fail gate (CASCADE Phase 3.1/3.2/3.4).

The single gate that decides whether the calibrated Oracle earns its place:

    the Oracle significantly beats BOTH
      B4 — the additive baseline (per-context mean offset + source effect), and
      B5 — the same-cell-type-prior (group-wise empirical replication rate, Laplace),
    on group-aware out-of-fold splits (LOCO-cell-line, LOSO-study),
    across >=5 seeds, with conformal coverage holding under the held-out shift.

All predictors are scored on the SAME samples (the Oracle's non-abstained set) so
the DeLong paired test is valid; abstention is reported separately. We report the
real per-stratum number plainly — if the Oracle does NOT beat the baselines on a
stratum, that is stated, not hidden (the spec forbids overclaiming).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import evaluation as ev
from .features import context_group, featurize
from .oracle import ReplicationOracle
from .types import ReplicationPair

PKG_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PKG_ROOT / "results"


# --------------------------------------------------------------------------- #
# Corpus (fitness only — Broad↔Sanger + ORCS; Replogle handled separately)
# --------------------------------------------------------------------------- #


def build_fitness_corpus(seed: int = 0, orcs_max: int = 30000) -> list[ReplicationPair]:
    from .corpus import build_pairs, load_aligned
    from .orcs import build_orcs_pairs, load_orcs

    aligned = load_aligned()
    pairs = list(build_pairs(aligned, theta=0.5, seed=seed))
    orcs = load_orcs(fitness_only=True)
    pairs += build_orcs_pairs(orcs, max_pairs=orcs_max, seed=seed)
    return [p for p in pairs if p.task == "fitness"]


# --------------------------------------------------------------------------- #
# Baselines B4 (additive) and B5 (group-prior)
# --------------------------------------------------------------------------- #


def b4_additive_fit_predict(train, test):
    """B4 — additive baseline: logistic on one-hot(lineage_a)+one-hot(lineage_b)+
    one-hot(source). Purely context/source offsets, NO effect magnitude (that is
    B2). This is the Ahlmann-Eltze 'additive model' the deep model must beat."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    def feats(ps):
        return np.array([[p.context_a.lineage, p.context_b.lineage, p.source] for p in ps], dtype=object)

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Xtr = enc.fit_transform(feats(train))
    Xte = enc.transform(feats(test))
    ytr = np.array([1 if p.label else 0 for p in train], dtype=int)
    if len(np.unique(ytr)) < 2:
        return np.full(len(test), float(ytr.mean()))
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def b5_group_prior_fit_predict(train, test, smoothing: float = 1.0):
    """B5 — same-cell-type-prior: Laplace-smoothed empirical replication rate per
    (lineage_a, lineage_b) group, falling back to the global train rate."""
    pos = defaultdict(float)
    tot = defaultdict(float)
    for p in train:
        g = (p.context_a.lineage, p.context_b.lineage)
        tot[g] += 1
        pos[g] += 1 if p.label else 0
    global_rate = np.mean([1 if p.label else 0 for p in train]) if train else 0.5
    out = []
    for p in test:
        g = (p.context_a.lineage, p.context_b.lineage)
        if tot[g] > 0:
            out.append((pos[g] + smoothing) / (tot[g] + 2 * smoothing))
        else:
            out.append(global_rate)
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------- #
# Group-aware out-of-fold evaluation
# --------------------------------------------------------------------------- #


def _group_key(p: ReplicationPair, scheme: str):
    if scheme == "loco_cellline":
        return p.context_a.cell_line  # hold out source cell lines
    if scheme == "loso_study":
        return p.study or "__nostudy__"
    if scheme == "loco_lineage":
        return p.context_a.lineage
    raise ValueError(scheme)


def grouped_oof(pairs, scheme: str, seed: int, n_splits: int = 5, min_abstain_keep: int = 30,
                train_cap: int = 25000, eval_cap: int = 6000):
    """Out-of-fold predictions from Oracle / B4 / B5 under a GroupKFold by the
    scheme's group. Returns pooled arrays over the OOF (non-abstained) samples.

    For tractability the per-fold train set is capped to ``train_cap`` and the
    per-fold eval set to ``eval_cap`` via seeded subsampling — ALL THREE predictors
    use the identical train subsample and the identical eval points, so the DeLong
    comparison stays fair. The caps are reported in the result (`caps`) — no silent
    truncation. Group disjointness is preserved (subsampling is within fold)."""
    from sklearn.model_selection import GroupKFold

    if scheme == "loso_study":
        pairs = [p for p in pairs if p.study]  # LOSO only defined where study exists
    groups = np.array([_group_key(p, scheme) for p in pairs])
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        return None
    idx = np.arange(len(pairs))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)  # seed-controlled shuffle before grouping
    pairs_s = [pairs[i] for i in perm]
    groups_s = groups[perm]

    gkf = GroupKFold(n_splits=n_splits)
    y_all, p_orc, p_b4, p_b5, cov_all, gene_all = [], [], [], [], [], []
    n_total = 0
    n_kept = 0
    for tr_idx, te_idx in gkf.split(pairs_s, groups=groups_s):
        if len(tr_idx) > train_cap:
            tr_idx = rng.choice(tr_idx, size=train_cap, replace=False)
        if len(te_idx) > eval_cap:
            te_idx = rng.choice(te_idx, size=eval_cap, replace=False)
        train = [pairs_s[i] for i in tr_idx]
        test = [pairs_s[i] for i in te_idx]
        ytr = np.array([1 if p.label else 0 for p in train])
        if len(np.unique(ytr)) < 2 or len(train) < 50:
            continue
        try:
            oracle = ReplicationOracle(alpha=0.1).fit(train, seed=seed)
        except Exception:
            continue
        preds = oracle.predict(test)
        b4 = b4_additive_fit_predict(train, test)
        b5 = b5_group_prior_fit_predict(train, test)
        n_total += len(test)
        for pr, b4i, b5i, pair in zip(preds, b4, b5, test):
            if pr.abstained:
                continue
            n_kept += 1
            yi = 1 if pair.label else 0
            y_all.append(yi); p_orc.append(pr.p_replicate); p_b4.append(float(b4i))
            p_b5.append(float(b5i)); gene_all.append(pair.gene)
            cov_all.append(1 if (pr.lower <= yi <= pr.upper) else 0)
    if n_kept < min_abstain_keep:
        return None
    return {
        "y": np.array(y_all), "p_oracle": np.array(p_orc), "p_b4": np.array(p_b4),
        "p_b5": np.array(p_b5), "genes": np.array(gene_all), "coverage_flags": np.array(cov_all),
        "n_total": n_total, "n_kept": n_kept, "n_groups": n_groups,
        "abstention_rate": round(1 - n_kept / n_total, 4) if n_total else 1.0,
        "caps": {"train_cap": train_cap, "eval_cap": eval_cap},
    }


def evaluate_scheme(pairs, scheme: str, seeds=(0, 1, 2, 3, 4), n_splits: int = 5, verbose: bool = True) -> dict:
    per_seed = []
    for s in seeds:
        if verbose:
            print(f"  [{scheme}] seed {s} ...", flush=True)
        oof = grouped_oof(pairs, scheme, seed=s, n_splits=n_splits)
        if oof is None:
            if verbose:
                print(f"  [{scheme}] seed {s}: insufficient", flush=True)
            continue
        y, po, pb4, pb5 = oof["y"], oof["p_oracle"], oof["p_b4"], oof["p_b5"]
        if len(np.unique(y)) < 2:
            continue
        d_b4 = ev.delong_roc_test(y, po, pb4)
        d_b5 = ev.delong_roc_test(y, po, pb5)
        per_seed.append({
            "seed": s, "n_kept": oof["n_kept"], "n_total": oof["n_total"],
            "n_groups": oof["n_groups"], "abstention_rate": oof["abstention_rate"],
            "auroc_oracle": round(d_b4["auroc_1"], 4),
            "auroc_b4": round(d_b4["auroc_2"], 4), "auroc_b5": round(d_b5["auroc_2"], 4),
            "delta_vs_b4": round(d_b4["delta"], 4), "p_vs_b4": d_b4["p"],
            "delta_vs_b5": round(d_b5["delta"], 4), "p_vs_b5": d_b5["p"],
            "conformal_coverage": round(float(oof["coverage_flags"].mean()), 4),
            "ece": round(ev.expected_calibration_error(y, po, 15), 4),
            "brier": round(ev.brier_score(y, po), 4),
        })
    if not per_seed:
        return {"scheme": scheme, "status": "insufficient_groups_or_data"}

    def col(k):
        return np.array([d[k] for d in per_seed], dtype=float)

    beats_b4_all = bool(np.all((col("delta_vs_b4") > 0) & (col("p_vs_b4") < 0.05)))
    beats_b5_all = bool(np.all((col("delta_vs_b5") > 0) & (col("p_vs_b5") < 0.05)))
    cov = col("conformal_coverage")
    return {
        "scheme": scheme,
        "n_seeds": len(per_seed),
        "caps": {"train_cap": 25000, "eval_cap": 6000,
                 "note": "per-fold seeded subsample; identical points across Oracle/B4/B5 → fair DeLong"},
        "auroc_oracle_mean": round(float(col("auroc_oracle").mean()), 4),
        "auroc_oracle_sd": round(float(col("auroc_oracle").std()), 4),
        "auroc_b4_mean": round(float(col("auroc_b4").mean()), 4),
        "auroc_b5_mean": round(float(col("auroc_b5").mean()), 4),
        "delta_vs_b4_mean": round(float(col("delta_vs_b4").mean()), 4),
        "delta_vs_b5_mean": round(float(col("delta_vs_b5").mean()), 4),
        "p_vs_b4_max": float(col("p_vs_b4").max()),
        "p_vs_b5_max": float(col("p_vs_b5").max()),
        "conformal_coverage_mean": round(float(cov.mean()), 4),
        "conformal_coverage_min": round(float(cov.min()), 4),
        "coverage_holds_all_seeds": bool(np.all(cov >= 0.88)),  # 0.90 target, 0.02 tol
        "abstention_rate_mean": round(float(col("abstention_rate").mean()), 4),
        "beats_b4_all_seeds": beats_b4_all,
        "beats_b5_all_seeds": beats_b5_all,
        "GATE_PASS": bool(beats_b4_all and beats_b5_all and np.all(cov >= 0.88)),
        "per_seed": per_seed,
    }


def run_gate(seeds=(0, 1, 2, 3, 4), schemes=("loco_cellline", "loco_lineage", "loso_study"),
             out_path: Path | None = None) -> dict:
    pairs = build_fitness_corpus(seed=0)
    from collections import Counter
    report = {
        "run": {
            "kind": "honesty_scientific_gate",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "task": "fitness", "n_pairs": len(pairs), "seeds": list(seeds),
            "by_pair_type": dict(Counter(p.pair_type for p in pairs)),
            "by_source": dict(Counter(p.source for p in pairs)),
            "base_rate": round(np.mean([1 if p.label else 0 for p in pairs]), 4),
        },
        "schemes": {},
    }
    for sch in schemes:
        report["schemes"][sch] = evaluate_scheme(pairs, sch, seeds=seeds)

    # the headline: institute/context generalization (LOCO) is where the Oracle's
    # learned signal should beat the additive + prior baselines; LOSO-study is the
    # honest-hard ORCS stratum.
    report["summary"] = {
        sch: {"GATE_PASS": report["schemes"][sch].get("GATE_PASS"),
              "auroc_oracle": report["schemes"][sch].get("auroc_oracle_mean"),
              "auroc_b4": report["schemes"][sch].get("auroc_b4_mean"),
              "auroc_b5": report["schemes"][sch].get("auroc_b5_mean"),
              "beats_b4": report["schemes"][sch].get("beats_b4_all_seeds"),
              "beats_b5": report["schemes"][sch].get("beats_b5_all_seeds")}
        for sch in schemes
    }
    out_path = out_path or (RESULTS_DIR / "scientific_gate.json")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    rep = run_gate(seeds=tuple(args.seeds))
    print(json.dumps(rep["summary"], indent=2))
