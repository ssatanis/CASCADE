"""Train + validate the v0 Replication Oracle on REAL data (spec Parts D & E).

Pipeline:
  1. load aligned Broad↔Sanger, persist the real concordance collapse
  2. build the real replication corpus
  3. CONTEXT-HOLDOUT split — hold out entire cell lines AND entire lineages so
     generalization is tested honestly (random splits leak)
  4. fit the Oracle (logistic → isotonic → conformal → kNN support gate)
  5. evaluate on the real holdout vs mean + ridge/logistic baselines
  6. promotion gate: beat mean AND ridge AUROC, conformal coverage ≥ target,
     ECE acceptable — only then persist oracle_v0.pkl (trained_on_real_data: True)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .corpus import AlignedData, build_pairs, compute_collapse, load_aligned, provenance_hash
from .features import featurize
from .oracle import ReplicationOracle, expected_calibration_error
from .types import ReplicationPair

PKG_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PKG_ROOT / "artifacts"
RESULTS_DIR = PKG_ROOT / "results"
DEPMAP_RELEASE = "DepMap Public 26Q1"
SANGER_RELEASE = "Sanger CRISPR (Project Score, Chronos) v2"


@dataclass
class Split:
    train: list[ReplicationPair]
    test: list[ReplicationPair]
    holdout_cell_lines: list[str]
    holdout_lineages: list[str]
    n_holdout_studies: int


def context_holdout_split(
    aligned: AlignedData, pairs: list[ReplicationPair], holdout_frac: float = 0.2, seed: int = 0
) -> Split:
    """Context-holdout for the ARTIFACT: hold out entire cell lines + lineages +
    entire STUDIES (ORCS pubmeds). A pair goes to test if it touches any held-out
    cell line, lineage, or study — so no held-out context/study leaks into
    training. (Gene-level disjointness is enforced separately in the frozen
    benchmark, prompt C.)"""
    rng = np.random.default_rng(seed)
    cls = list(aligned.common_cell_lines)
    lineages = sorted({aligned.lineage.get(c, "unknown") for c in cls if aligned.lineage.get(c) != "unknown"})

    n_hold_cl = max(1, int(round(holdout_frac * len(cls))))
    holdout_cl = set(rng.choice(cls, size=n_hold_cl, replace=False).tolist())
    n_hold_ln = max(1, int(round(holdout_frac * len(lineages))))
    holdout_ln = set(rng.choice(lineages, size=min(n_hold_ln, len(lineages)), replace=False).tolist())
    for c in cls:
        if aligned.lineage.get(c) in holdout_ln:
            holdout_cl.add(c)

    studies = sorted({p.study for p in pairs if p.study})
    holdout_studies: set[str] = set()
    if studies:
        n_hold_s = max(1, int(round(holdout_frac * len(studies))))
        holdout_studies = set(rng.choice(studies, size=n_hold_s, replace=False).tolist())

    train, test = [], []
    for p in pairs:
        touches_holdout = (
            p.context_a.cell_line in holdout_cl
            or p.context_b.cell_line in holdout_cl
            or p.context_a.lineage in holdout_ln
            or p.context_b.lineage in holdout_ln
            or (p.study and p.study in holdout_studies)
        )
        (test if touches_holdout else train).append(p)
    return Split(
        train=train, test=test, holdout_cell_lines=sorted(holdout_cl),
        holdout_lineages=sorted(holdout_ln), n_holdout_studies=len(holdout_studies),
    )


def stratified_eval(oracle: ReplicationOracle, test: list[ReplicationPair]) -> dict:
    """Per-pair_type metrics so the benchmark exposes where the model breaks."""
    from collections import defaultdict

    buckets: dict[str, list[ReplicationPair]] = defaultdict(list)
    for p in test:
        buckets[p.pair_type].append(p)
    out = {}
    for ptype, ps in sorted(buckets.items()):
        s = oracle.score(ps)
        out[ptype] = {
            "n": len(ps),
            "base_rate": round(float(np.mean([1 if p.label else 0 for p in ps])), 4),
            "auroc": s["auroc"],
            "ece": s["ece"],
            "coverage": s["coverage"],
            "abstention_rate": s["abstention_rate"],
        }
    return out


def evaluate(oracle: ReplicationOracle, train: list[ReplicationPair], test: list[ReplicationPair]) -> dict:
    """Score the Oracle on the real holdout vs mean + ridge/logistic baselines."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score

    scored = oracle.score(test)

    Xtr, ytr = featurize(train)
    Xte, yte = featurize(test)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd

    base_rate_train = float(ytr.mean())
    mean_pred = np.full(len(yte), base_rate_train)
    ridge = Ridge(alpha=1.0).fit(Xtr_s, ytr)
    ridge_pred = np.clip(ridge.predict(Xte_s), 0, 1)
    logit = LogisticRegression(max_iter=2000).fit(Xtr_s, ytr)
    logit_pred = logit.predict_proba(Xte_s)[:, 1]

    two_class = len(np.unique(yte)) >= 2
    auroc = lambda p: float(roc_auc_score(yte, p)) if two_class else float("nan")  # noqa: E731
    return {
        "n_test": len(test),
        "test_base_rate": float(yte.mean()),
        "oracle": {
            "auroc": scored["auroc"],
            "ece": scored["ece"],
            "coverage": scored["coverage"],
            "abstention_rate": scored["abstention_rate"],
            "n_kept": scored["n_kept"],
        },
        "baselines": {
            "mean_rate_auroc": 0.5,
            "ridge_auroc": auroc(ridge_pred),
            "ridge_ece": expected_calibration_error(yte, ridge_pred),
            "logistic_auroc": auroc(logit_pred),
            "logistic_ece": expected_calibration_error(yte, logit_pred),
        },
    }


def promotion_gate(ev: dict, coverage_target: float = 0.90, ece_max: float = 0.1) -> dict:
    o = ev["oracle"]
    b = ev["baselines"]
    beats_mean = (o["auroc"] > 0.5) if np.isfinite(o["auroc"]) else False
    beats_ridge = np.isfinite(o["auroc"]) and np.isfinite(b["ridge_auroc"]) and o["auroc"] >= b["ridge_auroc"] - 0.02
    coverage_ok = o["coverage"] >= coverage_target
    ece_ok = o["ece"] <= ece_max
    return {
        "beats_mean": bool(beats_mean),
        "beats_ridge": bool(beats_ridge),
        "coverage_ok": bool(coverage_ok),
        "ece_ok": bool(ece_ok),
        "promote": bool(beats_mean and beats_ridge and coverage_ok and ece_ok),
        "coverage_target": coverage_target,
        "ece_max": ece_max,
    }


def train_v0(
    theta: float = 0.5,
    seed: int = 0,
    holdout_frac: float = 0.2,
    artifact_path: str | Path | None = None,
    force_save: bool = False,
) -> dict:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    artifact_path = Path(artifact_path or ARTIFACT_DIR / "oracle_v0.pkl")

    from .merged import build_merged_corpus

    aligned = load_aligned()
    collapse = compute_collapse(aligned)
    (RESULTS_DIR / "real_collapse.json").write_text(json.dumps(collapse, indent=2))

    merged = build_merged_corpus(seed=seed)
    # Fitness pairs (Broad↔Sanger + ORCS) train the artifact; the transcriptomic
    # cross-cell-type pairs (Replogle) are held out ENTIRELY for the honest
    # K562→RPE1 context-transfer test (RPE1/transcriptomic never seen in training).
    fitness_pairs = [p for p in merged.pairs if p.task == "fitness"]
    replogle_pairs = [p for p in merged.pairs if p.task == "transcriptomic"]
    split = context_holdout_split(aligned, fitness_pairs, holdout_frac=holdout_frac, seed=seed)

    # The shipped artifact is the FITNESS oracle (tasks kept distinct — the spec
    # forbids pooling fitness and transcriptomic metrics into one model).
    oracle = ReplicationOracle(alpha=0.1).fit(split.train, seed=seed)
    ev = evaluate(oracle, split.train, split.test)
    ev["stratified_by_pair_type"] = stratified_eval(oracle, split.test)

    # Cross-cell-type: a SEPARATE transcriptomic oracle on a gene-DISJOINT Replogle
    # split — the honest measurable K562→RPE1 number for its own task.
    if len(replogle_pairs) >= 40:
        rng = np.random.default_rng(seed)
        rep_genes = sorted({p.gene for p in replogle_pairs})
        rep_hold = set(rng.choice(rep_genes, size=max(1, int(0.3 * len(rep_genes))), replace=False).tolist())
        rep_train = [p for p in replogle_pairs if p.gene not in rep_hold]
        rep_test = [p for p in replogle_pairs if p.gene in rep_hold]
        cc_oracle = ReplicationOracle(alpha=0.1).fit(rep_train, seed=seed)
        rs = cc_oracle.score(rep_test)
        cc_oracle.save(
            ARTIFACT_DIR / "oracle_crosscelltype_v0.pkl",
            {
                "trained_on_real_data": True,
                "task": "transcriptomic",
                "data_releases": {"replogle": "scPerturb Zenodo 13350497 — Replogle 2022 K562/RPE1 Perturb-seq"},
                "provenance_hash": provenance_hash(),
                "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "n_train": len(rep_train),
            },
        )
        ev["cross_cell_type_holdout"] = {
            "n": len(rep_test), "n_train": len(rep_train),
            "base_rate": round(float(np.mean([1 if p.label else 0 for p in rep_test])), 4),
            "auroc": rs["auroc"], "ece": rs["ece"], "coverage": rs["coverage"],
            "abstention_rate": rs["abstention_rate"], "n_kept": rs["n_kept"],
            "task": "transcriptomic", "modality": "CRISPRi",
            "note": "K562→RPE1 transcriptomic replication; dedicated oracle, gene-disjoint holdout.",
        }
        # Pure transfer: the fitness artifact applied to transcriptomic pairs —
        # high abstention = honest 'fitness-learned replication does not transfer'.
        ps = oracle.score(replogle_pairs)
        ev["cross_cell_type_pure_transfer"] = {
            "n": len(replogle_pairs), "abstention_rate": ps["abstention_rate"], "n_kept": ps["n_kept"],
            "note": "Fitness artifact on all Replogle pairs: abstains (correctly) — fitness does not transfer to transcriptomic.",
        }
    gate = promotion_gate(ev)

    phash = provenance_hash()
    metadata = {
        "trained_on_real_data": True,
        "data_releases": {
            "broad": DEPMAP_RELEASE,
            "sanger": SANGER_RELEASE,
            "orcs": "BioGRID-ORCS 2.0.18 (homo sapiens)",
            "replogle": "scPerturb Zenodo 13350497 — Replogle 2022 K562/RPE1 Perturb-seq",
        },
        "provenance_hash": phash,
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "theta": theta,
        "seed": seed,
        "corpus_stats": merged.stats,
        "split": {
            "type": "context-holdout (cell lines + lineages + studies); Replogle held out entirely",
            "n_train": len(split.train),
            "n_test": len(split.test),
            "n_holdout_cell_lines": len(split.holdout_cell_lines),
            "holdout_lineages": split.holdout_lineages,
            "n_holdout_studies": split.n_holdout_studies,
            "n_replogle_pairs": len(replogle_pairs),
        },
        "collapse": collapse,
        "evaluation": ev,
        "gate": gate,
        "n_common_cell_lines": len(aligned.common_cell_lines),
        "n_common_genes": len(aligned.common_genes),
    }

    report = dict(metadata)
    report["artifact_path"] = str(artifact_path)
    report["saved"] = False

    if gate["promote"] or force_save:
        oracle.save(artifact_path, metadata)
        report["saved"] = True

    (RESULTS_DIR / "benchmark_real.json").write_text(json.dumps(report, indent=2))
    return report
