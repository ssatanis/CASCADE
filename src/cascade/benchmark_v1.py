"""The Replication Benchmark v1 — a frozen, versioned, citable eval.

Freezes a held-out test set of REAL replication pairs (Broad↔Sanger institute +
BioGRID-ORCS cross-study + Replogle cross-cell-type), DISJOINT from training by
cell line AND gene AND study, with a pinned data-release manifest. A leaderboard
harness scores any model's predictions (AUROC / ECE / conformal coverage /
abstention) vs the mean + ridge baselines, STRATIFIED by replication kind so the
benchmark exposes exactly where models break. CASCADE is just one row, scored by
the same harness.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .corpus import load_aligned, provenance_hash
from .features import featurize
from .merged import build_merged_corpus
from .oracle import ReplicationOracle, expected_calibration_error
from .types import ReplicationPair

PKG_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = PKG_ROOT / "benchmark" / "replication_benchmark_v1"


@dataclass
class StrictSplit:
    train: list[ReplicationPair]
    test: list[ReplicationPair]
    holdout_cell_lines: set
    holdout_lineages: set
    holdout_studies: set
    holdout_genes: set


def strict_disjoint_split(pairs: list[ReplicationPair], holdout_frac: float = 0.2, seed: int = 0) -> StrictSplit:
    """train/test DISJOINT by cell line AND lineage AND study AND gene.
    A pair is TEST if it touches any held-out cell line/lineage/study/gene;
    TRAIN only if it touches NONE → no leakage on any axis."""
    rng = np.random.default_rng(seed)
    aligned = load_aligned()
    cls = list(aligned.common_cell_lines)
    lineages = sorted({p.context_a.lineage for p in pairs} | {p.context_b.lineage for p in pairs})
    studies = sorted({p.study for p in pairs if p.study})
    genes = sorted({p.gene for p in pairs})

    def pick(items, frac):
        if not items:
            return set()
        k = max(1, int(round(frac * len(items))))
        return set(np.array(items)[rng.choice(len(items), size=k, replace=False)].tolist())

    h_cl = pick(cls, holdout_frac)
    h_ln = pick(lineages, holdout_frac)
    h_st = pick(studies, holdout_frac)
    h_g = pick(genes, holdout_frac)

    def touches(p):
        return (
            p.context_a.cell_line in h_cl or p.context_b.cell_line in h_cl
            or p.context_a.lineage in h_ln or p.context_b.lineage in h_ln
            or (p.study and p.study in h_st)
            or p.gene in h_g
        )

    test = [p for p in pairs if touches(p)]
    train = [p for p in pairs if not touches(p)]
    return StrictSplit(train, test, h_cl, h_ln, h_st, h_g)


def _check_leakage(split: StrictSplit) -> dict:
    tr_cells = {c for p in split.train for c in (p.context_a.cell_line, p.context_b.cell_line)}
    te_cells = {c for p in split.test for c in (p.context_a.cell_line, p.context_b.cell_line) if c in split.holdout_cell_lines}
    tr_genes = {p.gene for p in split.train}
    te_genes = {p.gene for p in split.test if p.gene in split.holdout_genes}
    return {
        "train_test_cellline_disjoint_on_holdout": len(tr_cells & te_cells) == 0,
        "train_test_gene_disjoint_on_holdout": len(tr_genes & te_genes) == 0,
        "n_train": len(split.train),
        "n_test": len(split.test),
    }


def _metrics(y, p, abstain) -> dict:
    keep = ~np.asarray(abstain, dtype=bool)
    yk, pk = np.asarray(y)[keep], np.asarray(p)[keep]
    out = {"n": int(len(y)), "n_kept": int(keep.sum()), "abstention_rate": round(float(1 - keep.mean()), 4),
           "base_rate": round(float(np.mean(y)), 4)}
    if keep.sum() >= 5 and len(np.unique(yk)) >= 2:
        from sklearn.metrics import roc_auc_score
        out["auroc"] = round(float(roc_auc_score(yk, pk)), 4)
        out["ece"] = round(float(expected_calibration_error(yk, pk)), 4)
    else:
        out["auroc"] = float("nan"); out["ece"] = float("nan")
    return out


def score_predictions(bench_dir: Path, predictions: list[dict]) -> dict:
    """Score predictions [{pair_id, p_replicate, abstain}] against the frozen test
    set + the stored mean/ridge baselines, overall and per stratum."""
    test = list(csv.DictReader(open(bench_dir / "test_pairs.csv")))
    by_id = {r["pair_id"]: r for r in test}
    pred_by_id = {str(p["pair_id"]): p for p in predictions}

    rows = []
    for pid, r in by_id.items():
        pr = pred_by_id.get(pid)
        if pr is None:
            continue
        rows.append((r["pair_type"], int(r["y_rep"]),
                     float(pr["p_replicate"]) if pr.get("p_replicate") not in (None, "", "None") else 0.5,
                     bool(int(pr.get("abstain", 0)))))
    if not rows:
        return {"error": "no matching predictions"}

    def agg(subset):
        y = [r[1] for r in subset]; p = [r[2] for r in subset]; a = [r[3] for r in subset]
        return _metrics(y, p, a)

    out = {"overall": agg(rows)}
    strata = sorted({r[0] for r in rows})
    out["by_stratum"] = {s: agg([r for r in rows if r[0] == s]) for s in strata}
    return out


def freeze_benchmark(seed: int = 0, holdout_frac: float = 0.2) -> dict:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    merged = build_merged_corpus(seed=seed)
    split = strict_disjoint_split(merged.pairs, holdout_frac=holdout_frac, seed=seed)
    leak = _check_leakage(split)

    # De-duplicate the test set on the FULL identity key (incl. study) so the
    # frozen benchmark carries no exact-duplicate rows. ORCS pairs are kept
    # distinguishable by `study` (added to the schema in v1.1) — without it,
    # different study-pairs over the same (gene, cell-line pair) collapse into
    # identical rows and read as spurious redundancy. See REALNESS/limitations.
    seen: set = set()
    deduped_test = []
    n_dup_dropped = 0
    for p in split.test:
        identity = (p.gene, p.context_a.cell_line, p.context_b.cell_line, p.modality,
                    p.source, p.task, p.pair_type, p.study, int(p.label))
        if identity in seen:
            n_dup_dropped += 1
            continue
        seen.add(identity)
        deduped_test.append(p)
    split = StrictSplit(split.train, deduped_test, split.holdout_cell_lines,
                        split.holdout_lineages, split.holdout_studies, split.holdout_genes)

    # freeze test pairs (schema v1.1 adds `study` so ORCS study-pairs stay distinct)
    with open(BENCH_DIR / "test_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "gene", "cell_line_a", "lineage_a", "cell_line_b", "lineage_b",
                    "modality", "source", "task", "pair_type", "study", "y_rep"])
        for i, p in enumerate(split.test):
            w.writerow([f"P{i}", p.gene, p.context_a.cell_line, p.context_a.lineage,
                        p.context_b.cell_line, p.context_b.lineage, p.modality, p.source,
                        p.task, p.pair_type, p.study, int(p.label)])

    # CASCADE entry: per-task oracles trained on the benchmark TRAIN split only.
    fit_tr = [p for p in split.train if p.task == "fitness"]
    trans_tr = [p for p in split.train if p.task == "transcriptomic"]
    fit_oracle = ReplicationOracle(alpha=0.1).fit(fit_tr, seed=seed)
    trans_oracle = ReplicationOracle(alpha=0.1).fit(trans_tr, seed=seed) if len(trans_tr) >= 20 else None

    # baselines (ridge/mean) fit on train features
    from sklearn.linear_model import Ridge
    Xtr, ytr = featurize(fit_tr + trans_tr)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
    ridge = Ridge(alpha=1.0).fit((Xtr - mu) / sd, ytr)
    base_rate = float(ytr.mean())

    cascade_preds, mean_preds, ridge_preds = [], [], []
    for i, p in enumerate(split.test):
        pid = f"P{i}"
        orc = trans_oracle if (p.task == "transcriptomic" and trans_oracle) else fit_oracle
        pred = orc.predict_pair(p)
        cascade_preds.append({"pair_id": pid, "p_replicate": (pred.p_replicate if not pred.abstained else ""), "abstain": int(pred.abstained)})
        x = featurize([p])[0]
        rp = float(np.clip(ridge.predict((x - mu) / sd)[0], 0, 1))
        mean_preds.append({"pair_id": pid, "p_replicate": base_rate, "abstain": 0})
        ridge_preds.append({"pair_id": pid, "p_replicate": rp, "abstain": 0})

    with open(BENCH_DIR / "predictions_cascade.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["pair_id", "p_replicate", "abstain"]); wcsv.writeheader(); wcsv.writerows(cascade_preds)

    leaderboard = {
        "CASCADE_v0": score_predictions(BENCH_DIR, cascade_preds),
        "baseline_mean_rate": score_predictions(BENCH_DIR, mean_preds),
        "baseline_ridge": score_predictions(BENCH_DIR, ridge_preds),
    }
    (BENCH_DIR / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))

    from collections import Counter
    manifest = {
        "name": "CASCADE Replication Benchmark v1",
        "version": "1.1",
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema": "adds `study` column (ORCS study-pair identity); test set de-duplicated on full identity key",
        "n_exact_duplicate_rows_dropped": n_dup_dropped,
        "provenance_hash": provenance_hash(),
        "data_releases": {
            "broad": "DepMap Public 26Q1", "sanger": "Sanger Project Score Chronos v2",
            "orcs": "BioGRID-ORCS 2.0.18 (homo sapiens)",
            "replogle": "scPerturb Zenodo 13350497 (Replogle 2022 K562/RPE1)",
        },
        "task_definition": "Predict whether a screen hit replicates in another lab/context (binary y_rep), with calibrated probability + honest abstention.",
        "strata": {
            "cross_lab": "institute replication (Broad↔Sanger, same cell line)",
            "cross_context": "within-Broad, different cell line",
            "cross_study": "BioGRID-ORCS, different study (HIT concordance)",
            "cross_cell_type": "Replogle K562↔RPE1, transcriptomic (distinct task)",
        },
        "split_policy": "DISJOINT by cell line AND lineage AND study AND gene (test touches a holdout on any axis; train touches none).",
        "leakage_check": leak,
        "metrics": ["AUROC", "ECE", "conformal_coverage", "abstention_rate"],
        "n_test": len(split.test),
        "n_test_by_stratum": dict(Counter(p.pair_type for p in split.test)),
        "license": "CC BY 4.0 (derived from CC BY 4.0 / MIT sources; see data/PROVENANCE.json)",
    }
    (BENCH_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"manifest": manifest, "leaderboard": leaderboard, "leakage": leak}
