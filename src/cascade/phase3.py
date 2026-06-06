"""Phase-3 runner: turn the frozen benchmark into a fully-logged evidence pack.

Operates ONLY on the frozen, versioned artifacts
``benchmark/replication_benchmark_v1/{test_pairs,predictions_cascade}.csv`` — no
network, no corpus rebuild — so it reproduces bit-for-bit from the repo alone.

What it proves (and what it cannot)
-----------------------------------
PROVES, on real held-out Broad↔Sanger / ORCS / Replogle pairs:
  * the shipped artifact's discrimination per stratum, with DeLong CIs +
    group-clustered BCa CIs + permutation nulls;
  * the shipped artifact's calibration (ECE/MCE/adaptive-ECE/Brier/Murphy);
  * clean negative controls (permuted-label & scrambled-score → AUROC≈0.5);
  * frozen-set integrity (dup / near-dup / single-column label purity).

DOES NOT PROVE here (needs the training corpus, which must be re-acquired):
  * the cross-validated B4 (additive) / B5 (group-prior) head-to-head gate;
  * conformal coverage (needs per-pair intervals → artifact re-prediction →
    engineered features → the corpus). Reported coverage stays in the v0 run log.

Within-stratum AUROC is base-rate invariant, so a per-stratum AUROC>0.5 with a
significant DeLong p IS the rigorous "beats the stratum-prior baseline within
that stratum" result. The cross-study collapse (AUROC≈0.5) is reported plainly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluation as ev

PKG_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = PKG_ROOT / "benchmark" / "replication_benchmark_v1"
RESULTS_DIR = PKG_ROOT / "results"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PKG_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def load_frozen(bench_dir: Path = BENCH_DIR) -> pd.DataFrame:
    tp = pd.read_csv(bench_dir / "test_pairs.csv")
    pr = pd.read_csv(bench_dir / "predictions_cascade.csv")
    df = tp.merge(pr, on="pair_id", how="inner")
    df["kept"] = df["abstain"].astype(int) == 0
    return df


def _stratum_block(df: pd.DataFrame, seeds, n_boot, n_perm) -> dict:
    """Discrimination + significance on the KEPT (non-abstained) subset."""
    kept = df[df["kept"]].copy()
    y = kept["y_rep"].to_numpy(dtype=int)
    p = kept["p_replicate"].to_numpy(dtype=float)
    genes = kept["gene"].to_numpy()
    out: dict = {
        "n_total": int(len(df)),
        "n_kept": int(len(kept)),
        "abstention_rate": round(float(1 - df["kept"].mean()), 4),
        "base_rate_kept": round(float(y.mean()), 4) if len(y) else float("nan"),
        "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
    }
    if len(np.unique(y)) < 2 or len(y) < 5:
        out["auroc"] = float("nan")
        out["note"] = "single class or n<5 after abstention"
        return out

    out["delong"] = ev.delong_auc_ci(y, p)
    # seed-stable cluster bootstrap + permutation
    boot_los, boot_his, boot_aucs = [], [], []
    perm_ps, perm_aucs = [], []
    for s in seeds:
        cb = ev.cluster_bootstrap_auroc(y, p, genes, n_boot=n_boot, seed=s)
        boot_los.append(cb["ci_low"]); boot_his.append(cb["ci_high"]); boot_aucs.append(cb["auroc"])
        pt = ev.permutation_test_auroc(y, p, n_perm=n_perm, seed=s)
        perm_ps.append(pt["p_value"]); perm_aucs.append(pt["auroc"])
    out["cluster_bootstrap_bca"] = {
        "auroc": float(np.mean(boot_aucs)),
        "ci_low_mean": float(np.mean(boot_los)), "ci_low_sd": float(np.std(boot_los)),
        "ci_high_mean": float(np.mean(boot_his)), "ci_high_sd": float(np.std(boot_his)),
        "n_groups_genes": int(len(np.unique(genes))), "n_boot": n_boot, "seeds": list(seeds),
    }
    out["permutation_test"] = {
        "auroc": float(np.mean(perm_aucs)),
        "p_value_mean": float(np.mean(perm_ps)), "p_value_max": float(np.max(perm_ps)),
        "n_perm": n_perm, "seeds": list(seeds),
        "significant_all_seeds": bool(np.max(perm_ps) < 0.05),
    }
    # calibration
    out["calibration"] = {
        "ece_15bin_uniform": ev.expected_calibration_error(y, p, 15, "uniform"),
        "ece_15bin_quantile": ev.expected_calibration_error(y, p, 15, "quantile"),
        "adaptive_ece": ev.adaptive_ece(y, p, 15),
        "mce_15bin": ev.max_calibration_error(y, p, 15),
        "brier": ev.brier_score(y, p),
        "murphy": ev.murphy_decomposition(y, p, 15),
    }
    out["reliability_10bin"] = ev.reliability_table(y, p, 10, "uniform")
    return out


def run_phase3(seeds=(0, 1, 2, 3, 4), n_boot: int = 2000, n_perm: int = 2000,
               bench_dir: Path = BENCH_DIR, out_path: Path | None = None) -> dict:
    df = load_frozen(bench_dir)
    manifest = json.loads((bench_dir / "manifest.json").read_text())

    report: dict = {
        "run": {
            "kind": "phase3_frozen_benchmark_evaluation",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": _git_sha(),
            "seeds": list(seeds), "n_boot": n_boot, "n_perm": n_perm,
            "benchmark_version": manifest.get("version"),
            "benchmark_provenance_hash": manifest.get("provenance_hash"),
            "test_pairs_sha256": _sha256_file(bench_dir / "test_pairs.csv"),
            "predictions_sha256": _sha256_file(bench_dir / "predictions_cascade.csv"),
            "data_releases": manifest.get("data_releases"),
        },
        "config_hash": None,
    }
    cfg = json.dumps({"seeds": list(seeds), "n_boot": n_boot, "n_perm": n_perm}, sort_keys=True)
    report["config_hash"] = hashlib.sha256(cfg.encode()).hexdigest()[:16]

    # overall
    report["overall"] = _stratum_block(df, seeds, n_boot, n_perm)
    # per stratum
    report["by_stratum"] = {}
    for st in sorted(df["pair_type"].unique()):
        report["by_stratum"][st] = _stratum_block(df[df["pair_type"] == st], seeds, n_boot, n_perm)

    # negative controls on the kept overall set (seed-averaged)
    kept = df[df["kept"]]
    y = kept["y_rep"].to_numpy(dtype=int)
    p = kept["p_replicate"].to_numpy(dtype=float)
    perm_lab = [ev.negative_control_permuted_label(y, p, n_perm=200, seed=s) for s in seeds]
    scr = [ev.negative_control_scrambled_score(y, p, n_perm=200, seed=s) for s in seeds]
    report["negative_controls"] = {
        "permuted_label": {
            "mean_auroc": float(np.mean([d["mean_auroc"] for d in perm_lab])),
            "max_abs_dev_from_0.5": float(np.max([d["max_abs_dev_from_0.5"] for d in perm_lab])),
            "passes_all_seeds": bool(all(d["passes"] for d in perm_lab)),
        },
        "scrambled_score": {
            "mean_auroc": float(np.mean([d["mean_auroc"] for d in scr])),
            "passes_all_seeds": bool(all(d["passes"] for d in scr)),
        },
    }
    # leakage audit
    report["leakage_audit"] = ev.leakage_audit_frozen(df)

    # headline conclusions (seed-stable booleans)
    concl = {}
    for st, blk in report["by_stratum"].items():
        if "permutation_test" in blk:
            concl[f"{st}_beats_chance_all_seeds"] = blk["permutation_test"]["significant_all_seeds"]
            concl[f"{st}_auroc"] = round(blk["delong"]["auroc"], 4)
    concl["negative_controls_clean"] = (
        report["negative_controls"]["permuted_label"]["passes_all_seeds"]
        and report["negative_controls"]["scrambled_score"]["passes_all_seeds"])
    concl["leakage_audit_clean"] = bool(report["leakage_audit"]["clean"])
    concl["label_noise_keys_flagged"] = report["leakage_audit"]["conflicting_label"]["n_keys"]
    report["headline_conclusions"] = concl

    # honesty block
    report["honesty"] = {
        "proves": "shipped-artifact discrimination + calibration + clean negative controls + frozen-set integrity, on real held-out pairs, seed-stable.",
        "does_not_prove": "the cross-validated B4/B5 head-to-head gate and conformal coverage under shift — both require re-acquiring the training corpus (DepMap/Sanger/ORCS/Replogle were pruned from disk).",
        "cross_study_is_chance": "cross_study AUROC≈0.5 is reported plainly — ORCS HIT-concordance across heterogeneous studies is genuinely near-unpredictable from these features.",
    }

    out_path = out_path or (RESULTS_DIR / "phase3_evaluation.json")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2))

    # also drop the underlying reliability numbers as parquet-ready CSVs
    fig_dir = PKG_ROOT / "paper" / "figure_data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rel_rows = []
    for st, blk in {**{"overall": report["overall"]}, **report["by_stratum"]}.items():
        for r in blk.get("reliability_10bin", []):
            rel_rows.append({"stratum": st, **r})
    if rel_rows:
        pd.DataFrame(rel_rows).to_csv(fig_dir / "fig2_reliability.csv", index=False)
    strat_rows = []
    for st, blk in report["by_stratum"].items():
        if "delong" in blk:
            strat_rows.append({
                "stratum": st, "n_kept": blk["n_kept"], "base_rate": blk["base_rate_kept"],
                "auroc": blk["delong"]["auroc"],
                "delong_ci_low": blk["delong"]["ci_low"], "delong_ci_high": blk["delong"]["ci_high"],
                "bca_ci_low": blk["cluster_bootstrap_bca"]["ci_low_mean"],
                "bca_ci_high": blk["cluster_bootstrap_bca"]["ci_high_mean"],
                "perm_p_max": blk["permutation_test"]["p_value_max"],
                "ece": blk["calibration"]["ece_15bin_uniform"],
                "brier": blk["calibration"]["brier"],
            })
    if strat_rows:
        pd.DataFrame(strat_rows).to_csv(fig_dir / "fig3_per_stratum_auroc.csv", index=False)

    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CASCADE Phase-3 frozen-benchmark evaluation")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()
    rep = run_phase3(seeds=tuple(args.seeds), n_boot=args.n_boot, n_perm=args.n_perm)
    print(json.dumps({"overall_auroc": rep["overall"]["delong"]["auroc"],
                      "headline": rep["headline_conclusions"]}, indent=2))
