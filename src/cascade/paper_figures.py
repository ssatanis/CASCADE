"""Render the CASCADE Frontier figures from the logged result JSONs.

Each figure ships alongside its underlying-numbers CSV in paper/figure_data/ (the
portable, reviewable artifact). If matplotlib is unavailable, the CSVs are still
written and the PNGs are skipped — the numbers, not the pixels, are the evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PKG_ROOT / "results"
FIG_DIR = PKG_ROOT / "paper" / "figure_data"
PNG_DIR = PKG_ROOT / "paper" / "figures"


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def fig3_per_stratum(plt) -> None:
    p = RESULTS / "phase3_evaluation.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    rows = []
    for st, b in r["by_stratum"].items():
        if "delong" not in b:
            continue
        d = b["delong"]
        rows.append({"stratum": st, "n_kept": b["n_kept"], "auroc": d["auroc"],
                     "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                     "ece": b["calibration"]["ece_15bin_uniform"], "brier": b["calibration"]["brier"]})
    df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG_DIR / "fig3_per_stratum_auroc.csv", index=False)
    if plt is None:
        return
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(df))
    err = np.vstack([df.auroc - df.ci_low, df.ci_high - df.auroc])
    ax.barh(y, df.auroc, xerr=err, color="#3b6", alpha=0.8, capsize=3)
    ax.axvline(0.5, ls="--", c="gray", label="chance")
    ax.set_yticks(y); ax.set_yticklabels(df.stratum)
    ax.set_xlabel("AUROC (DeLong 95% CI)"); ax.set_xlim(0, 1)
    ax.set_title("CASCADE replication AUROC by stratum (frozen benchmark v1, 5 seeds)")
    ax.legend(); fig.tight_layout()
    fig.savefig(PNG_DIR / "fig3_per_stratum_auroc.png", dpi=150)
    plt.close(fig)


def fig2_reliability(plt) -> None:
    p = RESULTS / "phase3_evaluation.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    rows = []
    for st in ["overall", "cross_lab", "cross_context", "cross_cell_type"]:
        blk = r["overall"] if st == "overall" else r["by_stratum"].get(st, {})
        for b in blk.get("reliability_10bin", []):
            rows.append({"stratum": st, **b})
    df = pd.DataFrame(rows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG_DIR / "fig2_reliability.csv", index=False)
    if plt is None or df.empty:
        return
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="gray", label="perfect")
    for st, g in df.groupby("stratum"):
        ax.plot(g.confidence, g.accuracy, marker="o", label=st, alpha=0.8)
    ax.set_xlabel("predicted P(replicate)"); ax.set_ylabel("observed replication rate")
    ax.set_title("Reliability (frozen benchmark v1)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PNG_DIR / "fig2_reliability.png", dpi=150)
    plt.close(fig)


def table1_gate(_plt) -> None:
    p = RESULTS / "scientific_gate.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    rows = []
    for sch, blk in r.get("schemes", {}).items():
        if "auroc_oracle_mean" not in blk:
            continue
        rows.append({"scheme": sch, "auroc_oracle": blk["auroc_oracle_mean"],
                     "auroc_b4": blk["auroc_b4_mean"], "auroc_b5": blk["auroc_b5_mean"],
                     "delta_vs_b4": blk["delta_vs_b4_mean"], "p_vs_b4_max": blk["p_vs_b4_max"],
                     "delta_vs_b5": blk["delta_vs_b5_mean"], "p_vs_b5_max": blk["p_vs_b5_max"],
                     "coverage": blk["conformal_coverage_mean"], "GATE_PASS": blk["GATE_PASS"]})
    if rows:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(FIG_DIR / "table1_gate.csv", index=False)


def main() -> None:
    plt = _mpl()
    fig2_reliability(plt)
    fig3_per_stratum(plt)
    table1_gate(plt)
    print(f"figure_data → {FIG_DIR}" + ("" if plt else "  (matplotlib absent: CSV-only)"))


if __name__ == "__main__":
    main()
