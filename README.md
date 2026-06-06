# CASCADE

[![bioRxiv](https://img.shields.io/badge/bioRxiv-preprint-red)]()
[![PyPI](https://img.shields.io/pypi/v/cascade-oracle)](https://pypi.org/project/cascade-oracle)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Tests](https://github.com/ssatanis/CASCADE/actions/workflows/tests.yml/badge.svg)](https://github.com/ssatanis/CASCADE/actions/workflows/tests.yml)

**CASCADE predicts whether a CRISPR hit observed in one lab will replicate in another — calibrated probability, honest uncertainty, before you spend a year chasing it.**

CASCADE is a federated CRISPR-screen meta-analysis engine plus the **Replication Oracle**: given a screen hit (gene effect in context A) it returns a calibrated P(replicate in context B) with a conformal interval and an *honest abstention* when the context pair falls outside observed support. The provably-correct core is inverse-variance random-effects meta-analysis with provenance/QC weighting; the Oracle is a deliberately shallow head (logistic → isotonic → Mondrian conformal → kNN support gate) over real cross-lab labels.

---

## Quick start

```bash
pip install cascade-oracle
```

```python
from cascade.oracle import ReplicationOracle
oracle, meta = ReplicationOracle.load("artifacts/oracle_v0.pkl")   # trained_on_real_data: true
pred = oracle.predict_pair(pair)        # pair = a ReplicationPair (gene effect in context A)
print(pred.p_replicate, pred.lower, pred.upper, pred.abstained)
```

```bash
cascade meta '{"studies":[{"beta":-1.2,"variance":0.04},{"beta":-0.9,"variance":0.06}]}'
cascade benchmark --leaderboard      # score any model on the frozen Replication Benchmark v1
cascade mcp-serve                    # the splicr-mcp authority server (stdio)
```

---

## Benchmark results

Frozen **Replication Benchmark v1** (gene + cell-line + study-disjoint test set, real DepMap 26Q1 + Sanger + BioGRID-ORCS + Replogle). AUROC, all methods scored by the same harness:

| Method | overall | cross_lab | cross_context | cross_study | cross_cell_type |
|---|---:|---:|---:|---:|---:|
| mean (prior) | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| ridge | 0.758 | 0.562 | 0.599 | 0.511 | 0.612 |
| WBC (Billmann 2023) | 0.596 | 0.431 | 0.459 | 0.646 | 0.470 |
| MAIC (IC meta-analysis) | **0.905** | **0.919** | **0.912** | **0.738** | 0.545 |
| **CASCADE Oracle** | 0.849 | 0.862 | 0.781 | 0.527 | **0.618** |

> **Honest reading (this is the point, not a footnote).** A simple information-content
> meta-analysis (MAIC) — which is essentially an essentiality detector — **beats the
> CASCADE Oracle on raw AUROC for most *fitness* strata.** Much of "replication" in
> fitness screens reduces to "is this a common-essential gene," which a one-line IC score
> captures. CASCADE's distinct, defensible value is: (1) **calibrated probabilities + honest
> abstention** (MAIC/WBC emit uncalibrated raw scores); (2) it is the **only method that
> conditions on the context pair and wins on the hardest stratum, cross-cell-type (0.618)**;
> (3) the replication-prediction framing with distribution-free conformal coverage. CASCADE
> beats WBC on cross_lab (0.862 vs 0.431) and beats ridge overall (0.849 vs 0.758).

The §HONESTY scientific gate (Oracle vs additive B4 + group-prior B5, 5 seeds):
**LOCO-cell-line PASS** (Oracle 0.849 vs 0.71, +0.14 AUROC, p≈0, coverage 0.92);
**LOSO-study FAIL** (0.41, reported plainly — see limitations).

---

## ⚠️ Limitations (read these — see [docs/known_issues.md](docs/known_issues.md))

- **LOSO-study AUROC ≈ 0.41** (worse than chance): cross-study replication over BioGRID-ORCS is near-unpredictable — heterogeneous phenotype readouts and per-study thresholds. Root-cause analysis in `results/loso_failure_analysis.json`.
- **cross_study stratum AUROC 0.527** (near chance) for the same reason; label noise documented.
- **High abstention (~40%)**: the Oracle declines outside the observed support manifold rather than guessing.
- **cross_cell_type n = 213**: real signal (p≈0.002) but modest and ceiling-limited — Replogle gives only two cell types (K562↔RPE1).
- **Common-essentials ceiling**: AUROC on fitness strata is inflated by easy pan-essential genes; a simple IC baseline (MAIC) exploits this and outperforms CASCADE there.
- **Federation is a 2-party same-machine pilot** (FSCP), not production.

---

## Reproduce all results

```bash
pip install cascade-oracle
cascade benchmark --leaderboard     # reproduces the benchmark table from committed result JSONs
# Full regeneration from real data (requires the pinned sources, see Data):
python scripts/compute_wbc_baseline.py
python scripts/compute_maic_baseline.py
python scripts/run_ablation.py
python scripts/run_loso_analysis.py
python scripts/run_biological_interpretation.py
cascade phase3 --seeds 0 1 2 3 4    # frozen-benchmark evidence pack
cascade gate   --seeds 0 1 2 3 4    # the scientific gate
# Expected: cross-lab AUROC 0.862; runtime a few minutes on CPU per step.
```

Every result JSON carries `trained_on_real_data: true` + a provenance hash; `scripts/ci_realness.sh` enforces no fabricated data in product paths.

---

## Data (all real, all pinned)

Every source is recorded in `data/PROVENANCE.json` with url + sha256 + release + license; nothing is substituted with synthetic data. Raw files are **not** committed (large; re-fetchable) — see `data/acquire.py`.

| Source | Release | License |
|---|---|---|
| Broad DepMap (Chronos gene effect) | Public 26Q1 | CC BY 4.0 |
| Sanger Project Score (Chronos) | v2 | CC BY 4.0 |
| BioGRID-ORCS | 2.0.18 (homo sapiens) | MIT / ORCS terms |
| scPerturb / Replogle 2022 (Perturb-seq) | Zenodo 13350497 | CC BY 4.0 |
| DepMap omics + STRING v12 + GO | 26Q1 / v12 / current | CC BY 4.0 |

---

## Citation

```bibtex
@software{cascade2026,
  title  = {CASCADE: a federated CRISPR-screen replication oracle},
  author = {Satani, Sahaj},
  year   = {2026},
  url    = {https://github.com/ssatanis/CASCADE},
  note   = {bioRxiv preprint TODO:DOI}
}
```

Baselines compared: WBC — Billmann et al. 2023, *Cell Systems* (PMID:37201508); MAIC — Baillie lab (github.com/baillielab/maic).

## License

Apache-2.0 — see [LICENSE](LICENSE).
