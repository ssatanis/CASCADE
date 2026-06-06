# CASCADE Journal-Ready Report

Five gap-closing scientific additions on existing real data, professional package
restructure, and a GitHub release. Every number is real and regenerable; every new
result carries `trained_on_real_data: true` + a provenance hash. No retraining of
the shipped oracle, no new downloads.

---

## 1. Files created / modified

**New scientific modules** (`src/cascade/`)
- `baselines_wbc.py` — WBC (Billmann 2023) per-gene within/between-lineage concordance baseline, scored on the frozen benchmark.
- `baselines_maic.py` — MAIC information-content meta-analysis baseline (Baillie lab; IC-formula fallback, documented).
- `ablation.py` — 5 nested variants A→E on the oracle_v0 split.
- `loso_analysis.py` — LOSO-study root-cause (per-study AUROC × phenotype/library/quality + regression).
- `biological_interpretation.py` — per-gene P(replicate) ranking, essentiality enrichment, case studies.
- `cli.py` — `cascade benchmark --leaderboard` now works with no argument (prints the committed leaderboard).

**New result files** (`results/`, all real + provenance-hashed)
- `wbc_baseline.json`, `maic_baseline.json`, `ablation.json`, `loso_failure_analysis.json`, `biological_interpretation.json`
- `benchmark/replication_benchmark_v1/leaderboard.json` — WBC + MAIC rows added.

**New figure data** (`paper/figure_data/`)
- `fig_ablation.csv`, `fig_loso_breakdown.csv`, `fig_bio_interpretation.csv`

**New tests** — `tests/test_wbc_baseline.py`, `test_ablation.py`, `test_biological_interpretation.py`

**Run scripts** (`scripts/`) — `compute_wbc_baseline.py`, `compute_maic_baseline.py`, `run_ablation.py`, `run_loso_analysis.py`, `run_biological_interpretation.py`

**Package / publication** — `pyproject.toml` (hatchling, Apache-2.0, src-layout), `README.md`, `LICENSE` (Apache-2.0), `Dockerfile`, `.gitignore`, `.env.example`, `CONTRIBUTION.md`, `MANIFEST.in`, `.github/workflows/tests.yml`, `docs/known_issues.md`, `tests/conftest.py` (skip-if-no-data), `data/PROVENANCE.json`.

---

## 2. WBC vs CASCADE (frozen benchmark AUROC)

| Stratum | WBC (Billmann 2023) | CASCADE | Winner |
|---|---:|---:|:--|
| cross_lab | 0.431 | **0.862** | **CASCADE** (+0.431) |
| overall | 0.596 | **0.849** | **CASCADE** |

WBC is a context-*specificity* contrast — the wrong sign for predicting cross-lab
replication (below chance on cross_lab). **CASCADE beats WBC decisively on cross_lab**
(acceptance criterion ✅). WBC scored 18,441 genes.

## 3. MAIC vs CASCADE (the honest finding)

| Stratum | MAIC (IC meta-analysis) | CASCADE | Winner |
|---|---:|---:|:--|
| overall | **0.905** | 0.849 | MAIC |
| cross_lab | **0.919** | 0.862 | MAIC |
| cross_context | **0.912** | 0.781 | MAIC |
| cross_study | **0.738** | 0.527 | MAIC |
| cross_cell_type | 0.545 | **0.618** | **CASCADE** |

**A simple information-content meta-analysis beats the CASCADE Oracle on raw AUROC for
every *fitness* stratum.** Reported plainly. Much of "replication" in fitness screens is
"is this a common-essential gene," which a one-line IC score captures. CASCADE's distinct
value: it is the **only method that conditions on the context pair and wins on the hardest
stratum (cross-cell-type, 0.618)**, plus calibrated probabilities + honest abstention
(MAIC/WBC emit uncalibrated raw scores). `MAIC_approx` (IC formula) was used because the
`maic` package expects categorised ranked-list files, not a gene-effect matrix —
documented in `maic_baseline.json`.

## 4. Ablation (A→E, same split as oracle_v0, held-out AUROC)

| Variant | AUROC | ECE | coverage | abstention |
|---|---:|---:|---:|---:|
| A logistic only | 0.8601 | 0.063 | — | — |
| B + isotonic | 0.8610 | 0.028 | — | — |
| C + conformal | 0.8610 | 0.028 | 0.928 | — |
| D + provenance weighting | 0.8616 | 0.026 | 0.927 | — |
| **E full CASCADE** | **0.8682** | 0.051 | 0.937 | 0.392 |

**E reproduces oracle_v0.pkl exactly (0.8682 vs 0.8682, within 0.002 ✅).** Isotonic and
conformal are monotone → they do not change AUROC (A≈B≈C); their value is calibration
(ECE 0.063→0.026) and distribution-free coverage. The kNN support gate (E) lifts AUROC by
abstaining on 39% of out-of-support pairs — trading coverage for accuracy. Δ(A→E) = +0.008.

## 5. LOSO dominant failure factor

Pooled LOSO-study AUROC **0.408** (109 studies tested, 37 too small). Per-study AUROC
clusters near chance: fitness 0.519, drug_response 0.496, other 0.482. Per library: TKO
0.520, Brunello 0.504, Avana 0.483, GeCKO 0.483. **Regression R² = 0.071 — no single factor
explains the failure** (nominal dominant = library, but the effect is weak). **The failure is
intrinsic to cross-study label heterogeneity:** per-study ranking is ~chance while
study-level score-scale differences push the pooled AUROC below 0.5. Recommendation: filter
cross-study pairs to same-phenotype-class; never pool the cross_study label with
institute/context replication. (`results/loso_failure_analysis.json`)

## 6. Biological interpretation headline

- **Top replicators (high P):** FAU (0.998), RRM1 (0.996), SNRPD1 (0.996), SNRPA1, PRELID1 — **TOP 20 are 100% common-essential** (ribosomal, spliceosome, nucleotide synthesis). They deplete in every lineage, so they replicate everywhere.
- **Bottom replicators (low P):** ARHGEF5 (0.358), EXOC7 (0.365), ARHGAP11A, LSM14A, KHSRP — Rho-GEF/GAP signaling + exocyst, context-specific (BOTTOM 20 are 50% essential).
- **Enrichment:** common essentials significantly over-represented in TOP 20 (Fisher exact p = 0.0093, Bonferroni-significant; OR → ∞ as all 20 are essential).
- **Finding:** CASCADE assigns high P(replicate) to core essential genes and low P to context-specific genes — consistent with known CRISPR-screen biology. (`results/biological_interpretation.json`)

---

## 7. Gates

| Gate | Result |
|---|:--|
| `scripts/check_no_demo.sh` (no fabricated data in product paths) | ✅ pass |
| Full cascade pytest (monorepo, with real data) | ✅ **123 passed** |
| Standalone pytest (CI mode, data-tests skip) | ✅ 96 passed, 2 skipped |
| New tests (WBC/MAIC/ablation/bio) | ✅ 11 passed |
| `cascade benchmark --leaderboard` on committed JSONs | ✅ runs, exit 0 |
| `pip install -e .` (fresh venv, hatchling) | ✅ exit 0 |
| `.github/workflows/tests.yml` valid YAML | ✅ |
| `docker build` | ⚠️ not executed — Docker daemon not running on this host. Dockerfile is valid and its only real step (`pip install -e .`) is verified in a clean Python 3.12 venv (exit 0). |
| git committer = ssatanis / ss4497@cornell.edu | ✅ (no Claude attribution) |
| No data/ large files committed | ✅ (only PROVENANCE.json + 4.4MB frozen test_pairs.csv) |

## 8. Commit + GitHub

- **Standalone repo commits** (committer `ssatanis <ss4497@cornell.edu>`):
  - `b9fc4de` add WBC/MAIC baselines, ablation, biological interpretation
  - `278afac` merge of the repo's initial scaffolding (kept full tree)
- **Monorepo commits** (worktree `practical-boyd-7e39ed`, also `ssatanis`):
  - `8135de0` add WBC/MAIC baselines, ablation, biological interpretation
  - `6777418` support cascade benchmark --leaderboard with no arg
- **Pushed to:** https://github.com/ssatanis/CASCADE (branch `main`, `c591aaf..278afac`).
- **Zero Claude attribution** anywhere in the pushed history (verified).

> Docker note: `docker build` was **not** run — the Docker daemon is not running on
> this host. The Dockerfile is a trivial 4-line `pip install -e .` whose install step
> is verified to succeed in a clean venv; the image build is expected to succeed once
> the daemon is up but is not claimed as executed here.

---

## What this proves — and what it does not

**Proves:** CASCADE beats WBC and ridge; reproduces oracle_v0 exactly under ablation;
provides calibrated + abstaining + cross-cell-type-winning predictions; and the per-gene
ranking recovers known essential-gene biology (Fisher p = 0.009).

**Does NOT prove (stated plainly):** CASCADE does not have the best raw AUROC — a simple IC
meta-analysis (MAIC) beats it on every fitness stratum; cross-study (LOSO) replication is
near-chance and intrinsic to ORCS label heterogeneity; cross-cell-type is real but modest
(n=213, two cell types). CASCADE's contribution is calibration, honest abstention, and
context-pair conditioning — not raw discrimination on essential-gene-dominated fitness data.
