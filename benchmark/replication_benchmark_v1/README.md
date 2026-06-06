# CASCADE Replication Benchmark v1

The first public, versioned benchmark for **cross-lab CRISPR-screen hit
replication**: given a screen hit, predict whether it replicates in another
lab / context, as a calibrated probability with honest abstention.

Built from REAL public data only (no synthetic). Reproducible from `manifest.json`
+ `data/PROVENANCE.json` on a clean checkout.

## Task

Binary `y_rep` per `(gene, context_A → context_B)`: did the hit replicate?
Models output `p_replicate ∈ [0,1]` and may `abstain` (out of support). Scored by
**AUROC**, **ECE** (calibration), and **abstention rate**; conformal coverage is
an Oracle-intrinsic extra. Mean-rate and ridge baselines are always reported.

## Strata (where models break is the point)

| stratum | what it measures | source |
|---|---|---|
| `cross_lab` | institute replication, same cell line | Broad ↔ Sanger (Chronos) |
| `cross_context` | within-lab, different cell line | DepMap (Broad) |
| `cross_study` | different study (HIT concordance) | BioGRID-ORCS 2.0.18 |
| `cross_cell_type` | K562 ↔ RPE1, **transcriptomic** (distinct task) | Replogle 2022 (scPerturb) |

Fitness (KO) and transcriptomic (CRISPRi) are **distinct tasks**, never pooled.

## Split / leakage policy

Test is **DISJOINT from training by cell line AND lineage AND study AND gene** — a
pair is in test if it touches any held-out axis; train only if it touches none.
`manifest.json.leakage_check` asserts cell-line and gene disjointness.

## Files

- `manifest.json` — releases, provenance hash, split policy, leakage check, n/stratum, license
- `test_pairs.csv` — the frozen held-out pairs `{pair_id, gene, contexts, modality, source, task, pair_type, y_rep}`
- `predictions_cascade.csv` — CASCADE's predictions (one row on its own leaderboard)
- `leaderboard.json` — CASCADE + mean + ridge, overall and per stratum

## Submitting

Produce `predictions.csv` with columns `pair_id,p_replicate,abstain` for the
`test_pairs.csv` ids, then:

```bash
cascade benchmark --leaderboard predictions.csv
```

CASCADE is scored by the exact same harness — no special-casing.

## Citation

CASCADE Replication Benchmark v1, derived from DepMap 26Q1 (CC BY 4.0), Sanger
Project Score (CC BY 4.0), BioGRID-ORCS 2.0.18 (MIT terms), and Replogle 2022 via
scPerturb (CC BY 4.0). See `manifest.json` for exact releases + sha256 provenance.
