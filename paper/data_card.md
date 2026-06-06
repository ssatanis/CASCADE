# Data Card / Datasheet — CASCADE replication corpus

Following Gebru et al. (2018). Every source is real and pinned in
`data/PROVENANCE.json` (url + sha256 + bytes + release + access_date + license).

## Motivation
Mint a supervised cross-lab / cross-context replication signal — "will this hit
hold up elsewhere?" — that no single screen contains. The label exists only where
the same perturbation is observed in ≥2 labs/contexts.

## Composition (merged corpus)
| Source | Release | Role | Pairs | License |
|---|---|---|---:|---|
| Broad DepMap | Public 26Q1 (Chronos) | institute A | — | CC BY 4.0 |
| Sanger Project Score | Chronos v2 | institute B | 45,000 cross-lab + 19,930 cross-context | CC BY 4.0 |
| BioGRID-ORCS | 2.0.18 (homo sapiens) | cross-study | 29,777 cross-study + 223 same-cell | MIT/ORCS terms |
| scPerturb / Replogle 2022 | Zenodo 13350497 | cross-cell-type (transcriptomic) | K562↔RPE1 pairs | CC BY 4.0 |

- **Total fitness pairs:** 94,930; base replication rate 0.61; 37,569 non-hits (the denominator most public DBs lack).
- **Modalities:** KO (fitness), CRISPRi (transcriptomic) — kept distinct.

## Collection & harmonization
- Downloaded via `data/acquire.py` (DepMap portal API + Zenodo + BioGRID Spaces), sha256-checksummed, browser-UA + 429 backoff. Unresolvable sources are recorded `status=unresolved`, never substituted with synthetic.
- Gene IDs reconciled (symbol+Entrez); sgRNA→target mapped for raw screens; unmapped counted, not silently dropped.
- Batch/lab/site treated as nuisance; Z-normalized/Chronos versions used where provided.

## Label construction
- `y_rep` = concordance (sign + magnitude / E-distance agreement) of the perturbation's effect across the two contexts, pre-registered in `replication_label_definition` (τ, FDR, sign rule) — never tuned on test.
- Pair tags: {cross_lab, cross_context, cross_study, cross_cell_type} × {KO, CRISPRi}.

## Known issues (honest)
- **ORCS cross_study label noise:** different study-pairs over the same (gene, cell-line pair) can carry contradictory labels; 22 such keys in Benchmark v1 (fixed in v1.1 by adding the `study` column + de-duplication). Caps achievable AUROC for that stratum.
- **Lineage imbalance:** cancer-heavy; non-cancer (neuron/glia/primary-T) absent — the binding limitation on cross-cell-type generalization.
- **Two cell types only** for the transcriptomic cross-cell-type signal (K562, RPE1).

## Splits (frozen, hashed)
- Benchmark v1 test set: disjoint by gene AND cell line AND study AND lineage (`benchmark/replication_benchmark_v1/`), leakage-checked at freeze.
- Gate: group-aware GroupKFold by cell-line (LOCO) / lineage / study (LOSO).

## Maintenance
- Re-acquire: `python data/acquire.py`. Provenance hash changes invalidate the merged-corpus cache automatically.
