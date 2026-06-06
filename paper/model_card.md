# Model Card — CASCADE Replication Oracle (v0)

Following Mitchell et al. (2019). All metrics are real, on held-out data; this
card states limitations as prominently as performance.

## Model details
- **What:** a calibrated classifier predicting P(a CRISPR screen hit replicates in another lab/context A→B), with honest abstention outside the observed support manifold.
- **Architecture (deliberately shallow):** standardize → logistic head → isotonic calibration → Mondrian (group-conditional) conformal interval → k-NN support gate. Per the 2025 evidence (Ahlmann-Eltze et al., Nat Methods) that deep perturbation models fail to beat linear baselines, generalization is sought from feature/embedding geometry, not depth.
- **Two artifacts:** `oracle_v0.pkl` (fitness, KO) and `oracle_crosscelltype_v0.pkl` (transcriptomic, CRISPRi). Tasks are never pooled into one metric.
- **Realness:** every artifact carries `trained_on_real_data: true` + a provenance hash; the loader refuses any artifact not so flagged.

## Training data
- Broad DepMap Public 26Q1 (Chronos gene effect) + Sanger Project Score (Chronos) → institute replication.
- BioGRID-ORCS 2.0.18 → cross-study (HIT concordance).
- scPerturb/Replogle 2022 K562↔RPE1 Perturb-seq → cross-cell-type (transcriptomic).
- All pinned in `data/PROVENANCE.json` (url + sha256 + release + license), 0 unresolved.

## Intended use
- Triage which screen hits to chase before committing wet-lab validation budget.
- Reviewer / reproducibility-risk assessment of published screens.
- NOT for: clinical decisions; claiming a hit will/won't replicate when the model abstains; cross-cell-type extrapolation beyond observed support.

## Performance (frozen Replication Benchmark v1, 5 seeds; DeLong 95% CI)
| Stratum | AUROC | 95% CI | ECE | calibrated? |
|---|---:|:--:|---:|:--:|
| cross_lab (institute) | 0.862 | [0.856, 0.868] | 0.020 | yes |
| cross_context (cell line) | 0.781 | [0.767, 0.794] | 0.040 | yes |
| cross_cell_type (transcriptomic) | 0.618 | [0.545, 0.690] | 0.092 | fair |
| cross_study (ORCS) | 0.527 | [0.513, 0.540] | 0.091 | weak |

## Limitations (read these)
- **Cross-cell-type is modest and ceiling-limited:** Replogle gives only two cell types (K562↔RPE1); the 0.618 is real (p≈0.002) but small with n=213. Improving it needs context-diverse data (CRISPRbrain, Tahoe, JUMP/RxRx) — not yet ingested.
- **cross_study (ORCS) is at chance** due to label noise at the feature resolution (study-pair identity not carried in v0 features).
- **Overall margin over ridge is thin (~0.01):** durable value is in calibration + the replication target baselines cannot access, not in raw point-prediction.
- **Abstention is high (~40%)** by design — the model declines outside support rather than guessing.
- **No demographic axes apply** (cell lines, not people), but **lineage imbalance** exists (cancer-heavy); non-cancer contexts are under-represented.

## Ethical / privacy
- Federation path (FSCP) shares only masked DP gradients (ε≤4, RDP-accounted); raw effects never cross a party boundary. Local-first MCP server keeps data on-device.

## Caveats enforced in product
- The MCP envelope surfaces `validation.beats_baseline` and `calibration.abstained` on every result; the server honest-declines when the corpus/artifact is absent.
