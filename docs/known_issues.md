# Known issues & honest limitations

CASCADE reports what it does *and does not* prove. Every number below is from a
committed result JSON and is regenerable.

## 1. LOSO-study AUROC ≈ 0.41 (worse than chance)
On leave-one-study-out over BioGRID-ORCS, the Oracle scores ~0.41 (5-seed gate;
`results/scientific_gate.json`). **Why:** ORCS hit labels are defined per study
with study-specific thresholds and heterogeneous phenotype readouts (viability,
reporter, proliferation, drug-response). Root-cause analysis
(`results/loso_failure_analysis.json`): per-study AUROC clusters around chance
(fitness 0.52, drug-response 0.50, other 0.48), no single factor explains it
(regression R² = 0.07). Per-study ranking is ~chance while study-level score-scale
heterogeneity pushes the *pooled* AUROC below 0.5. **What it means:** cross-study
replication across incompatible readouts is not biologically expected. **What it
does NOT mean:** institute/context replication is fine (cross_lab 0.862).
**Fix:** filter cross-study pairs to same-phenotype-class; never pool the
cross_study label with institute/context replication.

## 2. cross_study stratum AUROC 0.527 (near chance)
Same cause as #1 — ORCS phenotype heterogeneity + label noise. 223 exact-duplicate
rows + 22 conflicting-label keys were diagnosed in Benchmark v1 (feature-resolution
label noise, not train/test leakage); fixed in the v1.1 freeze (adds a `study`
column + de-duplication).

## 3. High abstention (~40%)
The Oracle abstains when the (A→B) context pair falls outside the observed
cross-lab support manifold (kNN radius) or lacks group-conditional calibration
data. This is by design — it declines rather than guessing. Interpret a returned
probability only when `abstained == false`.

## 4. cross_cell_type modest (AUROC 0.618, n = 213)
Real signal (DeLong p ≈ 0.002, stable across 5 seeds) but small and ceiling-limited:
the transcriptomic cross-cell-type data (Replogle) covers only **two** cell types
(K562 ↔ RPE1). CASCADE is the only benchmarked method that wins here (vs MAIC 0.545,
WBC 0.470), but materially improving it needs context-diverse data (CRISPRbrain,
Tahoe, JUMP/RxRx) not yet ingested.

## 5. Common-essentials ceiling — a simple baseline beats CASCADE on fitness AUROC
Much of "replication" in fitness screens reduces to "is this a common-essential
gene." A one-line information-content meta-analysis (MAIC) exploits this and
**outperforms the CASCADE Oracle on raw AUROC** for overall (0.905 vs 0.849),
cross_lab (0.919 vs 0.862), cross_context (0.912 vs 0.781) and cross_study (0.738
vs 0.527). We report this plainly. CASCADE's distinct value is calibrated
uncertainty + honest abstention + context-pair conditioning (and the cross-cell-type
win), NOT raw discrimination on essential-gene-dominated fitness strata. Harder,
non-essential replication cases are where a conditioned model should matter — and
where better benchmarks are needed.

## 6. Federation is a pilot, not production
FSCP is a 2-party (Broad ↔ Sanger) same-machine simulation: secure-aggregation +
analytic-Gaussian DP (ε ≤ 4, RDP-accounted), only masked DP gradients cross the
boundary. Because both parties screen the same cancer lines, the lift is small by
construction (+0.007). Real network transport, an external party, and a
context-different third party (to demonstrate genuine federation lift) are future
work — not claimed here.
