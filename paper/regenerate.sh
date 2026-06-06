#!/usr/bin/env bash
# Regenerate every CASCADE Frontier figure/table from real data, one command.
# Each step writes a versioned JSON/CSV artifact with a git SHA + provenance hash.
# Requires the pinned venv and the real corpus on disk (data/cascade/raw/).
set -euo pipefail
cd "$(dirname "$0")/.."          # cascade/
PY="${CASCADE_PY:-.venv/bin/python}"

echo "[1/4] Restore real sources (cached if present, sha256-verified)…"
"$PY" data/acquire.py --only core || echo "  (acquire skipped/failed — using cached raw/)"

echo "[2/4] Phase-3 frozen-benchmark evidence (DeLong/BCa/calibration/neg-controls, 5 seeds)…"
"$PY" -m cascade.phase3 --seeds 0 1 2 3 4 --n-boot 2000 --n-perm 2000
#   -> results/phase3_evaluation.json
#   -> paper/figure_data/fig2_reliability.csv, fig3_per_stratum_auroc.csv

echo "[3/4] Scientific gate: Oracle vs B4/B5 under LOCO/LOSO, 5 seeds…"
"$PY" -m cascade.gate --seeds 0 1 2 3 4
#   -> results/scientific_gate.json

echo "[4/4] Figures from the underlying numbers…"
"$PY" -m cascade.paper_figures || echo "  (matplotlib absent — figure_data CSVs are the portable artifact)"

echo "Done. See results/*.json and paper/figure_data/*.csv (each table ships its numbers)."
