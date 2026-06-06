#!/usr/bin/env bash
# Realness CI for the standalone CASCADE repo. No-demo guard + tests always run;
# the dataset-presence manifest check runs only when the (gitignored, re-fetchable)
# raw data is on disk. No cli/ step (that lives in the SplicR monorepo).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "── 1/3 no-demo guard ─────────────────────────────"
bash scripts/check_no_demo.sh

echo "── 2/3 realness manifest (raw data present?) ─────"
if [ -d data/cascade/raw ]; then
  python scripts/check_realness_manifest.py
else
  echo "  raw data absent (re-fetch via data/acquire.py) — dataset-presence check skipped;"
  echo "  provenance pins remain in data/PROVENANCE.json."
fi

echo "── 3/3 test suite ────────────────────────────────"
python -m pytest tests/ -q

echo "✅ realness CI passed."
