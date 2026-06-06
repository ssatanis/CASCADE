# Contributing to CASCADE

Thanks for your interest. CASCADE is real-data-only and reproducibility-first;
contributions must keep both invariants intact.

## Ground rules (non-negotiable)
- **No synthetic/demo/placeholder data in product or validation paths.** Synthetic
  belongs only in `tests/` behind the import guard. `bash scripts/check_no_demo.sh`
  must stay green.
- **Every dataset is pinned** in `data/PROVENANCE.json` (url + sha256 + release +
  license). Unresolvable sources are recorded `status=unresolved`, never substituted.
- **Every result file** carries `trained_on_real_data: true` + a provenance hash.
- **Report what a result does and does not prove.** No overclaiming.

## Add a new baseline to the leaderboard
1. Implement it in `src/cascade/baselines_<name>.py` with a `run()` that scores the
   frozen benchmark via `cascade.benchmark_v1.score_predictions`.
2. Write `results/<name>_baseline.json` (mirror `wbc_baseline.json` schema) and add a
   row to `benchmark/replication_benchmark_v1/leaderboard.json`.
3. Add `tests/test_<name>_baseline.py` checking JSON shape (do not recompute in CI).
4. State honestly whether CASCADE beats it — per stratum.

## Add a new dataset
- Add the fetch + checksum to `data/acquire.py`; record provenance in
  `data/PROVENANCE.json`. Never commit large raw files (they are gitignored).
- Harmonize gene IDs (symbol+Entrez), map sgRNA→target, count unmapped (don't drop
  silently), keep modality/task tags distinct.

## Run tests
```bash
pip install -e ".[dev]"
pytest tests/ -q            # data-dependent tests auto-skip if data/ is absent
bash scripts/ci_realness.sh # realness gate (no-demo + manifest + suites)
```

## Code style
- `black .` and `ruff check .` before opening a PR.

## PR requirements
- Tests pass; `scripts/check_no_demo.sh` green; no synthetic in product paths.
- New numbers are regenerable by one command and committed as JSON.
