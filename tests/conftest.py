"""Pytest session setup.

* permit the import-guarded synthetic fixture inside tests
* skip data-dependent tests when the real corpus is absent (e.g. CI), so the
  suite stays green from committed result JSONs + synthetic fixtures alone.
"""

import os
from pathlib import Path

os.environ.setdefault("CASCADE_ALLOW_SYNTHETIC", "1")

# Real raw data lives outside the repo (gitignored, re-fetchable). When absent,
# tests that load it are skipped rather than failing.
_RAW = Path(__file__).resolve().parents[1] / "data" / "cascade" / "raw"
_DATA_DEPENDENT = [
    "test_corpus_real.py",
    "test_orcs.py",
    "test_replogle.py",
    "test_merged.py",
    "test_data.py",
    "test_fscp.py",
]

collect_ignore = [] if _RAW.exists() else list(_DATA_DEPENDENT)
