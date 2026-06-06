"""Realness-invariant enforcement tests.

  - product/runtime code cannot import the synthetic fixture (import guard fires)
  - the Oracle artifact loader refuses anything not flagged trained_on_real_data
"""

import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from cascade.oracle import ReplicationOracle

CASCADE_ROOT = Path(__file__).resolve().parents[1]
SRC = CASCADE_ROOT / "src"
FIXTURES = CASCADE_ROOT / "tests"


def test_product_context_cannot_import_synthetic_fixture():
    """Importing the fixture without a test session must raise (no pytest, no env flag)."""
    code = (
        "import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s');"
        "import importlib;"
        "importlib.import_module('fixtures.synthetic_screens')" % (str(SRC), str(FIXTURES))
    )
    env = {"PATH": "/usr/bin:/bin"}  # deliberately NO CASCADE_ALLOW_SYNTHETIC
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode != 0, "fixture import should fail outside a test session"
    assert "TEST-ONLY" in proc.stderr or "must not be imported" in proc.stderr


def test_artifact_loader_refuses_non_real(tmp_path):
    bad = tmp_path / "fake_oracle.pkl"
    with open(bad, "wb") as f:
        pickle.dump({"oracle": {}, "metadata": {"trained_on_real_data": False}, "format": 1}, f)
    with pytest.raises(ValueError, match="trained_on_real_data"):
        ReplicationOracle.load(bad)


def test_artifact_loader_refuses_missing_flag(tmp_path):
    bad = tmp_path / "no_flag.pkl"
    with open(bad, "wb") as f:
        pickle.dump({"oracle": {}, "metadata": {}, "format": 1}, f)
    with pytest.raises(ValueError):
        ReplicationOracle.load(bad)


def test_save_refuses_without_real_flag():
    from fixtures.synthetic_screens import SyntheticConfig, generate_synthetic_cohort

    cohort = generate_synthetic_cohort(SyntheticConfig(seed=0))
    oracle = ReplicationOracle(alpha=0.1).fit(cohort.replication_pairs(), seed=0)
    with pytest.raises(ValueError, match="trained_on_real_data"):
        oracle.save("/tmp/should_not_write.pkl", metadata={"trained_on_real_data": False})


def test_check_no_demo_guard_passes():
    """The repo-wide no-demo guard must pass (wires the guard into the test suite)."""
    # scripts/ may sit at the repo root (standalone) or one level above the package
    # (monorepo). Find check_no_demo.sh in either layout.
    candidates = [CASCADE_ROOT / "scripts" / "check_no_demo.sh",
                  CASCADE_ROOT.parent / "scripts" / "check_no_demo.sh"]
    script = next((c for c in candidates if c.exists()), candidates[0])
    proc = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                          cwd=str(script.parent.parent))
    assert proc.returncode == 0, f"check_no_demo failed:\n{proc.stdout}\n{proc.stderr}"
