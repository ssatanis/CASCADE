"""CASCADE — federated CRISPR-screen meta-analysis + the Replication Oracle.

Design philosophy (load-bearing, from the spec): the 2025 evidence shows deep
perturbation models do not beat linear/mean baselines because of *data*, not
architecture. So CASCADE routes durable value to provably-correct components and
never claims an architecture is the moat:

  - inverse-variance random-effects meta-analysis (C3 core)
  - provenance-quality weighting (C3 multiplier)
  - group-conditional (Mondrian) conformal calibration (C4)
  - the Replication Oracle (C5) with honest abstention outside support
  - federated secure-aggregation + differential privacy (C2 / FSCP)

Every estimator is validated against ground truth via the synthetic cross-lab
screen generator, and against the mean/ridge baselines (the "always-beat-baseline"
gate from Ahlmann-Eltze et al., Nature Methods 2025).
"""

from .types import (
    QCBundle,
    GeneEffect,
    ScreenResult,
    Context,
    ReplicationPair,
    OraclePrediction,
)
from .metaanalysis import (
    fixed_effect,
    random_effects,
    MetaResult,
)
from .provenance import quality_weight, QCWeightParams
from .conformal import MondrianConformalRegressor, AdaptiveConformal
from .baselines import mean_baseline, ridge_baseline, beats_baseline_gate
from .oracle import ReplicationOracle

__version__ = "0.1.0"

__all__ = [
    "QCBundle",
    "GeneEffect",
    "ScreenResult",
    "Context",
    "ReplicationPair",
    "OraclePrediction",
    "fixed_effect",
    "random_effects",
    "MetaResult",
    "quality_weight",
    "QCWeightParams",
    "MondrianConformalRegressor",
    "AdaptiveConformal",
    "mean_baseline",
    "ridge_baseline",
    "beats_baseline_gate",
    "ReplicationOracle",
    "__version__",
]
