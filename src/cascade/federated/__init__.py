"""FSCP — the Federated Screen Contribution Protocol (CASCADE C2 / spec §5).

Raw counts/FASTQ never leave the lab. Only masked, DP-protected sufficient
statistics are shared:

  - secure_agg: pairwise additive masking (Bonawitz et al. 2017) so the server
    only ever sees the SUM of client updates, never an individual one.
  - dp: client-level differential privacy (gradient clipping + the analytic
    Gaussian mechanism) with an RDP accountant — closing MELLODDY's documented
    no-DP gap, since genomic FL without DP is membership-inference-attackable.
  - aggregation: the weighted federated meta-analysis that combines per-gene
    sufficient statistics through both the plain and the private (secure-agg +
    DP) paths.
"""

from .secure_agg import SecureAggregator, pairwise_masks
from .dp import (
    RDPAccountant,
    analytic_gaussian_sigma,
    gaussian_delta,
    clip_l2,
    dp_sgd_noisy_mean,
    amplify_by_subsampling,
)
from .aggregation import FederatedMetaAnalysis, GeneStatistic

__all__ = [
    "SecureAggregator",
    "pairwise_masks",
    "RDPAccountant",
    "analytic_gaussian_sigma",
    "gaussian_delta",
    "clip_l2",
    "dp_sgd_noisy_mean",
    "amplify_by_subsampling",
    "FederatedMetaAnalysis",
    "GeneStatistic",
]
