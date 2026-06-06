"""Federated weighted meta-analysis (CASCADE component C2).

Combines per-gene sufficient statistics across screens with provenance-quality
weights and a validity gate (positive-control E-distance), via:

  - a PLAIN path: the inverse-variance random-effects estimator from
    `metaanalysis`, weighted by `provenance.quality_weight`; and
  - a PRIVATE path that proves the FSCP contract end-to-end: each client clips its
    contribution, the two sums (Σ wβ, Σ w) are combined under secure aggregation
    (masks cancel), and Gaussian DP noise calibrated to (ε, δ) is added before the
    server recovers effect = Σwβ / Σw.

Raw counts never appear; only (β, σ², QC, E-dist) per gene do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..metaanalysis import MetaResult, random_effects, fixed_effect
from ..provenance import QCWeightParams, effective_quality
from ..types import ScreenResult
from .dp import analytic_gaussian_sigma, clip_l2
from .secure_agg import SecureAggregator


@dataclass
class GeneStatistic:
    gene: str
    effect: float
    se: float
    variance: float
    tau2: float
    i2: float
    k: int
    total_weight: float

    @classmethod
    def from_meta(cls, gene: str, m: MetaResult) -> "GeneStatistic":
        return cls(
            gene=gene,
            effect=m.effect,
            se=m.se,
            variance=m.variance,
            tau2=m.tau2,
            i2=m.i2,
            k=m.k,
            total_weight=float(m.weights.sum()),
        )


class FederatedMetaAnalysis:
    def __init__(
        self,
        qc_params: QCWeightParams | None = None,
        use_random_effects: bool = True,
        validity_edist: float = 0.05,
    ):
        self.qc_params = qc_params
        self.use_random_effects = use_random_effects
        self.validity_edist = validity_edist

    def collect(self, screens: list[ScreenResult], gene: str):
        """Pull (betas, variances, quality) for `gene` across valid screens."""
        betas, variances, quality = [], [], []
        for s in screens:
            if not s.is_valid(self.validity_edist):
                continue
            ge = s.effects.get(gene)
            if ge is None:
                continue
            betas.append(ge.beta)
            variances.append(ge.variance)
            quality.append(effective_quality(s.qc, self.qc_params))
        return np.array(betas), np.array(variances), np.array(quality)

    def aggregate_gene(self, screens: list[ScreenResult], gene: str) -> MetaResult | None:
        betas, variances, quality = self.collect(screens, gene)
        if betas.size == 0:
            return None
        fn = random_effects if self.use_random_effects else fixed_effect
        return fn(betas, variances, quality)

    def aggregate_all(self, screens: list[ScreenResult]) -> dict[str, MetaResult]:
        genes = sorted({g for s in screens for g in s.genes()})
        out: dict[str, MetaResult] = {}
        for g in genes:
            m = self.aggregate_gene(screens, g)
            if m is not None:
                out[g] = m
        return out

    def private_aggregate_gene(
        self,
        screens: list[ScreenResult],
        gene: str,
        epsilon: float = 3.0,
        delta: float = 1e-6,
        clip_norm: float = 10.0,
        seed: int = 0,
    ) -> dict:
        """Private pooled effect via clip → secure-agg → DP-noise → recover.

        Each valid screen contributes a 2-vector [w·β, w] (w = quality/σ²),
        L2-clipped to `clip_norm` (bounds per-client sensitivity). Secure
        aggregation yields the exact sums; Gaussian noise calibrated to (ε, δ)
        for sensitivity `clip_norm` is then added before recovery.
        """
        contributions: list[np.ndarray] = []
        for s in screens:
            if not s.is_valid(self.validity_edist):
                continue
            ge = s.effects.get(gene)
            if ge is None:
                continue
            w = effective_quality(s.qc, self.qc_params) / ge.variance
            contributions.append(clip_l2(np.array([w * ge.beta, w]), clip_norm))

        n = len(contributions)
        if n == 0:
            return {"gene": gene, "effect": None, "n_clients": 0}

        agg = SecureAggregator(seed=seed)
        secure_sum = agg.run(contributions)  # exact [Σwβ, Σw] (masks cancel)

        sigma = analytic_gaussian_sigma(epsilon, delta, sensitivity=clip_norm)
        rng = np.random.default_rng([seed, 99])
        noisy = secure_sum + rng.normal(0.0, sigma, size=2)
        num, den = float(noisy[0]), float(noisy[1])
        effect = num / den if den != 0 else float("nan")
        return {
            "gene": gene,
            "effect": effect,
            "noisy_numerator": num,
            "noisy_denominator": den,
            "sigma": sigma,
            "epsilon": epsilon,
            "delta": delta,
            "clip_norm": clip_norm,
            "n_clients": n,
        }
